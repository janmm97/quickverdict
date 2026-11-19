import math

import pytest
import torch

from quickverdict.engine import RequestError, _orderings
from quickverdict.schema import SystemOneRequest

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "Topic?",
              "criteria": {"billing": "charges", "shipping": None, "account": {"about": "login"}}},
    "many": {"type": "choice", "criteria": {f"opt{i}": None for i in range(30)}},  # multi-token keys
    "anger": {"type": "score", "criteria": ["calm", "annoyed", "furious"]},
    "refund": {"type": "noul", "instructions": "Owed a refund?", "criteria": {"true": "double charge"}},
}


def req(questions=QUESTIONS, state="I was charged twice for order 1234."):
    return SystemOneRequest.model_validate({"model": "m", "state": state, "questions": questions})


def naive_scores(engine, prefix, items):
    """Reference: every candidate scored as its own full sequence, no cache, no packing."""
    out = []
    with torch.inference_mode():
        for it in items:
            row = []
            for c in it.cands:
                targets = c if len(c) == 1 and all(len(x) == 1 for x in it.cands) else c + [engine.end]
                seq = prefix + it.suffix + targets
                logits = engine.head(engine.body(input_ids=torch.tensor([seq])).last_hidden_state[0]).float()
                lp = logits.log_softmax(-1)
                start = len(prefix) + len(it.suffix) - 1
                row.append(sum(lp[start + j, t].item() for j, t in enumerate(targets)))
            out.append(row)
    return out


# one packed pass / long-state prefill, then one pass per item / whichever mix 120 tokens gives
@pytest.mark.parametrize("chunk", [10**6, 1, 120])
def test_packed_matches_naive(engine, chunk):
    r = req()
    prefix = engine._ids(engine._pre + r.state + engine._mid)
    items = engine._items(r.questions)
    engine.chunk_tokens = chunk
    try:
        got, used = engine._scores(prefix, items)
    finally:
        engine.chunk_tokens = 2048
    want = naive_scores(engine, prefix, items)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=1e-3)
    assert used > len(prefix)


def test_answer_shapes(engine):
    r = req()
    out = engine.system_one(r.state, r.questions)
    a = out["answers"]
    assert set(a) == set(QUESTIONS)
    assert a["topic"]["type"] == "choice" and a["topic"]["choice"] in QUESTIONS["topic"]["criteria"]
    assert math.isclose(sum(a["topic"]["probabilities"].values()), 1, abs_tol=1e-6)
    assert len(a["many"]["probabilities"]) == 30
    s = a["anger"]
    assert set(s["legend"]) == set(s["probabilities"]) == {"0", "1", "2"}
    assert s["score"] == pytest.approx(sum(int(k) * v for k, v in s["probabilities"].items()))
    assert 0 <= s["confidence"] <= 1 and 0 <= a["refund"]["noul"] <= 1
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0


def test_deterministic(engine):
    r = req()
    assert engine.system_one(r.state, r.questions) == engine.system_one(r.state, r.questions)


def test_choice_permutation_average_is_order_free(engine):
    """With every rotation averaged, listing the options in another order gives the same probabilities."""
    engine.permutations = 3
    a = req({"q": {"type": "choice", "criteria": {"x": None, "y": None, "z": None}}})
    b = req({"q": {"type": "choice", "criteria": {"y": None, "z": None, "x": None}}})
    pa = engine.system_one(a.state, a.questions)["answers"]["q"]["probabilities"]
    pb = engine.system_one(b.state, b.questions)["answers"]["q"]["probabilities"]
    for k in pa:
        assert pa[k] == pytest.approx(pb[k], abs=1e-4)


def test_orderings():
    assert _orderings(4, 1) == [[0, 1, 2, 3]]
    rots = _orderings(4, 4)
    assert len(rots) == 4 and all(sorted(o) == [0, 1, 2, 3] for o in rots)
    assert {o[0] for o in rots} == {0, 1, 2, 3}  # every option gets the first slot
    assert len(_orderings(2, 9)) == 2


def test_box_and_long_rejected(engine):
    with pytest.raises(RequestError):
        r = req({"b": {"type": "box", "instructions": "find it"}})
        engine.system_one(r.state, r.questions)
    old, engine.max_len = engine.max_len, 20
    try:
        with pytest.raises(RequestError, match="max_tokens_exceeded"):
            r = req()
            engine.system_one(r.state, r.questions)
    finally:
        engine.max_len = old


def test_default_device_and_dtype():
    import torch

    from quickverdict.engine import default_device, default_dtype

    assert default_device() in ("cuda", "xpu", "mps", "cpu")
    assert default_dtype("cpu") == torch.float32
    assert default_dtype("mps") == torch.float16
    want = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) \
        else torch.float16
    if torch.cuda.is_available():
        assert default_dtype("cuda:0") == want


def test_pick_model_scales_with_vram():
    from quickverdict.hardware import SMALLEST, pick_model

    assert pick_model(4, True) == (SMALLEST, False)  # RTX 3050 Laptop
    assert pick_model(8, True) == ("Qwen/Qwen2.5-7B-Instruct", True)
    assert pick_model(12, True) == ("Qwen/Qwen2.5-14B-Instruct", True)
    assert pick_model(24, True) == ("Qwen/Qwen2.5-32B-Instruct", True)
    assert pick_model(31.8, True) == ("Qwen/Qwen2.5-32B-Instruct", True)  # RTX 5090
    assert pick_model(80, True) == ("Qwen/Qwen2.5-32B-Instruct", False)
    assert pick_model(24, False) == ("Qwen/Qwen2.5-7B-Instruct", False)  # no bitsandbytes (AMD, Apple, Intel)
    assert pick_model(8, False) == (SMALLEST, False)


def test_auto_model_on_cpu():
    from quickverdict.hardware import SMALLEST, auto_model

    assert auto_model("cpu")[:3] == (SMALLEST, "cpu", False)


def test_default_dtype_skips_emulated_bf16(monkeypatch):
    import torch

    from quickverdict.engine import default_dtype

    seen = {}

    def fake(including_emulation=True):
        seen["emulation"] = including_emulation
        return including_emulation  # a T4: bf16 only through emulation

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", fake)
    assert default_dtype("cuda") == torch.float16 and seen["emulation"] is False


def _gpus(monkeypatch, *gibs):
    """Pretend this machine has CUDA GPUs with the given memory."""
    import types

    import torch

    from quickverdict import hardware

    props = [types.SimpleNamespace(name=f"GPU{i}", total_memory=int(g * hardware.GIB)) for i, g in enumerate(gibs)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props[i if isinstance(i, int) else 0])


def test_detect_pools_cuda_gpus(monkeypatch):
    from quickverdict.hardware import detect

    _gpus(monkeypatch, 12, 12)
    assert detect() == ("auto", 24.0, "GPU0 + GPU1")
    assert detect("auto")[0] == "auto"
    assert detect("cuda:1")[:2] == ("cuda:1", 12.0)  # an explicit device is never pooled
    _gpus(monkeypatch, 8)
    assert detect("auto") == ("cuda", 8.0, "GPU0")  # one GPU: plain cuda


def test_detect_without_gpu(monkeypatch):
    import torch

    from quickverdict.hardware import detect

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch, "xpu", type("X", (), {"is_available": staticmethod(lambda: False)}), raising=False)
    assert detect() == ("cpu", 0.0, "cpu")
    assert detect("cuda")[1] == 0.0  # asked for a GPU we cannot see: smallest model, the load then fails loudly


def test_can_load_4bit_only_on_nvidia(monkeypatch):
    import torch

    from quickverdict import hardware

    monkeypatch.setattr(torch.version, "hip", None)
    assert hardware.can_load_4bit("cuda") and hardware.can_load_4bit("cuda:1") and hardware.can_load_4bit("auto")
    assert not hardware.can_load_4bit("mps") and not hardware.can_load_4bit("xpu")
    monkeypatch.setattr(torch.version, "hip", "6.3")  # ROCm builds report device "cuda" too
    assert not hardware.can_load_4bit("cuda")
