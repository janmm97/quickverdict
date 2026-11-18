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
