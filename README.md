# quickverdict

A local, open-source System One decision server. Ask a small language model on your own GPU
structured questions (Choice, Score, Yes/No) and get **calibrated probabilities** back, through an
API that is wire-compatible with [TypeSafe's System One (Jev)](https://docs.typesafe.ai/introduction).

- **Local and private:** your data never leaves the machine. No API key, no per-call bill.
- **Deterministic:** the same request always gives the same probabilities (0% run-to-run change).
- **Order-debiased:** Choice answers are averaged over several option orderings.
- **Drop-in:** the official `typesafe-sdk` client works unchanged, you only set `base_url`.
- **Any model:** any Hugging Face causal LM. Bigger GPU, bigger model, better answers.

quickverdict does not replace Jev. On a 4 GB laptop GPU with a 1.5B model it is less accurate and
slower than Jev (see [Benchmarks](#benchmarks)). Use it when privacy, cost or reproducibility matter
more than top accuracy, or when you can run a larger model.

## Requirements

- Python 3.10+
- An NVIDIA GPU is recommended (measured on an RTX 3050 Laptop, 4 GB). CPU works, slowly.
- About 3.5 GB of VRAM for the default model (`Qwen/Qwen2.5-1.5B-Instruct`, fp16), or less with `--load-in-4bit`.

## Quick start

```bash
git clone https://github.com/janmm97/quickverdict
cd quickverdict
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheel
pip install -e .
quickverdict serve                                 # http://127.0.0.1:8000
```

The first start downloads the model from Hugging Face.

## Usage

### With the official TypeSafe SDK

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(api_key="local", model="quickverdict", base_url="http://127.0.0.1:8000") as client:
    r = client.system_one(
        {"ticket": "My card was charged twice."},
        {
            "topic": Choice(criteria={"billing": None, "shipping": None, "other": None}),
            "urgency": Score(criteria=["low", "medium", "high"]),
            "refund": Noul(instructions="Does the customer want money back?"),
        },
    )
print(r.answers["topic"].choice, r.answers["topic"].probabilities)
```

### Over plain HTTP

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H "content-type: application/json" -d '{
  "model": "quickverdict",
  "state": "My card was charged twice.",
  "questions": {"refund": {"type": "noul", "instructions": "Does the customer want money back?"}}
}'
```

### From Python, no server

```python
from quickverdict.engine import Engine
from quickverdict.schema import SystemOneRequest

engine = Engine("Qwen/Qwen2.5-1.5B-Instruct")
qs = SystemOneRequest.model_validate({"model": "m", "state": "", "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer want money back?"}}}).questions
print(engine.system_one("My card was charged twice.", qs)["answers"])
```

### Options

| Flag / env var | Default | What it does |
|---|---|---|
| `--model` / `QUICKVERDICT_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model id or local path |
| `QUICKVERDICT_API_KEY` | empty | If set, requests must send `Authorization: Bearer <key>` |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where to listen. Keep it on localhost unless you set an API key |
| `--device`, `--dtype` | `cuda` if available, fp16 on GPU | Where and how to run the model |
| `--permutations` | `3` | Option orderings averaged per Choice (1 = off) |
| `--temperature` | `1.0` | Calibration temperature (must be > 0) |
| `--chunk-tokens` | `2048` | Max new tokens per forward pass. Lower it if a long state runs out of GPU memory |
| `--load-in-4bit` | off | nf4 weights, about 4x less VRAM (`pip install -e ".[4bit]"`) |
| `--compile` | off | `torch.compile`: faster calls, slow first request |

See `.env.example` for the variables. quickverdict reads them from the environment and does not load
`.env` itself. Never commit a real `.env`.

## What it supports

| | quickverdict v0.1 | Jev |
|---|---|---|
| `POST /v1/systemone`, `GET /v1/models` | yes | yes |
| Choice (1 to 255 options), Score (1 to 10 levels), Noul | yes | yes |
| Same answer shapes, confidence formulas and `{"detail": ...}` errors | yes | yes |
| gzip / zstd request bodies | yes | yes |
| Box questions and images | no (roadmap) | yes |
| Hosted service, rate limits | no, local only (optional bearer key) | yes |

## How it works

No text is generated. The state is encoded once, and every question is batched against it in a
single packed forward pass. Each option's probability comes from the model's next-token logits,
restricted to the allowed labels (multi-token labels are scored by sequence log-likelihood).
Choice questions are scored under several cyclic option orderings and averaged to cancel position bias.

## Benchmarks

Both recipes are reproduced exactly from TypeSafe's own cookbooks, so the numbers sit next to Jev's
published ones. Measured 2026-11-19 on an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)** with
`Qwen/Qwen2.5-1.5B-Instruct`, fp16, `--permutations 3`.

**CLERC legal re-ranking** ([cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe)): 40 queries,
BM25 shortlist of 30, one Noul per candidate, 1,200 calls.

| | Top-1 | Top-5 | Top-10 | Median latency per call |
|---|---|---|---|---|
| BM25 alone | 5% | 15% | 37.5% | |
| quickverdict, Qwen2.5-1.5B | 5% | 22.5% | 50% | 429 ms (p90 519 ms) |
| Jev (published) | 30% | 57% | 65% | ~100 ms |

Qwen3-1.7B scored 0% / 12.5% / 50% at 517 ms median on the same run.

**Self-consistency** ([cookbook](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook)):
40 moderation posts, 8 Choice questions (37 options), each post sent 15 times, 600 calls.

| | All options | Avg post, least stable option | Worst post | Posts above 1 in 10 | Median latency per call |
|---|---|---|---|---|---|
| quickverdict, Qwen2.5-1.5B | 0% | 0% | 0% | 0 / 40 | 996 ms (p90 1,037 ms) |
| Jev 1.14 (published) | 1.0% | 4.0% | 6.7% | 0 / 40 | ~103 ms |

quickverdict's 0% is by design: nothing is sampled, so the same request always returns the same
probabilities. It shows reproducibility, not better judgement.

What this means: quickverdict is fully reproducible and beats BM25 alone, but on this hardware a
1.5B model is well behind Jev on accuracy and about 4 to 10 times slower per call. Larger models and
GPUs should close part of that gap; we will publish those numbers as we measure them.

Reproduce:

```bash
pip install -e ".[bench]"
python bench/clerc_rerank.py --model Qwen/Qwen2.5-1.5B-Instruct --out clerc.json
python bench/consistency.py  --model Qwen/Qwen2.5-1.5B-Instruct --out consistency.json
```

Please include your GPU and model when you share results.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately, see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE). quickverdict is an independent project and is not affiliated with TypeSafe.
"TypeSafe" and "Jev" are used only to describe API compatibility.
