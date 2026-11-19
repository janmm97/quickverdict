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
- A GPU is recommended: NVIDIA (RTX 20 to 50-series), AMD (ROCm), Apple Silicon or Intel Arc. Measured on an RTX 3050 Laptop, 4 GB. CPU works, slowly.
- Any VRAM: quickverdict checks your GPU at start-up and loads the biggest model that fits (see "Your GPU picks the model"). A 4 GB card runs `Qwen/Qwen2.5-1.5B-Instruct`.

## Quick start

```bash
git clone https://github.com/janmm97/quickverdict
cd quickverdict
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # NVIDIA, incl. RTX 50-series; see "Other GPUs"
pip install -e .
quickverdict serve                                 # http://127.0.0.1:8000
```

The first start downloads the model from Hugging Face.

### Other GPUs

| GPU | Install torch with | Notes |
|---|---|---|
| NVIDIA RTX 20/30/40/**50** | `--index-url https://download.pytorch.org/whl/cu128` | RTX 50-series (Blackwell) needs torch 2.7+ with CUDA 12.8; older `cu124` wheels fail with "no kernel image is available" |
| AMD Radeon (Linux) | `--index-url https://download.pytorch.org/whl/rocm6.3` | ROCm reports the GPU as `cuda`, so nothing else changes. `--load-in-4bit` support is limited |
| Apple Silicon | plain `pip install torch` | Uses `mps` automatically |
| Intel Arc | `--index-url https://download.pytorch.org/whl/xpu` | Uses `xpu` automatically |
| CPU only | `--index-url https://download.pytorch.org/whl/cpu` | Works, slowly |

With `--model auto`, several GPUs are pooled automatically: their memory is added up and the model is split
across them (untested with 4-bit). With an explicit `--model`, only the first GPU is used unless you pass `--device auto`.
The first run downloads the model (about 3 GB for 1.5B, 15 GB for 7B, 30 GB for 14B, 66 GB for 32B) and prints the size.

### Your GPU picks the model

With the default `--model auto`, quickverdict reads your GPU's memory at start-up, prints what it chose
(for example `auto: NVIDIA GeForce RTX 5090 (32 GiB), using Qwen/Qwen2.5-32B-Instruct in 4-bit`) and loads
the biggest model that fits. A bigger model judges better, so a bigger card gives better answers with no
extra setup.

| GPU memory (all cards together) | Example cards | Model it loads |
|---|---|---|
| under 7 GB, or CPU | RTX 3050/4050 Laptop, RTX 2060 | Qwen2.5-1.5B (bf16) |
| 7–11 GB | RTX 3060 Ti, 4060, 5060, 3080 10 GB | Qwen2.5-7B (4-bit) |
| 12–21 GB | RTX 3060 12 GB, 4070, 4080, 5070 Ti, 5080 | Qwen2.5-14B (4-bit) |
| 23–71 GB | RTX 3090, 4090, **5090**, two 16 GB cards | Qwen2.5-32B (4-bit) |
| 72 GB and up | A100/H100 80 GB, several big cards | Qwen2.5-32B (bf16) |

4-bit loading needs an NVIDIA GPU. On AMD, Apple Silicon and Intel the same table applies with bf16
models only (e.g. 7B from 17 GB, 14B from 33 GB, 32B from 72 GB). Pass `--model <id>` to choose
yourself. The accuracy of the larger models against Jev has not been measured yet: the numbers under
"Benchmarks" are for the 1.5B model on a 4 GB card.

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

### Bigger models on another machine (remote backend)

quickverdict can score with any OpenAI-compatible server that returns `logprobs` instead of loading a model
itself. Use it to run a 32B or 70B model on a stronger box (vLLM, llama.cpp `llama-server`, Ollama,
SGLang, LM Studio) or with a hosted open-model provider that supports logprobs:

```bash
# e.g. vLLM on a 2x RTX 5090 machine: vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2
quickverdict serve --api-base http://gpu-box:8000/v1 --model Qwen/Qwen2.5-72B-Instruct-AWQ
# hosted provider: put its key in the environment, never on the command line
export QUICKVERDICT_BACKEND_API_KEY=...
```

Same prompts, same answer shapes. Each option's probability comes from the backend's first-token
`top_logprobs` (top 20). Answers stay deterministic when the backend is (temperature 0). The trade-offs:
one request per question and ordering instead of one packed pass, and **the state leaves your
machine** if the backend is not local. Closed APIs that don't return logprobs (Claude, for example)
can't be used: the method needs the model's probabilities, not its text.

Limits of the remote backend:
- Choice questions can have at most 26 options and Score questions at most 10 levels (each answer must be one token).
- The model must reply with the bare key. Turn thinking/reasoning mode off, or quickverdict answers 502.
- `--device`, `--dtype`, `--load-in-4bit`, `--chunk-tokens` and `--compile` are ignored.
- The install still needs `torch` for now.
- Backend failures (unreachable, 401, 5xx, no logprobs) return **502**. `--concurrency` caps the calls in flight across all requests.

### Options

| Flag / env var | Default | What it does |
|---|---|---|
| `--model` / `QUICKVERDICT_MODEL` | `auto` | `auto` loads the biggest model your GPU fits; or a Hugging Face model id or local path |
| `QUICKVERDICT_API_KEY` | empty | If set, requests must send `Authorization: Bearer <key>` |
| `--host`, `--port` | `127.0.0.1`, `8000` | Where to listen. Keep it on localhost unless you set an API key |
| `--device`, `--dtype` | first of `cuda`/`xpu`/`mps` found, else `cpu` (with `--model auto`, all CUDA GPUs together); bf16 on NVIDIA GPUs with native bf16 (RTX 30-series and newer), else fp16 | Where and how to run the model. `--device auto` splits the model across all GPUs |
| `--permutations` | `3` | Option orderings averaged per Choice (1 = off) |
| `--temperature` | `1.0` | Calibration temperature (must be > 0) |
| `--chunk-tokens` | `2048` | Max new tokens per forward pass. Lower it if a long state runs out of GPU memory |
| `--load-in-4bit` | off (`--model auto` turns it on when needed) | nf4 weights, about 4x less VRAM (NVIDIA only) |
| `--compile` | off | `torch.compile`: faster calls, slow first request |
| `--api-base` / `QUICKVERDICT_API_BASE` | empty | Score with an OpenAI-compatible server instead of a local model; `--model` is then its model name |
| `QUICKVERDICT_BACKEND_API_KEY` | empty | Bearer key sent to `--api-base` |
| `--concurrency` | `8` | Parallel requests to `--api-base` |

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
| Remote model backend (OpenAI-compatible, logprobs) | yes, opt-in (`--api-base`) | n/a |

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
python bench/clerc_rerank.py --out clerc.json   # --model auto by default: your GPU's model
python bench/consistency.py  --out consistency.json
```

Please include your GPU and model when you share results.

## Contributing

Issues and pull requests are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately, see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE). quickverdict is an independent project and is not affiliated with TypeSafe.
"TypeSafe" and "Jev" are used only to describe API compatibility.
