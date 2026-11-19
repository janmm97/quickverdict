# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Remote backend (`--api-base`): score with any OpenAI-compatible server that returns logprobs
  (vLLM, llama.cpp, Ollama, SGLang, LM Studio, hosted open-model providers), so bigger models on
  another machine can serve the same API. Opt-in; key via `QUICKVERDICT_BACKEND_API_KEY`.
- `--model auto` (the new default): reads the GPU's memory at start-up and loads the biggest Qwen2.5
  model that fits (1.5B, 7B, 14B or 32B; 4-bit on NVIDIA), so bigger cards get a better judge. Several
  GPUs are pooled. The benchmark scripts use it too.
- Auto device detection beyond NVIDIA: Intel Arc (`xpu`) and Apple Silicon (`mps`); AMD works through
  ROCm builds of torch. `--device auto` splits a big model across every GPU.
- README: install lines for RTX 50-series, AMD, Apple and Intel, and rough model sizes per card.

### Changed
- Default dtype on NVIDIA/AMD GPUs is now bf16 where supported (fp16 otherwise), because newer models
  overflow in fp16.
- Requires `torch>=2.7`; the README installs the `cu128` wheel.
- `bitsandbytes` and `accelerate` are now regular dependencies (no `[4bit]` extra needed; it stays as an
  empty alias).

### Fixed
- RTX 50-series (Blackwell) GPUs failed with "no kernel image is available" because the README installed
  the `cu124` torch wheel.

## [0.1.0] - 2026-11-19

### Added
- `quickverdict serve`: a local HTTP server wire-compatible with TypeSafe's `POST /v1/systemone` and
  `GET /v1/models`, so the official TypeSafe SDKs work by pointing `base_url` at it.
- Choice, Score and Noul questions answered from next-token probabilities in a single packed forward
  pass (no text generation), with TypeSafe's confidence formulas.
- Choice answers averaged over cyclic option orderings to cancel position bias (`--permutations`).
- Optional nf4 4-bit loading (`--load-in-4bit`) and `torch.compile` (`--compile`).
- Request hardening: optional bearer API key, 32 MiB body cap, gzip/zstd with a decompression cap,
  JSON depth limit, strict schema validation.
- `bench/clerc_rerank.py`: TypeSafe's published CLERC re-ranking recipe, reproduced against a local model.
- `bench/consistency.py`: TypeSafe's self-consistency recipe (40 posts x 15 repeats) with latency.
- A test that drives the server with the official `typesafe-sdk` client.
- README with quick start, usage, options and benchmark results (RTX 3050 Laptop, Qwen2.5-1.5B).

### Known limits
- On a 4 GB laptop GPU with a 1.5B model, accuracy and latency are behind Jev's published numbers
  (CLERC top-1/5/10 5/22.5/50% vs 30/57/65%; ~0.4 to 1 s per call vs ~100 ms).
- No Box questions or images, no streaming, no multi-GPU. Local use only.
