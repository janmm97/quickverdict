# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
