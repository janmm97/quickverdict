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
