# nanoserve — a small LLM inference engine

**Status:** ⬜ planned · **Why:** the best way to understand production serving is to build a tiny version and benchmark it honestly against a real one (vLLM).

> A design doc I fill in *before* coding, then turn into a results writeup.

## Goal metrics
What "good" means, decided up front:
- **TTFT** (time to first token) under load
- **Throughput** (tokens/sec) at a fixed latency SLO
- **Goodput** (requests meeting SLO) as concurrency rises
- Reference to compare against: **vLLM** on identical hardware

## Milestones
- [ ] v0 — single-request generation with a **KV cache**; correctness vs HF `generate()`
- [ ] v1 — **continuous batching** scheduler + token streaming (SSE)
- [ ] v2 — **paged KV cache** (block tables); measure fragmentation win
- [ ] v3 — **prefix caching** + chunked prefill; TTFT win on shared prompts
- [ ] v4 — **speculative decoding** with a draft model; report acceptance rate
- [ ] v5 — **quantized** (INT8/FP8) path; full benchmark sweep

## Architecture
_Diagram + component list once v0 exists (scheduler, KV manager, model runner, API layer)._

## Benchmarks
| Version | TTFT (ms) | Tokens/s | Goodput @ SLO | vs vLLM | Notes |
|---------|----------:|---------:|:-------------:|:-------:|-------|
| v0 | | | | — | |

## Log
- _running notes of what worked, what broke, what surprised me_
