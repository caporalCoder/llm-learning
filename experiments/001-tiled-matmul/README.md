# 001 — Tiled matmul vs naive (CUDA)

**Date:** 2026-__-__ · **Hardware:** _e.g. A100 80GB, CUDA 12.x_ · **Status:** ⬜

> A worked **example** so the repo isn't empty and you can see the shape of a good entry.
> Replace the placeholder numbers with your real ones.

## Goal
How close to cuBLAS can a hand-written tiled (shared-memory) SGEMM get, and where does the gap come from?

## Hypothesis
Naive matmul is memory-bound — every output element re-reads a full row/column from global memory. Tiling into shared memory should cut global traffic by the tile width and move us toward compute-bound, closing most of the gap to cuBLAS (which additionally uses tensor cores + register blocking I won't match by hand).

## Method
- `naive.cu` — one thread per output element, reads from global memory.
- `tiled.cu` — 32×32 shared-memory tiles.
- Baseline: cuBLAS `cublasSgemm`.
- Measured with CUDA events, 20 warmup + 100 timed iters, square matrices N=4096, fp32.

## Reproduce
```bash
nvcc -O3 -arch=sm_80 naive.cu -o naive && ./naive 4096
nvcc -O3 -arch=sm_80 tiled.cu -lcublas -o tiled && ./tiled 4096
```

## Results
| Variant | GFLOP/s | vs naive | % of cuBLAS |
|---------|--------:|:--------:|:-----------:|
| naive | _e.g. 210_ | 1.0× | _~2%_ |
| tiled (mine) | _e.g. 1,730_ | _8.2×_ | _~18%_ |
| cuBLAS | _e.g. 9,600_ | _46×_ | 100% |

## Why (the analysis)
_Fill in after running: what Nsight Compute showed — achieved occupancy, DRAM throughput vs peak, whether tiling actually moved you off the memory roof, and what the remaining gap to cuBLAS is (tensor cores, register tiling, bank conflicts)._

## Takeaway
> A tiled SGEMM hit ~8× over naive; profiling showed the win was cutting DRAM traffic, and the residual gap to cuBLAS is tensor-core + register blocking.
