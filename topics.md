# Topics I'm working through

An informal path through the LLM performance stack — not a schedule, just the order that makes sense to me. Build-first, a few hours a week.

| Area | What I'm learning | Where it lives |
|------|-------------------|----------------|
| Foundations | DL from scratch, transformers cold, FLOP/memory math | notes/, experiments/ |
| GPU architecture | memory hierarchy, tensor cores, roofline reasoning | notes/gpu-architecture |
| Kernels | CUDA → Triton (→ Pallas on TPU); fused ops, tiling | notes/kernels, experiments/ |
| Attention | FlashAttention from first principles | experiments/ |
| Inference systems | KV cache, paged attention, continuous batching, speculative decoding, quantization | projects/nanoserve |
| Distributed training | data / tensor / pipeline parallelism, ZeRO/FSDP, collectives | notes/distributed-training |
| Compilers & TPU | torch.compile / XLA, JAX / Pallas | notes/ |

Curiosity-driven and always in progress. If something here looks interesting, the write-ups in `notes/` and `experiments/` have the details.
