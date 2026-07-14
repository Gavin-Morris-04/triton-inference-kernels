# Friction Log — Triton kernels project

## T4 / Triton 3.6: FP16 tl.dot never lowers to tensor cores (sm_75)
- Symptom: fp16 GEMM ≈ fp32 speed; 8x behind cuBLAS fp16.
- Diagnosis: kernel.warmup() → asm["ptx"]: 0 mma instructions, all fma.
- Isolation: minimal single-tile mask-free tl.dot also 0 mma → not our kernel.
- Hardware capable: compute cap (7,5); cuBLAS fp16 = 7ms proves tensor cores work.
- Prediction to test on A100: same code should emit mma and show large fp16 gain.

## FP16 tl.dot needs num_warps=8 on T4; default 4 is catastrophically slow (~20x)

## 128/128/32 config exceeds T4 shared memory: 1659ms (thrashing), no error raised

## Colab free tier: runtime restarted mid-session, wiped all state. Re-run defining cells; keep CSVs downloaded.
