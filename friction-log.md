# Friction Log — Triton Inference Kernels Project

A running log of everything that confused, surprised, or broke — kept as
input for future upstream issues and as an honest record of what
cross-vendor GPU work is actually like (mid-2026).

## Compiler / codegen

### T4 / Triton 3.6: FP16 tl.dot never lowers to tensor cores (sm_75)
- Symptom: fp16 GEMM ≈ fp32 speed; 8x behind cuBLAS fp16.
- Diagnosis: kernel.warmup() → asm["ptx"]: 0 mma instructions, all fma.
- Isolation: minimal single-tile mask-free tl.dot also 0 mma → not our kernel.
- Hardware capable: compute cap (7,5); cuBLAS fp16 = 7 ms proves tensor cores work.
- Prediction: same code on A100 should emit mma and show large fp16 gain.

### A100 / Triton 3.4: tensor-core prediction CONFIRMED
- Same unchanged kernel: 8 mma / 0 fma on sm_80 vs 0 mma / 512 fma on sm_75.
- fp16 GEMM went from ~2.4 TFLOP/s (T4) to ~73 TFLOP/s (A100), default config.

### A100: fp32 tl.dot silently uses TF32 by default (Ampere)
- allclose vs cuBLAS failed at tolerances that passed on T4 (no TF32 hw there).
- Fix: input_precision="ieee" in tl.dot. In IEEE mode: 96% of cuBLAS fp32.
- Lesson: portability isn't just "does it run" — it's "does it compute the
  same thing."

## Performance / tuning

### FP16 tl.dot needs num_warps=8 on T4; default 4 is catastrophically slow (~20x)

### 128/128/32 config exceeds T4 shared memory: 1659 ms (thrashing), no error raised

### Single-block-per-row softmax stops scaling around 8K columns (T4)
- BLOCK_SIZE = whole row → register pressure → occupancy drop; torch's
  split-row implementation overtakes ours at n_cols ≥ 8192 (fp16).
- Much milder on A100 (larger register file).

## Environment / infrastructure

### Colab free tier: runtime restarted mid-session, wiped all state
- Re-run defining cells; keep CSVs downloaded; push to GitHub every session.

### RunPod MI300X: out of capacity on first attempt
- Cross-vendor testing bottlenecked by AMD instance availability.

### MI300X pod #2: deployed with CUDA PyTorch template
- torch 2.8+cu128, no rocm-smi in container; nothing could address the GPU.
- Lesson: on AMD instances, check template names for "rocm" before deploying.

### MI300X pod #2 (cont.): environment surgery on a stale ROCm template
- Template shipped torch 2.4.0.dev+rocm6.0 (May 2024 snapshot) with NO triton.
- pip triton 3.7.1 → AMD backend init failure: "memory mapped libamdhip64.so
  does not point to a valid lib". torch/triton version matching is
  load-bearing on AMD.
- pip pytorch-triton-rocm silently pulled CUDA torch 2.13 as a dependency;
  cleanup left cu-torch shadowing rocm-torch. Terminated.
- Verdict: pip surgery on a metered instance loses.

### MI300X pod #3: AMD-official rocm/pytorch image (rocm7.1 / torch 2.10)
- Modern matched stack, but: no Jupyter, no RunPod web terminal, SSH key
  injection failed even after pod restart. Bare vendor images assume
  infrastructure that wasn't there. Terminated at budget cap.

### MI300X pod #4: official RunPod PyTorch 2.4/ROCm 6.1 template — no triton
- Official ROCm template ships torch without any triton module.

### Bottom line, AMD attempts: 4/4 failed on environment, 0/4 on kernel code.
### NVIDIA pods: 2/2 worked within minutes. The kernels were never the problem.
