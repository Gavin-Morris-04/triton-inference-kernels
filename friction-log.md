## RunPod MI300X out of capacity on first attempt
- Cross-vendor testing bottlenecked by AMD instance availability.

## MI300X pod #1: deployed with CUDA PyTorch template by default
- torch 2.8+cu128, no rocm-smi in container. Nothing could talk to the GPU.
- Lesson: on AMD instances, check template names for "rocm" before deploying.

## A100: fp32 tl.dot silently uses TF32 by default (Ampere)
- allclose vs cuBLAS failed at tolerances that passed on T4 (no TF32 hardware there).
- Fixed with input_precision="ieee". Portability isn't just "does it run" —
  it's "does it compute the same thing."

## A100 / Triton 3.4: tensor-core prediction CONFIRMED
- Same unchanged kernel: 8 mma / 0 fma on sm_80 vs 0 mma / 512 fma on sm_75 (T4).

## MI300X pod #2: environment hell on stale ROCm template
- Template shipped torch 2.4.0.dev+rocm6.0 (May 2024 snapshot!) with no triton.
- pip triton 3.7.1 → AMD backend init failure: "memory mapped libamdhip64.so
  does not point to a valid lib". Version matching is load-bearing on AMD.
- pip pytorch-triton-rocm silently pulled CUDA torch 2.13 as a dependency.
- Cleanup attempts left cu-torch shadowing rocm-torch. Terminated pod.
- Verdict: on AMD, torch+triton must ship pre-matched in the image;
  pip surgery on a metered instance loses.
