import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = x_ptr + pid * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(row_start + offsets, mask=mask, other=0.0)
    mean_sq = tl.sum(row * row, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
    result = row * rstd * w
    tl.store(out_ptr + pid * stride + offsets, result, mask=mask)


def rmsnorm_ref(x, w, eps=1e-6):
    """Unfused PyTorch reference implementation."""
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps) * w


def rmsnorm(x, w, eps=1e-6):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    rmsnorm_kernel[(n_rows,)](x, w, out, x.stride(0), n_cols, eps, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    for shape in [(128, 128), (500, 781), (32, 4096), (2048, 1000)]:
        t = torch.randn(*shape, device="cuda")
        w = torch.randn(shape[1], device="cuda")
        assert torch.allclose(rmsnorm(t, w), rmsnorm_ref(t, w), atol=1e-5), shape
    print("rmsnorm: all shapes pass")