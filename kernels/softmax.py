import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, stride, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = x_ptr + pid * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row = tl.load(row_start + offsets, mask=mask, other=float('-inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    out_row_start = out_ptr + pid * stride
    tl.store(out_row_start + offsets, result, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](x, out, x.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out


if __name__ == "__main__":
    for shape in [(128, 128), (500, 781), (32, 4096), (2048, 1000)]:
        t = torch.randn(*shape, device="cuda")
        assert torch.allclose(softmax(t), torch.softmax(t, dim=1), atol=1e-6), shape
    print("softmax: all shapes pass")