"""Time CUDA dense MMQ forward/backward against cuBLAS BF16 on the same shapes.

The cuBLAS column multiplies an already-dequantized BF16 weight, i.e. it is the
speed ceiling of the "materialize the whole weight" approach without its memory
cost. Packed weights are random bytes with the right geometry; timing does not
depend on values except through denormals, which random fp16 scales can produce,
so scales are overwritten with 1.0.

    python bench/benchmark_cuda_dense_mmq.py [--rows 4096]
"""

import argparse

import torch

import torch_ggml_ops

BLOCK = {12: (256, 144, (0,)), 13: (256, 176, (0,)), 14: (256, 210, (208,)), 8: (32, 34, (0,))}
NAMES = {12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 8: "Q8_0"}


def packed_weight(qt: int, n: int, k: int) -> torch.Tensor:
    values, nbytes, scale_offsets = BLOCK[qt]
    w = torch.randint(0, 256, (n, k // values, nbytes), dtype=torch.uint8, device="cuda")
    one = torch.tensor([0x00, 0x3C], dtype=torch.uint8, device="cuda")  # fp16 1.0
    for off in scale_offsets:
        w[:, :, off : off + 2] = one
    if qt in (12, 13):
        w[:, :, 2:4] = one  # dmin
    return w.reshape(n, -1).contiguous()


def timed(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    m = args.rows
    # (quant, N, K): Qwen3.8-27B projections plus the LM head.
    shapes = [
        (12, 17408, 5120), (13, 5120, 17408), (14, 12288, 5120),
        (12, 1024, 5120), (13, 5120, 6144), (8, 1024, 5120), (14, 248320, 5120),
    ]
    print(f"M={m}  times in ms, TFLOP/s in parentheses")
    print(f"{'quant':6s} {'N':>7s} {'K':>6s} | {'mmq fwd':>16s} {'mmq bwd':>16s} | {'cublas fwd':>16s} {'cublas bwd':>16s}")
    for qt, n, k in shapes:
        rows = min(m, 1024) if n > 100000 else m
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(rows, n, device="cuda", dtype=torch.bfloat16)
        w = packed_weight(qt, n, k)
        y = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
        gx = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
        ws = torch.empty(0, device="cuda", dtype=torch.uint8)
        flop = 2.0 * rows * n * k
        t_f = timed(lambda: torch_ggml_ops.mmq_inplace(x, w, qt, n, y, ws), args.iters)
        t_b = timed(lambda: torch_ggml_ops.mmq_grad_input_inplace(g, w, qt, k, gx), args.iters)
        dense = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        t_cf = timed(lambda: torch.mm(x, dense.T, out=y), args.iters)
        t_cb = timed(lambda: torch.mm(g, dense, out=gx), args.iters)
        del dense
        fmt = lambda t: f"{t:8.2f} ({flop / t / 1e9:5.1f})"
        print(f"{NAMES[qt]:6s} {n:7d} {k:6d} | {fmt(t_f)} {fmt(t_b)} | {fmt(t_cf)} {fmt(t_cb)}  M={rows}")


if __name__ == "__main__":
    main()
