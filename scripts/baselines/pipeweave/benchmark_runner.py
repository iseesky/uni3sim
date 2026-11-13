#!/usr/bin/env python3
"""Run optional PipeWeave-specific operator benchmarks on target hardware.

Runs on the operator-selected target hardware through its configured access method.
Outputs a JSON file with measured durations per operator shape, which is then
combined with analytical features (computed offline) to produce the CSV training data.

The canonical kernel datasets are the preferred training source. This runner is
retained for reproducing PipeWeave's original framework-level measurements.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
HARDWARE_IDS = ("ascend_910b", "metax_c500", "nvidia_a800")


def _device_runtime(hardware_id):
    if hardware_id == "ascend_910b":
        try:
            import torch_npu  # noqa: F401 - registers the torch NPU backend
        except ImportError as exc:
            raise RuntimeError("ascend_910b benchmark requires torch_npu") from exc
        torch.npu.set_device(0)
        return "npu", torch.npu.synchronize
    if hardware_id in ("metax_c500", "nvidia_a800"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{hardware_id} benchmark requires a CUDA-compatible runtime")
        torch.cuda.set_device(0)
        return "cuda", torch.cuda.synchronize
    raise ValueError(f"unsupported PipeWeave benchmark hardware: {hardware_id!r}")


def _average_duration_ms(operation, synchronize, warmup, repeat):
    for _ in range(warmup):
        operation()
    synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        operation()
    synchronize()
    return (time.perf_counter() - start) / repeat * 1000.0


def benchmark_gemm(M, N, K, device, synchronize,
                   dtype=torch.bfloat16, warmup=10, repeat=50):
    """Benchmark GEMM: C = A @ B where A=[M,K], B=[K,N]."""
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    return _average_duration_ms(lambda: torch.mm(A, B), synchronize, warmup, repeat)


def benchmark_rmsnorm(seq, dim, device, synchronize,
                      dtype=torch.bfloat16, warmup=10, repeat=100):
    """Benchmark RMSNorm."""
    x = torch.randn(seq, dim, dtype=dtype, device=device)
    w = torch.ones(dim, dtype=dtype, device=device)
    eps = 1e-6

    def operation():
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
        return x / rms * w
    return _average_duration_ms(operation, synchronize, warmup, repeat)


def benchmark_siluandmul(seq, dim, device, synchronize,
                         dtype=torch.bfloat16, warmup=10, repeat=100):
    """Benchmark SiLU * Mul (gate_up activation)."""
    x = torch.randn(seq, 2 * dim, dtype=dtype, device=device)

    def operation():
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up
    return _average_duration_ms(operation, synchronize, warmup, repeat)


def benchmark_flash_attn(bs, seq_len, nh, hd, device, synchronize,
                         dtype=torch.bfloat16, warmup=5, repeat=20):
    """Benchmark scaled dot-product attention (PyTorch SDPA)."""
    Q = torch.randn(bs, nh, seq_len, hd, dtype=dtype, device=device)
    K = torch.randn(bs, nh, seq_len, hd, dtype=dtype, device=device)
    V = torch.randn(bs, nh, seq_len, hd, dtype=dtype, device=device)
    operation = lambda: F.scaled_dot_product_attention(
        Q, K, V, is_causal=True)
    return _average_duration_ms(operation, synchronize, warmup, repeat)

def generate_gemm_shapes():
    """Generate GEMM benchmark shapes covering serving workloads."""
    shapes = []
    # Typical LLM shapes: M varies (batch*seq), N/K from model dims
    Ns = [2048, 4096, 5120, 6144, 7168, 8192, 11008, 14336, 16384, 27648, 28672]
    Ks = [2048, 4096, 5120, 6144, 7168, 8192, 11008, 14336, 16384, 27648, 28672]
    Ms = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for M in Ms:
        for N in Ns:
            for K in Ks:
                if N == K or (N in [4096, 8192, 14336] and K in [4096, 8192, 14336]):
                    shapes.append((M, N, K))
    # Deduplicate
    shapes = sorted(set(shapes))
    return shapes

def generate_norm_shapes():
    """Generate RMSNorm/SiLU benchmark shapes."""
    shapes = []
    seqs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    dims = [2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 14336]
    for seq in seqs:
        for dim in dims:
            shapes.append((seq, dim))
    return shapes

def generate_attn_shapes():
    """Generate FlashAttention benchmark shapes."""
    shapes = []
    # (bs, seq_len, num_heads, head_dim)
    batch_sizes = [1, 2, 4, 8]
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    head_configs = [(32, 128), (40, 128), (64, 128), (8, 128), (16, 128)]
    for bs in batch_sizes:
        for sl in seq_lens:
            for nh, hd in head_configs:
                if bs * sl * nh * hd * 2 < 4e9:  # avoid OOM
                    shapes.append((bs, sl, nh, hd))
    return shapes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardware-id", required=True, choices=HARDWARE_IDS)
    parser.add_argument("--op", choices=["gemm", "attn", "rmsnorm", "siluandmul", "all"],
                        default="all")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-shapes", type=int, default=None,
                        help="Limit number of shapes per op (for quick testing)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        REPO_ROOT / "profiler_output/baselines/pipeweave"
        / args.hardware_id / "benchmark.json")
    device, synchronize = _device_runtime(args.hardware_id)

    results = {}

    if args.op in ("gemm", "all"):
        print("=== Benchmarking GEMM ===")
        shapes = generate_gemm_shapes()
        if args.max_shapes:
            shapes = shapes[:args.max_shapes]
        gemm_results = []
        for i, (M, N, K) in enumerate(shapes):
            dur_ms = benchmark_gemm(M, N, K, device, synchronize)
            gemm_results.append(
                {"M": M, "N": N, "K": K, "avg_duration": dur_ms})
            if (i + 1) % 50 == 0:
                print(
                    f"  GEMM {i+1}/{len(shapes)}: M={M} N={N} K={K} "
                    f"→ {dur_ms:.4f} ms")
        results["gemm"] = gemm_results
        print(f"  Total GEMM: {len(gemm_results)} shapes")

    if args.op in ("rmsnorm", "all"):
        print("=== Benchmarking RMSNorm ===")
        shapes = generate_norm_shapes()
        if args.max_shapes:
            shapes = shapes[:args.max_shapes]
        norm_results = []
        for i, (seq, dim) in enumerate(shapes):
            dur_ms = benchmark_rmsnorm(seq, dim, device, synchronize)
            norm_results.append(
                {"seq": seq, "dim": dim, "avg_duration": dur_ms})
        results["rmsnorm"] = norm_results
        print(f"  Total RMSNorm: {len(norm_results)} shapes")

    if args.op in ("siluandmul", "all"):
        print("=== Benchmarking SiLU*Mul ===")
        shapes = generate_norm_shapes()
        if args.max_shapes:
            shapes = shapes[:args.max_shapes]
        silu_results = []
        for i, (seq, dim) in enumerate(shapes):
            dur_ms = benchmark_siluandmul(seq, dim, device, synchronize)
            silu_results.append(
                {"seq": seq, "dim": dim, "avg_duration": dur_ms})
        results["siluandmul"] = silu_results
        print(f"  Total SiLU: {len(silu_results)} shapes")

    if args.op in ("attn", "all"):
        print("=== Benchmarking FlashAttention ===")
        shapes = generate_attn_shapes()
        if args.max_shapes:
            shapes = shapes[:args.max_shapes]
        attn_results = []
        for i, (bs, sl, nh, hd) in enumerate(shapes):
            dur_ms = benchmark_flash_attn(
                bs, sl, nh, hd, device, synchronize)
            attn_results.append({
                "bs": bs,
                "seq_len": sl,
                "nh": nh,
                "nkv": nh,
                "hd": hd,
                "avg_duration": dur_ms,
            })
            if (i + 1) % 20 == 0:
                print(
                    f"  ATTN {i+1}/{len(shapes)}: bs={bs} sl={sl} "
                    f"nh={nh} hd={hd} → {dur_ms:.4f} ms")
        results["attn"] = attn_results
        print(f"  Total Attn: {len(attn_results)} shapes")

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "hardware_id": args.hardware_id,
        "requested_operators": sorted(results),
        "operators": results,
    }
    with output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {output}")

if __name__ == "__main__":
    main()
