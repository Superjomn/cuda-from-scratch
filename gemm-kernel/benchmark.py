#! /usr/bin/env python3
"""
GEMM Benchmark Suite

This script benchmarks different GEMM (General Matrix Multiply) implementations:
- PyTorch reference implementation (torch.addmm)
- CUDA naive implementation
- CUDA tiled implementation
- CUDA tensor core MMA implementation

Operation: D = A @ B + C
Where A is MxK, B is KxN, C is MxN, D is MxN

Usage:
    python benchmark.py              # Quick test
    python benchmark.py quick        # Quick test with small matrices
    python benchmark.py comprehensive # Full benchmark with various sizes
    python benchmark.py profile      # Memory access pattern analysis
    python benchmark.py build        # Build CUDA extensions
"""

import sys
from typing import List, Optional, Tuple

import click
import numpy as np
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def build_torch_lib():
    source_files = ["launcher.cu"]

    setup(
        name="torch_lib",  # The name of your module
        ext_modules=[
            CppExtension("torch_lib", source_files),
        ],
        cmdclass={"build_ext": BuildExtension},
    )


def load_cuda_lib():
    """Load the compiled CUDA library"""
    sys.path.append("build/lib.linux-x86_64-cpython-312")
    try:
        import torch_lib as lib

        return lib
    except ImportError as e:
        print("✗ Failed to import torch_lib. Did you run the build script?")
        print("  Run: python build.py")
        print(f"  Error: {e}")
        sys.exit(1)


def gemm_naive_f32_torch(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, *args
) -> torch.Tensor:
    """Baseline implementation of GEMM in torch: D = A @ B + C"""
    D.copy_(torch.addmm(C, A, B))
    return D


torch.set_grad_enabled(False)


def run_benchmark(
    perf_func: callable,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    tag: str,
    D: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """Run benchmark for GEMM function: D = A @ B + C"""
    load_cuda_lib()

    M, K = A.size(0), A.size(1)
    K_B, N = B.size(0), B.size(1)

    if K != K_B:
        raise ValueError(
            f"Matrix dimensions don't match for multiplication: A is {M}x{K}, B is {K_B}x{N}"
        )

    if D is None:
        # Create output tensor D with dimensions M x N
        D = torch.empty((M, N), dtype=torch.float32, device=A.device)

    # Determine if this is a torch function or CUDA function
    is_torch_func = tag.startswith("torch")

    # Warmup
    for _ in range(warmup):
        if is_torch_func:
            perf_func(A, B, C, D, **kwargs)
        else:
            perf_func(A, B, C, D, M, N, K, K, N, N)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        if is_torch_func:
            perf_func(A, B, C, D, **kwargs)
        else:
            perf_func(A, B, C, D, M, N, K, K, N, N)
    end_event.record()

    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event)  # ms
    mean_time = total_time / iters

    # Calculate FLOPS and memory bandwidth for GEMM
    # GEMM operations: M * N * K multiply-add operations = 2 * M * N * K FLOPs
    total_flops = 2 * M * N * K
    gflops = (total_flops / (mean_time * 1e-3)) / 1e9  # GFLOPS

    # Memory bandwidth: Read A (M*K) + B (K*N) + C (M*N), Write D (M*N)
    bytes_per_element = A.element_size()
    total_bytes = (M * K + K * N + M * N + M * N) * bytes_per_element
    bandwidth_gbps = (total_bytes / (mean_time * 1e-3)) / (1024**3)  # GB/s

    D_info = f"D_{tag}"
    D_val = D.flatten().detach().cpu().numpy().tolist()[:12]
    D_val = [round(v, 8) for v in D_val]
    print(
        f"{D_info:>18}: {D_val}, time: {mean_time:.4f}ms, {gflops:.2f} GFLOPS, BW: {bandwidth_gbps:.2f} GB/s"
    )

    if show_all:
        print(D)

    return D, mean_time


def validate_result(
    result: torch.Tensor,
    reference: torch.Tensor,
    tag: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Validate the result against reference implementation"""
    load_cuda_lib()

    assert result.dtype == torch.float32
    assert reference.dtype == torch.float32

    try:
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        print(f"✓ {tag} validation passed")
        return True
    except AssertionError as e:
        print(f"✗ {tag} validation failed: {e}")
        # Add detailed error analysis
        diff = torch.abs(result.to(torch.float32) - reference.to(torch.float32))
        max_abs_diff, max_abs_diff_idx = torch.max(diff, dim=None)

        # To avoid division by zero or near-zero, add a small epsilon
        reference_abs = torch.abs(reference)
        relative_diff = diff / (reference_abs + 1e-12)

        max_rel_diff, max_rel_diff_idx = torch.max(relative_diff, dim=None)

        max_abs_diff_idx_cpu = max_abs_diff_idx.cpu()
        max_rel_diff_idx_cpu = max_rel_diff_idx.cpu()

        if result.dim() > 1:
            max_abs_diff_idx_unraveled = np.unravel_index(
                max_abs_diff_idx_cpu, result.shape
            )
            max_rel_diff_idx_unraveled = np.unravel_index(
                max_rel_diff_idx_cpu, result.shape
            )
        else:
            max_abs_diff_idx_unraveled = max_abs_diff_idx_cpu
            max_rel_diff_idx_unraveled = max_rel_diff_idx_cpu

        print(
            f"  Max absolute difference: {max_abs_diff.item()} at index {max_abs_diff_idx_unraveled}"
        )
        print(f"    - Result value: {result[max_abs_diff_idx_unraveled]}")
        print(f"    - Reference value: {reference[max_abs_diff_idx_unraveled]}")

        print(
            f"  Max relative difference: {max_rel_diff.item()} at index {max_rel_diff_idx_unraveled}"
        )
        print(f"    - Result value: {result[max_rel_diff_idx_unraveled]}")
        print(f"    - Reference value: {reference[max_rel_diff_idx_unraveled]}")

        return False


def benchmark_gemm(
    gemm_sizes: List[Tuple[int, int, int]], warmup: int = 10, iters: int = 100
) -> None:
    """Comprehensive benchmark for GEMM implementations"""
    lib = load_cuda_lib()

    print("=" * 80)
    print("GEMM Benchmark: D = A @ B + C")
    print("=" * 80)

    for M, N, K in gemm_sizes:
        print(f"\nGEMM size: M={M}, N={N}, K={K} (A: {M}x{K}, B: {K}x{N}, C: {M}x{N})")
        print("-" * 60)

        # Create input matrices
        A = torch.randn(M, K, dtype=torch.float32, device="cuda").contiguous()
        B = torch.randn(K, N, dtype=torch.float32, device="cuda").contiguous()
        C = torch.randn(M, N, dtype=torch.float32, device="cuda").contiguous()
        reference_D = torch.empty(
            (M, N), dtype=torch.float32, device="cuda"
        ).contiguous()

        A_half = A.to(torch.float16).contiguous()
        B_half = B.to(torch.float16).contiguous()
        C.to(torch.float16).contiguous()

        implementations = [
            ("torch_f32", gemm_naive_f32_torch),
            ("cuda_naive_f32", lib.gemm_naive_f32),
            ("cuda_tiled_f32", lib.gemm_naive_tiled_f32),
            ("cuda_mma_f16_16x16x16", lib.gemm_tiled_mma_16x16x16_f16),
            ("cuda_mma_f16_32x32x16", lib.gemm_tiled_mma_32x32x16_f16),
            # ("cuda_mma_f16_64x64x16", lib.gemm_tiled_mma_64x64x16_f16),
            # ("cuda_mma_f16_128x128x64", lib.gemm_tiled_mma_128x128x64_f16),
            # ("cuda_mma_f16_256x256x16", lib.gemm_tiled_mma_256x256x16_f16),
            ("cuda_mma_f16_v2_16x16x16", lib.gemm_tiled_mma_v2_16x16x16_f16),
            ("cuda_mma_f16_v2_32x32x16", lib.gemm_tiled_mma_v2_32x32x16_f16),
            ("cuda_mma_f16_v2_128x128x64", lib.gemm_tiled_mma_v2_128x128x64_f16),
            # ("cuda_mma_f16_v2_64x64x16", lib.gemm_tiled_mma_v2_64x64x16_f16),
            # ("cuda_mma_f16_v2_128x128x64", lib.gemm_tiled_mma_v2_128x128x64_f16),
            # ("cuda_mma_f16_v2_256x256x16", lib.gemm_tiled_mma_v2_256x256x16_f16),
        ]

        # Get reference result
        gemm_naive_f32_torch(A, B, C, reference_D)

        results = {}
        times = {}

        for name, func in implementations:
            D = torch.empty((M, N), dtype=torch.float32, device="cuda")

            # For CUDA functions, we need to pass matrix dimensions and leading dimensions
            if name.startswith("cuda"):
                if "f16" in name:
                    A_input, B_input, C_input = A_half, B_half, C
                    D = torch.empty((M, N), dtype=torch.float32, device="cuda")
                else:
                    A_input, B_input, C_input = A, B, C
                    D = torch.empty((M, N), dtype=torch.float32, device="cuda")

                D, mean_time = run_benchmark(
                    func,
                    A_input,
                    B_input,
                    C_input,
                    name,
                    D=D,
                    warmup=warmup,
                    iters=iters,
                    M=M,
                    N=N,
                    K=K,
                    lda=K,
                    ldb=N,
                    ldc=N,
                )
            else:
                D, mean_time = run_benchmark(
                    func, A, B, C, name, D=D, warmup=warmup, iters=iters
                )

            # Validate result (skip for torch reference)
            if not name.startswith("torch"):
                validate_result(D, reference_D, name, rtol=30, atol=5e-1)

            results[name] = D
            times[name] = mean_time

        # Performance comparison
        if len(times) > 1:
            print("\nPerformance comparison:")
            torch_time = times.get("torch_f32", float("inf"))
            for name, time in times.items():
                if name != "torch_f32":
                    speedup = torch_time / time if time > 0 else 0
                    print(f"  {name:>12}: {speedup:.2f}x speedup over torch")


def quick_test():
    """Quick test for development"""
    gemm_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    benchmark_gemm(gemm_sizes, warmup=5, iters=50)


def comprehensive_test():
    """Comprehensive benchmark with multiple GEMM sizes"""
    gemm_sizes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (1024, 2048, 512),  # Rectangular cases
        (2048, 1024, 512),
        (4096, 4096, 4096),
        (8192, 8192, 1024),  # Large matrices with smaller K
        (4096, 8192, 2048),  # Various aspect ratios
        (8192, 4096, 2048),
    ]

    benchmark_gemm(gemm_sizes, warmup=10, iters=100)


def profile_memory_access_patterns():
    """Profile different memory access patterns for GEMM"""
    print("\n" + "=" * 80)
    print("GEMM Memory Access Pattern Analysis")
    print("=" * 80)

    lib = load_cuda_lib()

    # Test with different GEMM shapes to analyze memory access patterns
    test_cases = [
        (1024, 1024, 1024, "Square matrices"),
        (8192, 1024, 512, "Tall A matrix (8:1 M:N ratio)"),
        (1024, 8192, 512, "Wide B matrix (1:8 M:N ratio)"),
        (2048, 2048, 128, "Small K dimension"),
        (512, 512, 4096, "Large K dimension"),
        (4096, 1024, 1024, "Very tall A matrix"),
        (1024, 4096, 1024, "Very wide B matrix"),
    ]

    for M, N, K, description in test_cases:
        print(f"\n{description}: M={M}, N={N}, K={K}")
        print("-" * 60)

        A = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B = torch.randn(K, N, dtype=torch.float32, device="cuda")
        C = torch.randn(M, N, dtype=torch.float32, device="cuda")
        reference_D = torch.empty((M, N), dtype=torch.float32, device="cuda")
        gemm_naive_f32_torch(A, B, C, reference_D)

        # Test only CUDA kernels for memory pattern analysis
        cuda_implementations = [
            ("cuda_naive", lib.gemm_naive_f32),
            ("cuda_tiled", lib.gemm_naive_tiled_f32),
            ("cuda_mma", lib.gemm_tiled_mma_f32),
        ]

        for name, func in cuda_implementations:
            D = torch.empty((M, N), dtype=torch.float32, device="cuda")
            _, mean_time = run_benchmark(
                func,
                A,
                B,
                C,
                name,
                D=D,
                warmup=5,
                iters=50,
                M=M,
                N=N,
                K=K,
                lda=K,
                ldb=N,
                ldc=N,
            )
            if "f16" in name or "mma" in name:
                validate_result(D, reference_D, name, rtol=30, atol=5e-1)
            else:
                validate_result(D, reference_D, name, rtol=30, atol=5e-1)


import click


@click.group()
def cli():
    """GEMM Benchmark CLI"""


@cli.command()
def quick():
    """Run quick test with small GEMM operations"""
    quick_test()


@cli.command()
def profile():
    """Run memory access pattern profiling for GEMM"""
    profile_memory_access_patterns()


@cli.command()
def comprehensive():
    """Run comprehensive benchmark with various GEMM sizes"""
    comprehensive_test()


@cli.command()
def build():
    """Build the CUDA library"""
    build_torch_lib()


@cli.command()
def default():
    """Default: run quick test and show help"""
    print(
        "Running quick GEMM test. Use 'comprehensive' for full benchmark or 'profile' for memory analysis"
    )
    print("Available commands:")
    print("  quick          : Quick test with small GEMM operations")
    print("  comprehensive  : Full benchmark with various GEMM sizes")
    print("  profile        : Memory access pattern analysis for GEMM")
    print()
    quick_test()


def main():
    import sys

    if len(sys.argv) == 1:
        # No arguments: run default
        default()
    else:
        cli()


if __name__ == "__main__":
    main()
