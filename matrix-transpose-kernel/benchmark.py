#! /usr/bin/env python3

from tabnanny import verbose
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import time
from functools import partial
from typing import Optional, Tuple, List
import numpy as np

import torch
import os
import sys

def build_torch_lib():
    source_files = ["cuda-kernel.cu"]

    setup(
        name='torch_lib',  # The name of your module
        ext_modules=[
            CppExtension(
                'torch_lib',
                source_files
            ),
        ],
        cmdclass={
            'build_ext': BuildExtension
        },
        verbose=True,
    )


def load_cuda_lib():
    """Load the compiled CUDA library"""
    sys.path.append("build/lib.linux-x86_64-cpython-312")
    import torch_lib as lib
    return lib

def matrix_transpose_naive_f32_torch(a: torch.Tensor, out: torch.Tensor, *args) -> torch.Tensor:
    """ Baseline implementation of matrix transpose in torch """
    out.copy_(a.t())
    return out

torch.set_grad_enabled(False)
print("✓ CUDA kernels loaded successfully")

def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 1000,
    show_all: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, float]:
    """Run benchmark for matrix transpose function"""
    lib = load_cuda_lib()

    if out is None:
        # Create output tensor with transposed dimensions
        out = torch.empty((a.size(1), a.size(0)), dtype=a.dtype, device=a.device)

    out.fill_(0)

    # Warmup
    for _ in range(warmup):
        perf_func(a, out, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        perf_func(a, out, **kwargs)
    end_event.record()

    torch.cuda.synchronize()

    total_time = start_event.elapsed_time(end_event)  # ms
    mean_time = total_time / iters

    # Calculate memory bandwidth
    M, N = a.size(0), a.size(1)
    total_elements = M * N
    bytes_per_element = a.element_size()
    total_bytes = total_elements * bytes_per_element * 2  # Read + Write
    bandwidth_gbps = (total_bytes / (mean_time * 1e-3)) / (1024**3)  # GB/s

    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>18}: {out_val}, time: {mean_time:.4f}ms, BW: {bandwidth_gbps:.2f} GB/s")

    if show_all:
        print(out)

    return out, mean_time


def validate_result(result: torch.Tensor, reference: torch.Tensor, tag: str, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Validate the result against reference implementation"""
    lib = load_cuda_lib()

    try:
        torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)
        print(f"✓ {tag} validation passed")
        return True
    except AssertionError as e:
        print(f"✗ {tag} validation failed: {e}")
        return False


def benchmark_matrix_transpose(matrix_sizes: List[Tuple[int, int]],
                             warmup: int = 10,
                             iters: int = 100) -> None:
    """Comprehensive benchmark for matrix transpose implementations"""
    lib = load_cuda_lib()

    print("=" * 80)
    print("Matrix Transpose Benchmark")
    print("=" * 80)

    tile_size = 32

    implementations = [
        ("torch", matrix_transpose_naive_f32_torch),
        ("cuda_naive", lib.matrix_transpose_naive_f32),
        ("cuda_shared", lambda a, out: lib.matrix_transpose_shared_f32(a, out, tile_size)),
        ("cuda_shared_row_tiled", lambda a, out: lib.matrix_transpose_shared_row_tiled_f32(a, out, tile_size)),
        ("cuda_shared_row_tiled_4xf32", lambda a, out: lib.matrix_transpose_shared_row_tiled_4xf32(a, out, tile_size)),
        ("cuda_shared_row_tiled_4xf32_strided", lambda a, out: lib.matrix_transpose_shared_row_tiled_4xf32_strided(a, out, tile_size)),
    ]

    for M, N in matrix_sizes:
        print(f"\nMatrix size: {M} x {N}")
        print("-" * 40)

        # Create input matrix
        a = torch.randn(M, N, dtype=torch.float32, device='cuda')
        reference_out = torch.empty((N, M), dtype=torch.float32, device='cuda')

        # Get reference result
        matrix_transpose_naive_f32_torch(a, reference_out)

        results = {}
        times = {}

        for name, func in implementations:
            out = torch.empty((N, M), dtype=torch.float32, device='cuda')

            try:
                out, mean_time = run_benchmark(
                    func, a, name, out=out,
                    warmup=warmup, iters=iters
                )

                # Validate result (skip for torch reference)
                if name != "torch":
                    validate_result(out, reference_out, name)

                results[name] = out
                times[name] = mean_time

            except Exception as e:
                print(f"✗ {name} failed: {e}")
                continue

        # Performance comparison
        if len(times) > 1:
            print("\nPerformance comparison:")
            torch_time = times.get('torch', float('inf'))
            for name, time in times.items():
                if name != 'torch':
                    speedup = torch_time / time if time > 0 else 0
                    print(f"  {name:>12}: {speedup:.2f}x speedup over torch")


def quick_test():
    """Quick test for development"""
    matrix_sizes = [
        (512, 512),
        (1024, 1024),
    ]

    benchmark_matrix_transpose(matrix_sizes, warmup=5, iters=50)


def comprehensive_test():
    """Comprehensive benchmark with multiple matrix sizes"""
    matrix_sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        (4096, 4096),
        (8192, 8192),  # Large square matrix
        (4096, 8192),  # Rectangular matrices
        (8192, 4096),
    ]

    benchmark_matrix_transpose(matrix_sizes, warmup=10, iters=100)


def profile_memory_access_patterns():
    """Profile different memory access patterns"""
    print("\n" + "=" * 80)
    print("Memory Access Pattern Analysis")
    print("=" * 80)

    lib = load_cuda_lib()

    # Test with different matrix shapes to analyze memory access patterns
    test_cases = [
        (1024, 1024, "Square matrix"),
        (8192, 1024, "Tall matrix (8:1 ratio)"),
        (1024, 8192, "Wide matrix (1:8 ratio)"),
        (16384, 512, "Very tall matrix (32:1 ratio)"),
        (512, 16384, "Very wide matrix (1:32 ratio)"),
    ]

    for M, N, description in test_cases:
        print(f"\n{description}: {M} x {N}")
        print("-" * 50)

        a = torch.randn(M, N, dtype=torch.float32, device='cuda')
        reference_out = torch.empty((N, M), dtype=torch.float32, device='cuda')
        matrix_transpose_naive_f32_torch(a, reference_out)

        # Test only CUDA kernels for memory pattern analysis
        cuda_implementations = [
            ("cuda_naive", lib.matrix_transpose_naive_f32),
            ("cuda_shared", lib.matrix_transpose_shared_f32),
            ("cuda_coalesced", lib.matrix_transpose_coalesced_f32),
        ]

        for name, func in cuda_implementations:
            out = torch.empty((N, M), dtype=torch.float32, device='cuda')
            try:
                _, mean_time = run_benchmark(func, a, name, out=out, warmup=5, iters=50)
                validate_result(out, reference_out, name)
            except Exception as e:
                print(f"✗ {name} failed: {e}")


import click

@click.group()
def cli():
    """Matrix Transpose Benchmark CLI"""
    pass

@cli.command()
def quick():
    """Run quick test with small matrices"""
    quick_test()

@cli.command()
def profile():
    """Run memory access pattern profiling"""
    profile_memory_access_patterns()

@cli.command()
def comprehensive():
    """Run comprehensive benchmark with various matrix sizes"""
    comprehensive_test()

@cli.command()
def build():
    """Build the CUDA library"""
    build_torch_lib()

@cli.command()
def default():
    """Default: run quick test and show help"""
    print("Running quick test. Use 'comprehensive' for full benchmark or 'profile' for memory analysis")
    print("Available commands:")
    print("  quick          : Quick test with small matrices")
    print("  comprehensive  : Full benchmark with various matrix sizes")
    print("  profile        : Memory access pattern analysis")
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
