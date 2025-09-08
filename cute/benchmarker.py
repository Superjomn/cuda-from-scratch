#! /usr/bin/env python3
from functools import cache, partial

import tabulate
import torch


def benchmark(callable, *, num_warmups, num_iterations, amount: int):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    for _ in range(num_warmups):
        callable()

    start_event.record(stream=torch.cuda.current_stream())
    for _ in range(num_iterations):
        callable()
    end_event.record(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    elapsed_time = start_event.elapsed_time(end_event)
    avg_time = elapsed_time / num_iterations
    throughput = amount / (avg_time / 1000) / 1e9

    return avg_time, throughput


def profile_elementwise():
    import cutlass
    import cutlass.cute as cute
    import elementwise as elt
    from cutlass.cute.runtime import from_dlpack

    cutlass.cuda.initialize_cuda_context()

    # Helper to create template tensor for compilation
    def make_template_tensor(shape: list[int], reshape_for_tv=False):
        tensor = torch.empty(shape, device="cuda", dtype=torch.float16).contiguous()
        if reshape_for_tv:
            tensor = elt.reshape_torch_Tensor_for_tv_layout(tensor, 8 * 8, 256)
        return from_dlpack(tensor)

    @cache
    def compile_naive_kernel(shape: list[int]):
        _tensor = make_template_tensor(shape)
        return cute.compile(
            elt.naive_elementwise_1d, _tensor, _tensor, _tensor, elt.add
        )

    @cache
    def compile_tiled_kernel(shape: list[int]):
        _tensor = make_template_tensor(shape)
        return cute.compile(
            elt.elementwise_1d_tiled, _tensor, _tensor, _tensor, elt.add
        )

    @cache
    def compile_tv_layout_kernel(shape: list[int]):
        _tensor = make_template_tensor(shape, reshape_for_tv=True)
        return cute.compile(
            elt.elementwise_1d_tiled_tv_layout,
            _tensor,
            _tensor,
            _tensor,
            elems_per_thread=8 * 8,
            vec_width=8,
            op=elt.add,
        )

    def perfile_shape(shape):
        A = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
        B = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
        C = torch.zeros(shape, device="cuda", dtype=torch.float16).contiguous()

        torch_task = partial(torch.add, A, B)

        shape = A.shape
        _naive_elementwise_1d = compile_naive_kernel(shape)
        _elementwise_1d_tiled = compile_tiled_kernel(shape)
        _elementwise_1d_tiled_tv_layout = compile_tv_layout_kernel(shape)

        # Helper function to convert tensors using CuTe DSL pattern
        def make_cute_tensors(*tensors):
            return tuple(from_dlpack(t) for t in tensors)

        # Simplified kernel execution functions
        def naive_task(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
            _naive_elementwise_1d(*make_cute_tensors(A, B, C))

        def tiled_task(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
            _elementwise_1d_tiled(*make_cute_tensors(A, B, C))

        def tv_layout_task(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
            # Reshape tensors for TV layout
            reshaped = [
                elt.reshape_torch_Tensor_for_tv_layout(t, 8 * 8, 256) for t in (A, B, C)
            ]
            _elementwise_1d_tiled_tv_layout(*make_cute_tensors(*reshaped))

        naive_task = partial(naive_task, A, B, C)
        tiled_task = partial(tiled_task, A, B, C)
        tv_layout_task = partial(tv_layout_task, A, B, C)

        amount = A.numel() * 2  # 2 bytes per element

        torch_time, torch_throughput = benchmark(
            torch_task, num_warmups=5, num_iterations=100, amount=amount
        )
        naive_time, naive_throughput = benchmark(
            naive_task, num_warmups=5, num_iterations=100, amount=amount
        )
        tiled_time, tiled_throughput = benchmark(
            tiled_task, num_warmups=5, num_iterations=100, amount=amount
        )
        tv_layout_time, tv_layout_throughput = benchmark(
            tv_layout_task, num_warmups=5, num_iterations=100, amount=amount
        )

        record = {
            ("torch", torch_time, torch_throughput),
            ("naive", naive_time, naive_throughput),
            ("tiled", tiled_time, tiled_throughput),
            ("tv_layout", tv_layout_time, tv_layout_throughput),
        }

        print(f"Shape: {shape}:")
        print(
            tabulate.tabulate(
                record, headers=["Method", "Time (ms)", "Throughput (GB/s)"]
            ),
        )
        print("\n\n")

    for shape in [(2048, 2048), (4096, 4096), (4096, 8192)]:
        perfile_shape(shape)


if __name__ == "__main__":
    profile_elementwise()
