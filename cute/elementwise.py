"""
The elementwise kernels.
"""

from typing import Callable

import cutlass
import cutlass.cute as cute
import torch


@cute.jit
def add(a: cute.Tensor, b: cute.Tensor) -> cute.Tensor:
    return a + b


@cute.kernel
def naive_elementwise_1d_kernel(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, op: cutlass.Constexpr[Callable]
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bidx * bdim + tidx

    n = cute.size(A.shape)

    if tid < n:
        a_val = A[tid]
        b_val = B[tid]
        C[tid] = a_val + b_val


@cute.kernel
def elementwise_1d_tiled_kernel(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, op: cutlass.Constexpr[Callable]
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    tid = bidx * bdim + tidx
    n = cute.size(A.shape)

    if tid < n:
        # tiled to 2D: [?, 4]
        a_val = A[None, tid].load()
        b_val = B[None, tid].load()
        C[None, tid].store(a_val + b_val)


@cute.kernel
def elementwise_1d_tiled_tv_layout_kernel(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, tv_layout: cute.Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    block_coord = (None, bidx)

    # 1. get block tile first
    block_A = A[block_coord]
    block_B = B[block_coord]
    block_C = C[block_coord]

    # 2. compose for thread-index & value-index
    tid_frag_A = cute.composition(block_A, tv_layout)
    tid_frag_B = cute.composition(block_B, tv_layout)
    tid_frag_C = cute.composition(block_C, tv_layout)

    # 3. slice for thread-level view
    thrd_coord = (tidx, None)

    thr_A = tid_frag_A[thrd_coord]
    thr_B = tid_frag_B[thrd_coord]
    thr_C = tid_frag_C[thrd_coord]

    thr_C[None] = thr_A.load() + thr_B.load()  # vectorized access


# launchers ====
@cute.jit
def naive_elementwise_1d(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, op: cutlass.Constexpr[Callable]
):
    num_threads_per_block = 256

    kernel = naive_elementwise_1d_kernel(A, B, C, op)
    size = cute.size(A.shape)
    kernel.launch(
        grid=(size // num_threads_per_block, 1, 1), block=(num_threads_per_block, 1, 1)
    )


@cute.jit
def elementwise_1d_tiled(
    A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, op: cutlass.Constexpr[Callable]
):
    num_threads_per_block = 256

    size = cute.size(A.shape)

    A = cute.zipped_divide(A, tiler=(4,))
    B = cute.zipped_divide(B, tiler=(4,))
    C = cute.zipped_divide(C, tiler=(4,))

    kernel = elementwise_1d_tiled_kernel(A, B, C, op)
    num_blocks = size // (num_threads_per_block * 4)

    kernel.launch(grid=(num_blocks, 1, 1), block=(num_threads_per_block, 1, 1))


def reshape_torch_Tensor_for_tv_layout(
    tensor: torch.Tensor,
    elems_per_thread: cutlass.Constexpr[int],
    num_threads_per_block: cutlass.Constexpr[int],
) -> torch.Tensor:
    num_blocks = tensor.numel() // (elems_per_thread * num_threads_per_block)
    return tensor.reshape((-1, num_blocks))


@cute.jit
def elementwise_1d_tiled_tv_layout(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    elems_per_thread: cutlass.Constexpr[int],
    vec_width: cutlass.Constexpr[int],
    op: cutlass.Constexpr[Callable],
):
    # thread-level layout
    thr_layout = cute.make_layout((8, 32), stride=(256, 1))  # 256 threads

    # value-level layout: 8 elements per thread, read 8 rows
    num_rows = elems_per_thread // vec_width
    val_layout = cute.make_layout((num_rows, vec_width), stride=(vec_width, 1))

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    g_A = cute.zipped_divide(A, tiler=tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    g_B = cute.zipped_divide(B, tiler=tiler_mn)  # ((TileM, TileN), (RestM, RestN))
    g_C = cute.zipped_divide(C, tiler=tiler_mn)  # ((TileM, TileN), (RestM, RestN))

    size = cute.size(A.shape)
    num_blocks = size // (elems_per_thread * cute.size(thr_layout))

    kernel = elementwise_1d_tiled_tv_layout_kernel(g_A, g_B, g_C, tv_layout)
    kernel.launch(grid=(num_blocks, 1, 1), block=(cute.size(thr_layout), 1, 1))
