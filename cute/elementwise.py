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


@cute.kernel
def elementwise_add_tiled_copy_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    debug: cutlass.Constexpr[bool] = False,
):
    """
    Elementwise add with tiled copy.
    """
    # Both thread_idx and block_idx should be 1-D
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    if cutlass.const_expr(debug):
        if tidx == 0 and bidx == 0:
            print(f"gA: {gA.shape}, gB: {gB.shape}, gC: {gC.shape}")
            print(
                f"block size: {cute.size(thr_layout)}, val size: {cute.size(val_layout)}"
            )

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    # copy TVs
    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    if cutlass.const_expr(debug):
        if tidx == 0 and bidx == 0:
            print(f"thrA: {thrA.shape}, thrB: {thrB.shape}, thrC: {thrC.shape}")
        return

    # declare register fragments
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    # copy data to registers
    cute.copy(copy_atom_load, thrA, frgA)
    cute.copy(copy_atom_load, thrB, frgB)
    cute.copy(copy_atom_store, frgC, thrC)

    # perform elementwise add
    result = frgA.load() + frgB.load()

    # store result to registers
    frgC.store(result)

    # copy result to global memory
    cute.copy(copy_atom_store, frgC, thrC, pred=None)  # no predicate


# Reference cutlass/examples
def elementwise_add_pred_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,  # coordinate tensor
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    """
    Elementwise add with predicate.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None, None), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    copy_atom_load = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
    copy_atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gC.element_type)

    tiled_copy_A = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_B = cute.make_tiled_copy_tv(copy_atom_load, thr_layout, val_layout)
    tiled_copy_C = cute.make_tiled_copy_tv(copy_atom_store, thr_layout, val_layout)

    thr_copy_A = tiled_copy_A.get_slice(tidx)
    thr_copy_B = tiled_copy_B.get_slice(tidx)
    thr_copy_C = tiled_copy_C.get_slice(tidx)

    thrA = thr_copy_A.partition_S(blkA)
    thrB = thr_copy_B.partition_S(blkB)
    thrC = thr_copy_C.partition_S(blkC)

    # allocate registers
    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    thrCrd = thr_copy_C.partition_S(blkCrd)
    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    # Move data to registers
    cute.copy(copy_atom_load, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom_load, thrB, frgB, pred=frgPred)

    result = frgA.load() + frgB.load()

    # Save the result to registers
    frgC.store(result)

    # Copy the results back to C
    cute.copy(copy_atom_store, frgC, thrC, pred=frgPred)


### kernel launchers


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


@cute.jit
def elementwise_add_tiled_copy(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    copy_bits: cutlass.Constexpr[int] = 128,
):
    dtype = A.element_type
    vec_size = copy_bits // dtype.width

    # 128 threads per block
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vec_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f"tiler_mn = {tiler_mn} per CTA")

    gA = cute.zipped_divide(A, tiler=tiler_mn)
    gB = cute.zipped_divide(B, tiler=tiler_mn)
    gC = cute.zipped_divide(C, tiler=tiler_mn)
    grid = [cute.size(gC, mode=[1]), 1, 1]
    block = [cute.size(tv_layout, mode=[0]), 1, 1]
    print(f"grid: {grid}, block: {block}")
    elementwise_add_tiled_copy_kernel(gA, gB, gC, thr_layout, val_layout).launch(
        grid=grid,
        block=block,
    )


@cute.jit
def elementwise_add_pred(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    shape: cute.Shape,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
    copy_bits: cutlass.Constexpr[int] = 128,
):
    dtype = A.element_type
    vec_size = copy_bits // dtype.width

    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((4, vec_size), order=(1, 0))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gA = cute.zipped_divide(A, tiler=tiler_mn)  # block tile
    gB = cute.zipped_divide(B, tiler=tiler_mn)  # block tile
    gC = cute.zipped_divide(C, tiler=tiler_mn)  # block tile

    # identity tensor: value is coordinate
    idC = cute.make_identity_tensor(gC.shape)
    cC = cute.zipped_divide(idC, tiler=tiler_mn)  # coordinates in the block tile

    grid = [cute.size(gC, mode=[1]), 1, 1]
    block = [cute.size(tv_layout, mode=[0]), 1, 1]
    print(f"grid: {grid}, block: {block}")
    elementwise_add_pred_kernel(gA, gB, gC, cC, shape, thr_layout, val_layout).launch(
        grid=grid,
        block=block,
    )
