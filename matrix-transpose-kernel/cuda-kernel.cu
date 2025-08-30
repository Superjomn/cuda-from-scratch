// @org-executor :id common-header :code-block-begin
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>

#define CEIL(x, y) (((x) + (y) - 1) / (y))

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define CFLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT2(value) (reinterpret_cast<float2*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

inline void check_torch_dtype(torch::Tensor tensor,
                              torch::ScalarType expected_dtype) {
  if (tensor.dtype() != expected_dtype) {
    throw std::runtime_error("Tensor dtype mismatch");
  }
}

#define TORCH_BINDING_COMMON_EXTENSION(func) m.def(#func, &func, #func);
// @org-executor :code-block-end

// {
// @org-executor :id naive-kernel :code-block-begin
__global__ void matrix_transpose_naive_f32_kernel(const float* A, float* B,
                                                  int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    B[col * M + row] = A[row * N + col];
  }
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-kernel :code-block-begin
template <int TILE_SIZE = 32>
__global__ void matrix_transpose_shared_f32_kernel(const float* A, float* B,
                                                   int M, int N) {
  // Allocate a tile in shared memory. The +1 padding on the second dimension
  // helps avoid shared memory bank conflicts.
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  // Calculate the base indices for the input tile this block will process.
  // blockIdx.x corresponds to columns (N), blockIdx.y to rows (M).
  int bx = blockIdx.x * TILE_SIZE;
  int by = blockIdx.y * TILE_SIZE;

  // Get the thread's indices within the block.
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // --- Step 1: Load a tile from Global Memory (A) to Shared Memory (tile) ---
  // Calculate the global row and column indices for the read from A.
  int read_col = bx + tx;
  int read_row = by + ty;

  // Boundary check to handle matrices that are not perfectly divisible by
  // TILE_SIZE.
  if (read_row < M && read_col < N) {
    // A is row-major, so index is [row * num_cols + col].
    // This read is coalesced because threads in a warp (varying tx) access
    // consecutive memory locations.
    tile[ty][tx] = A[read_row * N + read_col];
  }

  // Synchronize to ensure the entire tile is loaded into shared memory before
  // proceeding.
  __syncthreads();

  // --- Step 2: Write the transposed tile from Shared Memory to Global Memory
  // (B) --- Calculate the base indices for the output tile. Note the swap: what
  // was the column block index (blockIdx.x) now relates to the row dimension of
  // B.
  int write_base_row = bx;
  int write_base_col = by;

  // Calculate the global row and column for the write to B.
  // This pattern ensures that writes are coalesced. Threads in a warp (constant
  // ty, varying tx) will write to consecutive columns in the same row of B.
  int write_row = write_base_row + ty;
  int write_col = write_base_col + tx;

  // Boundary check for the output matrix B (N rows, M columns).
  if (write_row < N && write_col < M) {
    // B is row-major (N rows, M columns), so index is [row * num_cols + col].
    // We read from tile[tx][ty] instead of tile[ty][tx] to perform the
    // transpose.
    B[write_row * M + write_col] = tile[tx][ty];
  }
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-tiled-kernel :code-block-begin
template <int TILE_SIZE = 32>
__global__ void matrix_transpose_shared_row_tiled_f32_kernel(const float* A,
                                                             float* B, int M,
                                                             int N) {
  // Use padding to prevent shared memory bank conflicts during the write phase.
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

  // Calculate the base indices for the input tile this block will process.
  int bx = blockIdx.x * TILE_SIZE;
  int by = blockIdx.y * TILE_SIZE;

  // Get the thread's indices within the block.
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  assert(TILE_SIZE == blockDim.y);
  const int elements_per_thread = TILE_SIZE / blockDim.x;

  int read_col = bx + tx * elements_per_thread;
  int read_row = by + ty;

  for (int i = 0; i < elements_per_thread; i++) {
    if (read_row < M && (read_col + i) < N) {
      tile[ty][tx * elements_per_thread + i] = A[read_row * N + read_col + i];
    }
  }

  __syncthreads();

  // Calculate base indices for the output tile.
  // Note the swap: input tile's column base (bx) is output's row base.
  int write_base_row = bx;
  int write_base_col = by;

  // The output is assumed to be row-major, with dimensions N x M.
  int write_row = write_base_row + ty;
  int write_col_base = write_base_col + tx * elements_per_thread;

  for (int i = 0; i < elements_per_thread; i++) {
    int current_write_col = write_col_base + i;
    // The write check was already correct.
    if (write_row < N && current_write_col < M) {
      // Read from the transposed position in the shared memory tile.
      B[write_row * M + current_write_col] =
          tile[tx * elements_per_thread + i][ty];
    }
  }
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-float4-kernel :code-block-begin
// This kernel uses float4 to do vectorized load and store
template <int TILE_SIZE = 32>
__global__ void matrix_transpose_shared_row_tiled_4xf32_kernel(const float* A,
                                                               float* B, int M,
                                                               int N) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
  const int block_size = blockDim.x * blockDim.y;
  const int elements_per_thread = TILE_SIZE / blockDim.x;
  assert(
      elements_per_thread % 4 == 0 &&
      "elements_per_block should be divisible by 4 to allow vectorized load");

  int bx = blockIdx.x * TILE_SIZE;
  int by = blockIdx.y * TILE_SIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by + ty;
  int col = bx + tx * elements_per_thread;

  // Load elements_per_thread elements from A into tile
  if (row >= M)
    return;
  if (col >= N)
    return;

  for (int pack_i = 0; pack_i < elements_per_thread; pack_i += 4) {
    int load_col = col + pack_i;
    if (load_col + 3 < N) { // fast path: vectorized load
      float4 pack = CFLOAT4(A[row * N + load_col]);
      tile[ty][tx * elements_per_thread + pack_i] = pack.x;
      tile[ty][tx * elements_per_thread + pack_i + 1] = pack.y;
      tile[ty][tx * elements_per_thread + pack_i + 2] = pack.z;
      tile[ty][tx * elements_per_thread + pack_i + 3] = pack.w;
    } else {
      for (int i = 0; i < 4 && load_col + i < N; i++) {
        tile[ty][tx * elements_per_thread + pack_i + i] =
            A[row * N + load_col + i];
      }
    }
  }
  __syncthreads();

  // Write transposed tile to global memory

  // Transpose the tile first
  int store_row_base = bx;
  int store_col_base = by;

  for (int pack_i = 0; pack_i < elements_per_thread; pack_i += 4) {
    int store_row = store_row_base + ty;
    int store_col = store_col_base + tx * elements_per_thread + pack_i;

    // output is N x M
    if (store_col + 3 < M && store_row < N) { // fast path: vectorized store
      float4 pack;
      pack.x = tile[tx * elements_per_thread + pack_i][ty];
      pack.y = tile[tx * elements_per_thread + pack_i + 1][ty];
      pack.z = tile[tx * elements_per_thread + pack_i + 2][ty];
      pack.w = tile[tx * elements_per_thread + pack_i + 3][ty];
      FLOAT4(B[store_row * M + store_col]) = pack; // B is column-major
    } else if (store_row < N) {
      // boundary handling for non-vectorized stores
      for (int i = 0; i < 4 && store_col + i < M; ++i) {
        B[store_row * M + store_col + i] =
            tile[tx * elements_per_thread + pack_i + i][ty];
      }
    }
  }
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-float4-strided-kernel :code-block-begin
// This kernel uses float4 with strided access pattern (Approach 2)
/*
template <int TILE_SIZE=32>
__global__ void matrix_transpose_shared_row_tiled_4xf32_strided_kernel(const
float *A, float *B, int M, int N) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
  const int block_size = blockDim.x * blockDim.y;
  const int elements_per_thread = TILE_SIZE / block_size;
  assert(elements_per_thread % 4 == 0 && "elements_per_block should be divisible
by 4 to allow vectorized load");
  int bx = blockIdx.x * TILE_SIZE;
  int by = blockIdx.y * TILE_SIZE;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int global_thread_id = ty * blockDim.x + tx;
  int row = by + ty;
  // Load elements with strided pattern
  if (row >= M) return;
  for (int iter = 0; iter < elements_per_thread / 4; iter++) {
    int float4_offset = global_thread_id * 4 + iter * block_size * 4;
    int col = bx + float4_offset;
    if (col + 3 < N) { // vectorized load
      float4 pack = CFLOAT4(A[row * N + col]);
      // Map to shared memory - need to compute the correct tile indices
      int tile_col_base = global_thread_id + iter * block_size;
      if (tile_col_base < TILE_SIZE) {
        tile[ty][tile_col_base * 4] = pack.x;
        tile[ty][tile_col_base * 4 + 1] = pack.y;
        tile[ty][tile_col_base * 4 + 2] = pack.z;
        tile[ty][tile_col_base * 4 + 3] = pack.w;
      }
    } else {
      // boundary handling for non-vectorized loads
      for (int i = 0; i < 4 && col + i < N; i++) {
        int tile_col = global_thread_id * 4 + iter * block_size * 4 + i;
        if (tile_col < TILE_SIZE) {
          tile[ty][tile_col] = A[row * N + col + i];
        }
      }
    }
  }
  __syncthreads();
  // Write transposed tile to global memory (similar pattern)
  for (int iter = 0; iter < elements_per_thread / 4; iter++) {
    int float4_offset = global_thread_id * 4 + iter * block_size * 4;
    int col = by + float4_offset; // Note: transposed coordinates
    if (row < N && col < M) {
      int tile_col_base = global_thread_id + iter * block_size;
      if (tile_col_base < TILE_SIZE) {
        B[row * M + col] = tile[tile_col_base * 4][tx];
        if (col + 1 < M) B[row * M + col + 1] = tile[tile_col_base * 4 + 1][tx];
        if (col + 2 < M) B[row * M + col + 2] = tile[tile_col_base * 4 + 2][tx];
        if (col + 3 < M) B[row * M + col + 3] = tile[tile_col_base * 4 + 3][tx];
      }
    }
  }
}
*/

// @org-executor :code-block-end
// }

// {
// @org-executor :id naive-kernel :code-block-begin
void matrix_transpose_naive_f32(torch::Tensor x, torch::Tensor out) {
  int M = x.size(0);
  int N = x.size(1);

  dim3 block(16, 16); // 16x16 = 256 threads per block
  dim3 grid(CEIL(M, block.x), CEIL(N, block.y));

  matrix_transpose_naive_f32_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(x.data_ptr()),
      reinterpret_cast<float*>(out.data_ptr()), M, N);
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-kernel :code-block-begin
void matrix_transpose_shared_f32(torch::Tensor x, torch::Tensor out,
                                 const int TILE_SIZE = 32) {
  int M = x.size(0);
  int N = x.size(1);

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));

  matrix_transpose_shared_f32_kernel<<<grid, block>>>(
      reinterpret_cast<const float*>(x.data_ptr()),
      reinterpret_cast<float*>(out.data_ptr()), M, N);
}
// @org-executor :code-block-end
// }

// {
// @org-executor :id shared-memory-tiled-kernel-macro :code-block-begin
#define DEFINE_SHARED_ROW_TILED_KERNEL(func_name, kernel_name,                 \
                                       elements_per_thread_val)                \
  void func_name(torch::Tensor x, torch::Tensor out,                           \
                 const int TILE_SIZE = 32) {                                   \
    int M = x.size(0);                                                         \
    int N = x.size(1);                                                         \
                                                                               \
    int elements_per_thread = elements_per_thread_val;                         \
                                                                               \
    dim3 block(CEIL(TILE_SIZE, elements_per_thread) /*col*/,                   \
               TILE_SIZE /*row*/);                                             \
    dim3 grid(CEIL(N, TILE_SIZE), CEIL(M, TILE_SIZE));                         \
                                                                               \
    kernel_name<<<grid, block>>>(reinterpret_cast<const float*>(x.data_ptr()), \
                                 reinterpret_cast<float*>(out.data_ptr()), M,  \
                                 N);                                           \
  }

// Define all shared row tiled kernel launchers using the macro
DEFINE_SHARED_ROW_TILED_KERNEL(matrix_transpose_shared_row_tiled_f32,
                               matrix_transpose_shared_row_tiled_f32_kernel<32>,
                               4)
DEFINE_SHARED_ROW_TILED_KERNEL(
    matrix_transpose_shared_row_tiled_4xf32,
    matrix_transpose_shared_row_tiled_4xf32_kernel<32>, 4)
// DEFINE_SHARED_ROW_TILED_KERNEL(matrix_transpose_shared_row_tiled_4xf32_strided,
// matrix_transpose_shared_row_tiled_4xf32_strided_kernel<32>, 4)
//  @org-executor :code-block-end
//  }

// {
// @org-executor :id register-kernels :code-block-begin
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matrix_transpose_naive_f32", &matrix_transpose_naive_f32,
        "Matrix Transpose Naive");
  m.def("matrix_transpose_shared_f32", &matrix_transpose_shared_f32,
        "Matrix Transpose Shared Memory");
  m.def("matrix_transpose_shared_row_tiled_f32",
        &matrix_transpose_shared_row_tiled_f32,
        "Matrix Transpose Shared Memory Row Tiled");
  m.def("matrix_transpose_shared_row_tiled_4xf32",
        &matrix_transpose_shared_row_tiled_4xf32,
        "Matrix Transpose Shared Memory Row Tiled 4xf32");
  // m.def("matrix_transpose_shared_row_tiled_4xf32_strided",
  // &matrix_transpose_shared_row_tiled_4xf32_strided, "Matrix Transpose Shared
  // Memory Row Tiled 4xf32 Strided");
}
// @org-executor :code-block-end
// }
