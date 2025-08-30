// @org-executor :id common-header :code-block-begin
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
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

// @org-executor :id gemm-naive-f32-kernel :code-block-begin
// Basic GEMM kernel for float32
__global__ void gemm_naive_f32_kernel(const float* A, const float* B,
                                      const float* C, float* D, int M, int N,
                                      int K, int lda, int ldb, int ldc) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float D_value = 0.0f;

    for (int i = 0; i < K; i++) {
      D_value += A[row * lda + i] * B[i * ldb + col];
    }

    D[row * ldc + col] = D_value + C[row * ldc + col];
  }
}
// @org-executor :code-block-end

// @org-executor :id gemm-naive-tiled-f32-kernel :code-block-begin
template <int TILE_DIM>
__global__ void gemm_naive_tiled_f32_kernel(const float* A, const float* B,
                                            const float* C, float* D, int M,
                                            int N, int K, int lda, int ldb,
                                            int ldc) {
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int row = blockIdx.y * TILE_DIM + ty;
  const int col = blockIdx.x * TILE_DIM + tx;

  float D_value = 0.0f;

  // Loop over the tiles in the K dimension
  for (int i = 0; i < CEIL(K, TILE_DIM); i++) {
    // 1. Load A and B tiles into shared memory
    int a_row = row;
    int a_col = i * TILE_DIM + tx; // K dim
    sA[ty][tx] = (a_row < M && a_col < K) ? A[a_row * lda + a_col] : 0.0f;

    int b_row = i * TILE_DIM + ty; // K dim
    int b_col = col;
    sB[ty][tx] = (b_row < K && b_col < N) ? B[b_row * ldb + b_col] : 0.0f;

    __syncthreads();

    // 2. Compute the dot product of the tiles
    for (int k = 0; k < TILE_DIM; k++) {
      D_value += sA[ty][k] * sB[k][tx];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    D[row * ldc + col] = D_value + C[row * ldc + col];
  }
}
// @org-executor :code-block-end

// @org-executor :id gemm-tiled-mma-f32-kernel :code-block-begin
using namespace nvcuda;

#include <mma.h>
#include <torch/extension.h>

template <int TILE_M, int TILE_N, int TILE_K,
int WMMA_M=16, int WMMA_N=16, int WMMA_K=16> class GemmTiledMmaF16 {
  // Define the WMMA fragments for matrices A, B, and the accumulator
  using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                                   wmma::row_major>;
  using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                                   wmma::row_major>;
  using FragmentAcc =
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

public:
  __forceinline__ __device__ void compute(const half* A, const half* B,
                                          const float* C, float* D, int M,
                                          int N, int K, int lda, int ldb,
                                          int ldc) {
    // Note the warp-uniform, which is a requirement for
    // WMMA. All threads in a warp must have the same warp_m and warp_n values.
    // This revised logic correctly maps each warp to a specific 16x16 output
    // tile.
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    // Number of 16x16 warps per tile dimension
    constexpr int warpsPerTileN = TILE_N / WMMA_N;

    // Each "MMA group" consists of multiple warps collaborating on one 16x16
    // tile. Calculate how many physical warps form one logical MMA group.
    constexpr int mmaWarpsPerTile = (TILE_M / WMMA_M) * (TILE_N / WMMA_N);
    constexpr int warpsInBlock = (TILE_M * TILE_N) / 32;
    constexpr int warpsPerMmaGroup = warpsInBlock / mmaWarpsPerTile;

    const int mmaGroupId = warpId / warpsPerMmaGroup;
    const int warp_m = mmaGroupId / warpsPerTileN;
    const int warp_n = mmaGroupId % warpsPerTileN;

    FragmentA a_frag;
    FragmentB b_frag;
    FragmentAcc acc;

    // Load initial accumulator values from matrix C
    int c_row_start = blockIdx.y * TILE_M + warp_m * WMMA_M;
    int c_col_start = blockIdx.x * TILE_N + warp_n * WMMA_N;
    if (c_row_start < M && c_col_start < N) {
      wmma::load_matrix_sync(acc, C + c_row_start * ldc + c_col_start, ldc,
                             wmma::mem_row_major);
    } else {
      wmma::fill_fragment(acc, 0.0f);
    }

    // Shared memory for tiles of A and B
    __shared__ half sA[TILE_M][TILE_K];
    __shared__ half sB[TILE_K][TILE_N];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Loop over the K dimension in tile-sized steps
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
      // 1. Load A and B tiles from global to shared memory
      int a_row = blockIdx.y * TILE_M + ty;
      int a_col = k_tile + tx;
      int b_row = k_tile + ty;
      int b_col = blockIdx.x * TILE_N + tx;

      // Load tile for A with boundary checks
      if (tx < TILE_K && ty < TILE_M) {
        sA[ty][tx] = (a_row < M && a_col < K) ? A[a_row * lda + a_col]
                                              : __float2half(0.0f);
      }

      // Load tile for B with boundary checks
      if (tx < TILE_N && ty < TILE_K) {
        sB[ty][tx] = (b_row < K && b_col < N) ? B[b_row * ldb + b_col]
                                              : __float2half(0.0f);
      }

      __syncthreads();

      // 2. Perform matrix multiplication on tiles in shared memory
      // This inner loop is generic; for 16x16x16, it runs once.
      for (int j = 0; j < TILE_K; j += WMMA_K) {
        half* a_ptr = &sA[warp_m * WMMA_M][j];
        half* b_ptr = &sB[j][warp_n * WMMA_N];

        // Load matrices into fragments
        wmma::load_matrix_sync(a_frag, a_ptr, TILE_K);
        wmma::load_matrix_sync(b_frag, b_ptr, TILE_N);

        // Perform matrix multiplication
        wmma::mma_sync(acc, a_frag, b_frag, acc);
      }
      __syncthreads();
    }

    // 3. Store the result from accumulator to D
    int d_row = blockIdx.y * TILE_M + warp_m * WMMA_M;
    int d_col = blockIdx.x * TILE_N + warp_n * WMMA_N;

    if (d_row < M && d_col < N) {
      wmma::store_matrix_sync(D + d_row * ldc + d_col, acc, ldc,
                              wmma::mem_row_major);
    }
  }
};

// Boilerplate kernel launcher
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled_mma_f16_kernel(const half* A, const half* B,
                                          const float* C, float* D, int M,
                                          int N, int K, int lda, int ldb,
                                          int ldc) {
  GemmTiledMmaF16<TILE_M, TILE_N, TILE_K> op;
  op.compute(A, B, C, D, M, N, K, lda, ldb, ldc);
}

struct MmaShape16x16x16 {
  static constexpr int M = 16;
  static constexpr int N = 16;
  static constexpr int K = 16;
};

// This simply loads the matrix A and B in 128-bit chunks, and then performs the matrix multiplication.
template <int TILE_M, int TILE_N, int TILE_K, typename MmaShape=MmaShape16x16x16> class GemmTiledMmaV2F16 {
  static constexpr int WMMA_M = MmaShape::M;
  static constexpr int WMMA_N = MmaShape::N;
  static constexpr int WMMA_K = MmaShape::K;

  // Define the WMMA fragments for matrices A, B, and the accumulator
  using FragmentA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                                   wmma::row_major>;
  using FragmentB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                                   wmma::row_major>;
  using FragmentAcc =
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

private:
  FragmentA a_frag_;
  FragmentB b_frag_;
  FragmentAcc acc_;
  int warp_m_;
  int warp_n_;

  // Load C tile
  // Since the C tile should align with the warp, so warm_m and warp_n are used
  __forceinline__ __device__ void load_C_tile(const float* C, int M, int N, int K, int ldc) {
    int c_row_start = blockIdx.y * TILE_M + warp_m_ * WMMA_M;
    int c_col_start = blockIdx.x * TILE_N + warp_n_ * WMMA_N;

    if (c_row_start < M && c_col_start < N) {
      wmma::load_matrix_sync(acc_, C + c_row_start * ldc + c_col_start, ldc,
                             wmma::mem_row_major);
    } else {
      wmma::fill_fragment(acc_, 0.0f);
    }
  }

  // Load A tile
  __forceinline__ __device__ void load_A_tile(
    half sA[TILE_M][TILE_K], const half* A, int M, int K, int lda, int k_tile) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tile_row = tid / TILE_K;
    int tile_col = tid % TILE_K;
    int a_row = blockIdx.y * TILE_M + tile_row;
    int a_col = k_tile + tile_col;

    if (tile_row < TILE_M && tile_col < TILE_K) {
      sA[tile_row][tile_col] = (a_row < M && a_col < K) ? A[a_row * lda + a_col]
                                                        : __float2half(0.0f);
    }
  }

  // Load B tile
  __forceinline__ __device__ void load_B_tile(
    half sB[TILE_K][TILE_N], const half* B, int K, int N, int ldb, int k_tile) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int tile_row = tid / TILE_N;
    int tile_col = tid % TILE_N;
    int b_row = k_tile + tile_row;
    int b_col = blockIdx.x * TILE_N + tile_col;

    if (tile_row < TILE_K && tile_col < TILE_N) {
      sB[tile_row][tile_col] = (b_row < K && b_col < N) ? B[b_row * ldb + b_col]
                                                        : __float2half(0.0f);
    }
  }

public:
  __forceinline__ __device__ void compute(const half* A, const half* B,
                                          const float* C, float* D, int M,
                                          int N, int K, int lda, int ldb,
                                          int ldc) {
    // Note the warp-uniform, which is a requirement for
    // WMMA. All threads in a warp must have the same warp_m and warp_n values.
    // This revised logic correctly maps each warp to a specific 16x16 output
    // tile.
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;

    // Number of 16x16 warps per tile dimension
    constexpr int warps_per_tile_n = TILE_N / WMMA_N;

    // Each "MMA group" consists of multiple warps collaborating on one 16x16
    // tile. Calculate how many physical warps form one logical MMA group.
    constexpr int mma_tiles_per_block = (TILE_M / WMMA_M) * (TILE_N / WMMA_N);
    constexpr int mma_warps_per_tile = (WMMA_M * WMMA_N) / 32;

    constexpr int warps_per_mma_group = mma_tiles_per_block / mma_warps_per_tile;

    const int mma_group_id = warpId / warps_per_mma_group;
    warp_m_ = mma_group_id / warps_per_tile_n;
    warp_n_ = mma_group_id % warps_per_tile_n;

    // Load initial accumulator values from matrix C
    load_C_tile(C, M, N, K, ldc);

    // Shared memory for tiles of A and B
    __shared__ half sA[TILE_M][TILE_K];
    __shared__ half sB[TILE_K][TILE_N];

    // Loop over the K dimension in tile-sized steps
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
      load_A_tile(sA, A, M, K, lda, k_tile);
      load_B_tile(sB, B, K, N, ldb, k_tile);
      __syncthreads();

      #pragma unroll
      for (int j = 0; j < TILE_K; j += WMMA_K) {
        half* a_ptr = &sA[warp_m_ * WMMA_M][j];
        half* b_ptr = &sB[j][warp_n_ * WMMA_N];

        // Load matrices into fragments
        wmma::load_matrix_sync(a_frag_, a_ptr, TILE_K);
        wmma::load_matrix_sync(b_frag_, b_ptr, TILE_N);

          // Perform matrix multiplication
        wmma::mma_sync(acc_, a_frag_, b_frag_, acc_);
      }

      __syncthreads();
    }

    // 3. Store the result from accumulator to D
    int d_row = blockIdx.y * TILE_M + warp_m_ * WMMA_M;
    int d_col = blockIdx.x * TILE_N + warp_n_ * WMMA_N;

    if (d_row < M && d_col < N) {
      wmma::store_matrix_sync(D + d_row * ldc + d_col, acc_, ldc,
                              wmma::mem_row_major);
    }
  }
};

// Boilerplate kernel launcher for V2
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void gemm_tiled_mma_v2_f16_kernel(const half* A, const half* B,
                                              const float* C, float* D, int M,
                                              int N, int K, int lda, int ldb,
                                              int ldc) {
  GemmTiledMmaV2F16<TILE_M, TILE_N, TILE_K> op;
  op.compute(A, B, C, D, M, N, K, lda, ldb, ldc);
}
// @org-executor :code-block-end