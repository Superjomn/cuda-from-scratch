#include "cuda-kernel.cuh"
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/types.h>

void gemm_naive_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                    torch::Tensor D, int M, int N, int K, int lda, int ldb,
                    int ldc) {
  dim3 block(16, 16);
  dim3 grid(CEIL(N, 16), CEIL(M, 16));

  gemm_naive_f32_kernel<<<grid, block>>>(
      (float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(),
      (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

template <int TILE_DIM = 16>
void gemm_naive_tiled_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                          torch::Tensor D, int M, int N, int K, int lda,
                          int ldb, int ldc) {
  dim3 block(TILE_DIM, TILE_DIM);
  dim3 grid(CEIL(N, TILE_DIM), CEIL(M, TILE_DIM));

  gemm_naive_tiled_f32_kernel<TILE_DIM><<<grid, block>>>(
      (float*)A.data_ptr(), (float*)B.data_ptr(), (float*)C.data_ptr(),
      (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

// Host function to launch the kernel
template<int TILE_M, int TILE_N, int TILE_K>
void gemm_tiled_mma_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                        torch::Tensor D, int M, int N, int K, int lda, int ldb,
                        int ldc) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));

  gemm_tiled_mma_f16_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(
      (const half*)A.data_ptr(), (const half*)B.data_ptr(),
      (const float*)C.data_ptr(), (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}

template<int TILE_M, int TILE_N, int TILE_K>
void gemm_tiled_mma_v2_f16(torch::Tensor A, torch::Tensor B, torch::Tensor C,
                        torch::Tensor D, int M, int N, int K, int lda, int ldb,
                        int ldc) {
  dim3 block(TILE_N, TILE_M);
  dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));

  gemm_tiled_mma_v2_f16_kernel<TILE_M, TILE_N, TILE_K><<<grid, block>>>(
      (const half*)A.data_ptr(), (const half*)B.data_ptr(),
      (const float*)C.data_ptr(), (float*)D.data_ptr(), M, N, K, lda, ldb, ldc);
}


// Register torch functions

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_naive_f32", &gemm_naive_f32, "GEMM Naive");
  m.def("gemm_naive_tiled_f32", &gemm_naive_tiled_f32<16>, "GEMM Naive Tiled");
  m.def("gemm_tiled_mma_16x16x16_f16", &gemm_tiled_mma_f16<16, 16, 16>, "GEMM Tiled MMA");
  m.def("gemm_tiled_mma_32x32x16_f16", &gemm_tiled_mma_f16<32, 32, 16>, "GEMM Tiled MMA");
  m.def("gemm_tiled_mma_128x128x64_f16", &gemm_tiled_mma_f16<128, 128, 64>, "GEMM Tiled MMA");

  m.def("gemm_tiled_mma_v2_16x16x16_f16", &gemm_tiled_mma_v2_f16<16, 16, 16>, "GEMM Tiled MMA V2");
  m.def("gemm_tiled_mma_v2_32x32x16_f16", &gemm_tiled_mma_v2_f16<32, 32, 16>, "GEMM Tiled MMA V2");
  m.def("gemm_tiled_mma_v2_128x128x64_f16", &gemm_tiled_mma_v2_f16<128, 128, 64>, "GEMM Tiled MMA V2");
}
