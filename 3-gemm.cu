#include <cublas_v2.h>
#include <gflags/gflags.h>
#include <cassert>
#include "common.cuh"

/*
M = 2048, N = 2048, K = 2048
Latency: 92.644409 ms
Compute Throughput: 185.438812 GFLOPS
Memory Throughput: 0.543278 GB/s
*/
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M,
                           int N,
                           int K,
                           float alpha,
                           float beta) {
  int m_ = blockIdx.x * blockDim.x + threadIdx.x;
  int n_ = blockIdx.y * blockDim.y + threadIdx.y;

  for (int m = m_; m < M; m += blockDim.x * gridDim.x) {    // In-case huge M
    for (int n = n_; n < N; n += blockDim.y * gridDim.y) {  // In-case huge N
      float sum = 0.;
      for (int k = 0; k < K; ++k) {
        // Issue: A is not visited in a coalesced manner, the m should be in the
        // fast-moving dimension
        // Issue: B is not visited in a coalesced manner
        sum += A[m * K + k] * B[k * N + n];  // A[m, k] * B[k, n]
      }
      // Issue: C is not visited in a coalesced manner
      C[m * N + n] = alpha * sum + C[m * N + n];
    }
  }
}

/*
M = 2048, N = 2048, K = 2048
Latency: 12.124947 ms
Compute Throughput: 1416.902668 GFLOPS
Memory Throughput: 4.151082 GB/s
*/
__global__ void gemm_coalesced(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M,
                               int N,
                               int K,
                               float alpha,
                               float beta) {
  int m_ = blockIdx.y * blockDim.y + threadIdx.y;
  int n_ = blockIdx.x * blockDim.x + threadIdx.x;

  for (int m = m_; m < M; m += blockDim.x * gridDim.x) {
    for (int n = n_; n < N; n += blockDim.y * gridDim.y) {
      float sum = 0.;
      for (int k = 0; k < K; ++k) {
        // A is visited by all the threads in broadcast manner
        // B is visited in a coalesced manner now
        sum += A[m * K + k] * B[k * N + n];
      }
      // C is visited in a coalesced manner now
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
  }
}

// This doesn't help compared to gemm_coalesced.
// TODO NCU this
template <int READ_N>
__global__ void gemm_coalesced_multiple(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M,
                                        int N,
                                        int K,
                                        float alpha,
                                        float beta) {
  int m_ = blockIdx.y * blockDim.y + threadIdx.y;
  int n_ = blockIdx.x * blockDim.x + threadIdx.x;

  for (int m = m_; m < M; m += blockDim.x * gridDim.x) {
    for (int n = n_; n < N; n += blockDim.y * gridDim.y) {
      float sum = 0.;
      for (int k = 0; k < K; k += READ_N) {
        // A is visited by all the threads in broadcast manner
        // B is visited in a coalesced manner now

        // Issue READ_N loads for both A and B each time
#pragma unroll
        for (int i = 0; i < READ_N; ++i) {
          sum += A[m * K + k + i] * B[(k + i) * N + n];
        }
      }
      // C is visited in a coalesced manner now
      C[m * N + n] = alpha * sum + beta * C[m * N + n];
    }
  }
}

/*
Basic ideas:

1. partition the A,B matrice into tiles of A_tile, B_tile, each of size TILE_SIZE x TILE_SIZE
2. load the tiles into shared memory collaboratively
3. compute the result of C_tile = A_tile * B_tile
4. store C back to global memory
*/
template <int BLOCK = 32>
__global__ void gemm_tiled_smem(float* __restrict__ A,
                                float* __restrict__ B,
                                float* __restrict__ C,
                                int M,
                                int N,
                                int K,
                                float alpha,
                                float beta) {
  // Each thread block helps to compute a C block
  // block offset: (cRow, cCol)
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // Allocate shared memory for a single A block and B block
  __shared__ float As[BLOCK][BLOCK];
  __shared__ float Bs[BLOCK][BLOCK];

  const int threadRow = threadIdx.x / BLOCK;
  const int threadCol = threadIdx.x % BLOCK;  // continuous

  // Move the pointers to the starting position within the block
  A += (cRow * BLOCK) * K;
  B += cCol * BLOCK;
  C += (cRow * BLOCK) * N + cCol * BLOCK;

  float sum = 0.;
  for (int blockIdx = 0; blockIdx < K; blockIdx += BLOCK) {
    // Load the A and B block into shared memory
    As[threadRow][threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow][threadCol] = B[threadRow * N + threadCol];

    // Move the pointers to the next block
    A += BLOCK;
    B += BLOCK * N;

    // Wait for all the threads to finish loading
    __syncthreads();

    // Compute the result of C_tile = A_tile * B_tile
    for (int k = 0; k < BLOCK; ++k) {
      sum += As[threadRow][k] * Bs[k][threadCol];
    }

    // Wait for all the threads to finish computing
    __syncthreads();
  }

  // Store the result back to the global memory
  C[threadRow * N + threadCol] = alpha * sum + beta * C[threadRow * N + threadCol];
}

void cublas_gemm(const float* A,
                 const float* B,
                 float* C,
                 int M,
                 int N,
                 int K,
                 float alpha,
                 float beta,
                 cublasHandle_t handle) {
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

DEFINE_int32(M, 4096, "M");
DEFINE_int32(N, 4096, "N");
DEFINE_int32(K, 2048, "K");
DEFINE_bool(profile, false, "profile");
DEFINE_bool(list, false, "list all kernels");
DEFINE_int32(kernel, 0, "kernel id");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  printf("M = %d, N = %d, K = %d\n", FLAGS_M, FLAGS_N, FLAGS_K);

  using kernel_t =
      std::function<void(float*, float*, float*, int, int, int, float, float, cudaStream_t)>;

  std::vector<std::tuple<std::string, kernel_t>> kernels;

  float alpha = 1.;
  float beta = 0.;

  /*
  M = 2048, N = 2048, K = 2048
  Running kernel cublas
  Latency: 9.210921 ms
  Compute Throughput: 1865.162957 GFLOPS
  Memory Throughput: 5.464345 GB/s
  */
  cublasHandle_t handle;
  cublasCreate(&handle);
  // cublasDestroy(handle);
  kernels.push_back(std::make_tuple(
      "cublas",
      [&](float* A,
          float* B,
          float* C,
          int M,
          int N,
          int K,
          float alpha,
          float beta,
          cudaStream_t stream) -> void { cublas_gemm(A, B, C, M, N, K, alpha, beta, handle); }));

  kernels.push_back(std::make_tuple("gemm_naive",
                                    [](float* A,
                                       float* B,
                                       float* C,
                                       int M,
                                       int N,
                                       int K,
                                       float alpha,
                                       float beta,
                                       cudaStream_t stream) -> void {
                                      dim3 block(32, 32);
                                      dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
                                      gemm_naive<<<grid, block, 0, stream>>>(
                                          A, B, C, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta);
                                    }));
  kernels.push_back(std::make_tuple("gemm_coalesced",
                                    [](float* A,
                                       float* B,
                                       float* C,
                                       int M,
                                       int N,
                                       int K,
                                       float alpha,
                                       float beta,
                                       cudaStream_t stream) -> void {
                                      dim3 block(32, 32);
                                      dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
                                      gemm_coalesced<<<grid, block, 0, stream>>>(
                                          A, B, C, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta);
                                    }));

  kernels.push_back(std::make_tuple("gemm_coalesced_multiple",
                                    [](float* A,
                                       float* B,
                                       float* C,
                                       int M,
                                       int N,
                                       int K,
                                       float alpha,
                                       float beta,
                                       cudaStream_t stream) -> void {
                                      dim3 block(32, 32);
                                      dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
                                      gemm_coalesced_multiple<4><<<grid, block, 0, stream>>>(
                                          A, B, C, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta);
                                    }));

  kernels.push_back(std::make_tuple("gemm_tiled_smem2x32",
                                    [](float* A,
                                       float* B,
                                       float* C,
                                       int M,
                                       int N,
                                       int K,
                                       float alpha,
                                       float beta,
                                       cudaStream_t stream) -> void {
                                      dim3 block(32, 32);
                                      dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
                                      gemm_tiled_smem<32><<<grid, block, 0, stream>>>(
                                          A, B, C, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta);
                                    }));

  // Display helper
  if (FLAGS_list) {
    int idx = 0;
    for (auto& item : kernels) {
      std::string name;
      kernel_t kernel;
      std::tie(name, kernel) = item;
      printf("%d\t%s\n", idx++, name.c_str());
    }

    return 0;
  }

  Matrix<float> A(FLAGS_M, FLAGS_K);
  Matrix<float> B(FLAGS_K, FLAGS_N);
  Matrix<float> C(FLAGS_M, FLAGS_N);

  A.randomize_float();
  B.randomize_float();
  C.zero();

  cudaStream_t stream;
  NVCHECK(cudaStreamCreate(&stream));

  std::string name;
  kernel_t kernel;
  std::tie(name, kernel) = kernels[FLAGS_kernel];

  printf("Running kernel %s\n", name.c_str());

  auto kernel_wrapper = [&](cudaStream_t stream) -> void {
    assert(FLAGS_kernel < kernels.size());
    kernel(A.data, B.data, C.data, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta, stream);
  };

  if (FLAGS_profile) {
    float time = measure_performance<void>(kernel_wrapper, stream, 100);
    float flops = (2 * FLAGS_M * FLAGS_N * FLAGS_K + FLAGS_M * FLAGS_N) / time / 1000;
    float bytes =
        (FLAGS_M * FLAGS_K + FLAGS_K * FLAGS_N + FLAGS_M * FLAGS_N) * sizeof(float) / time / 1000;

    printf("Latency: %f ms\n", time);
    printf("Compute Throughput: %f GFLOPS\n", flops);
    printf("Memory Throughput: %f GB/s\n", bytes);
  } else {
    kernel_wrapper(stream);
  }

  if (!FLAGS_profile) {
    printf("Validating\n");
    Matrix<float> C_ref(FLAGS_M, FLAGS_N);
    C_ref.zero();

    cublas_gemm(A.data, B.data, C_ref.data, FLAGS_M, FLAGS_N, FLAGS_K, alpha, beta, handle);

    auto C_data = C.toHost();
    auto C_ref_data = C_ref.toHost();

    printf("Checking the result\n");
    for (int m = 0; m < FLAGS_M; ++m) {
      for (int n = 0; n < FLAGS_N; ++n) {
        float ratio = fabs((C_data[m * FLAGS_N + n] - C_ref_data[m * FLAGS_N + n]) /
                           C_ref_data[m * FLAGS_N + n]);
        if (ratio > 5e-3) {
          printf("C[%d, %d] = %f, C_ref[%d, %d] = %f\n",
                 m,
                 n,
                 C_data[m * FLAGS_N + n],
                 m,
                 n,
                 C_ref_data[m * FLAGS_N + n]);
          assert(false);
        }
      }
    }

    std::cerr << "pass" << std::endl;
  }

  return 0;
}
