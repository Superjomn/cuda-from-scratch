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
                           int K) {
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
      C[m * N + n] += sum;
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
                               int K) {
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
      C[m * N + n] += sum;
    }
  }
}

// This doesn't help compared to gemm_coalesced.
template <int READ_N>
__global__ void gemm_coalesced_multiple(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M,
                                        int N,
                                        int K) {
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
      C[m * N + n] += sum;
    }
  }
}

void cublas_gemm(const float* A,
                 const float* B,
                 float* C,
                 int M,
                 int N,
                 int K,
                 cublasHandle_t handle) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  cublasSgemm(handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              N,
              M,
              K,
              &alpha,
              B,
              N,
              A,
              K,
              &beta,
              C,
              N);
}

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C);

DEFINE_int32(M, 4096, "M");
DEFINE_int32(N, 4096, "N");
DEFINE_int32(K, 2048, "K");
DEFINE_bool(profile, false, "profile");
DEFINE_int32(kernel, 0, "kernel id");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  printf("M = %d, N = %d, K = %d\n", FLAGS_M, FLAGS_N, FLAGS_K);

  Matrix<float> A(FLAGS_M, FLAGS_K);
  Matrix<float> B(FLAGS_K, FLAGS_N);
  Matrix<float> C(FLAGS_M, FLAGS_N);

  A.randomize_float();
  B.randomize_float();
  C.zero();

  cudaStream_t stream;
  NVCHECK(cudaStreamCreate(&stream));

  using kernel_t =
      std::function<void(float*, float*, float*, int, int, int, cudaStream_t)>;

  std::vector<std::tuple<std::string, kernel_t>> kernels;

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
  kernels.push_back(std::make_tuple("cublas",
                                    [&](float* A,
                                        float* B,
                                        float* C,
                                        int M,
                                        int N,
                                        int K,
                                        cudaStream_t stream) -> void {
                                      cublas_gemm(A, B, C, M, N, K, handle);
                                    }));

  kernels.push_back(std::make_tuple(
      "gemm_naive",
      [](float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream)
          -> void {
        dim3 block(32, 32);
        dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
        gemm_naive<<<grid, block, 0, stream>>>(
            A, B, C, FLAGS_M, FLAGS_N, FLAGS_K);
      }));
  kernels.push_back(std::make_tuple(
      "gemm_coalesced",
      [](float* A, float* B, float* C, int M, int N, int K, cudaStream_t stream)
          -> void {
        dim3 block(32, 32);
        dim3 grid(ceil(FLAGS_M, block.x), ceil(FLAGS_N, block.y));
        gemm_coalesced<<<grid, block, 0, stream>>>(
            A, B, C, FLAGS_M, FLAGS_N, FLAGS_K);
      }));

  std::string name;
  kernel_t kernel;
  std::tie(name, kernel) = kernels[FLAGS_kernel];

  printf("Running kernel %s\n", name.c_str());

  auto kernel_wrapper = [&](cudaStream_t stream) -> void {
    assert(FLAGS_kernel < kernels.size());
    kernel(A.data, B.data, C.data, FLAGS_M, FLAGS_N, FLAGS_K, stream);
  };

  if (FLAGS_profile) {
    float time = measure_performance<void>(kernel_wrapper, stream, 100);
    printf("Latency: %f ms\n", time);
    printf("Compute Throughput: %f GFLOPS\n",
           2. * FLAGS_M * FLAGS_N * FLAGS_K / time / 1e6);
    printf("Memory Throughput: %f GB/s\n",
           (FLAGS_M * FLAGS_K + FLAGS_K * FLAGS_N + FLAGS_M * FLAGS_N) *
               sizeof(float) / time / 1e6);
  } else {
    kernel_wrapper(stream);
  }

  if (!FLAGS_profile) {
    printf("Validating\n");
    auto A_data = A.toHost();
    auto B_data = B.toHost();
    auto C_data = C.toHost();

    std::vector<float> C_ref(FLAGS_M * FLAGS_N, 0.0f);

    printf("Running CPU\n");
    gemm_cpu(
        FLAGS_M, FLAGS_N, FLAGS_K, A_data.data(), B_data.data(), C_ref.data());

    printf("Checking the result\n");
    for (int m = 0; m < FLAGS_M; ++m) {
      for (int n = 0; n < FLAGS_N; ++n) {
        if (fabs(C_data[m * FLAGS_N + n] - C_ref[m * FLAGS_N + n]) > 1e-2) {
          printf("C[%d, %d] = %f, C_ref[%d, %d] = %f\n",
                 m,
                 n,
                 C_data[m * FLAGS_N + n],
                 m,
                 n,
                 C_ref[m * FLAGS_N + n]);
          assert(false);
        }
      }
    }

    /*
        for (int i = 0; i < FLAGS_M; ++i)
          for (int j = 0; j < FLAGS_N; ++j) {
            float sum = 0.;
            for (int k = 0; k < FLAGS_K; ++k) {
              sum += A_data[i * FLAGS_K + k] * B_data[k * FLAGS_N + j];
            }

            bool equal = fabs(C_data[i * FLAGS_N + j] - sum) < 1e-3;
            if (!equal) {
              printf(
                  "C[%d, %d] = %f, sum = %f\n", i, j, C_data[i * FLAGS_N + j],
      sum); assert(false);
            }
          }
      */
    std::cerr << "pass" << std::endl;
  }

  return 0;
}

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
  const int BLOCK_SIZE = 8;

  for (int i = 0; i < M; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      for (int k = 0; k < K; k += BLOCK_SIZE) {
        // Multiply the blocks
        for (int ii = i; ii < std::min(i + BLOCK_SIZE, M); ++ii) {
          for (int jj = j; jj < std::min(j + BLOCK_SIZE, N); ++jj) {
            float sum = 0.0f;
#pragma unroll
            for (int kk = k; kk < std::min(k + BLOCK_SIZE, K); ++kk) {
              sum += A[ii * K + kk] * B[kk * N + jj];
            }
            C[ii * N + jj] += sum;
          }
        }
      }
    }
  }
}
