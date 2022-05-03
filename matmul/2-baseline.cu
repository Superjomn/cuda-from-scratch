#include "common.h"

#define NO_MAIN

#include "1-baseline-host.cc"


__global__
void matmul_dev(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float x = 0.f;
        for (int k = 0; k < K; k++) {
             x += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = x;
    }
}

__global__
void matmul_dev_k_interchanged(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    int m = blockIdx.x * blockDim.x + threadIdx.x;

    if (k < K && m < M) {
        float x{};
        for (int n = 0; n < N; n++) {
            // band conflict?
            C[m * N + n] += A[m * K + k] * B[k * N + n];
        }
    }
}

int main() {
    std::vector<float> A_host, B_host, C_host;
    auto A = CreateDeviceVector(M * K, &A_host, true);
    auto B = CreateDeviceVector(K * N, &B_host, true);
    auto C = CreateDeviceVector(M * N, &C_host);
    auto C1 = CreateDeviceVector(M * N, &C_host);

    {
        const int THREADS = 32;
        const int BLOCKS = N / THREADS;

        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS, BLOCKS);

        matmul_dev<<<blocks, threads>>>(A, B, C, M, N, K);
        matmul_host(A_host.data(), B_host.data(), C_host.data(), M, N, K);

        auto res = VerifyDeviceResult(C_host.data(), C, M, N);
        std::cerr << "res: " << res << std::endl;

        for (int i = 0; i < REPEAT; i++) {
            matmul_dev<<<blocks, threads>>>(A, B, C, M, N, K);
        }
    }

    {
        const int THREADS = 32;
        const int BLOCKS = K / THREADS;
        dim3 threads(THREADS, THREADS);
        dim3 blocks(BLOCKS, BLOCKS);

        matmul_dev_k_interchanged<<<blocks, threads>>>(A, B, C, M, N, K);
        auto res = VerifyDeviceResult(C_host.data(), C1, M, N);
        std::cerr << "res: " << res << std::endl;

        for (int i = 0; i < REPEAT; i++) {
            matmul_dev_k_interchanged<<<blocks, threads>>>(A, B, C, M, N, K);
        }
    }

    DestroyDeviceVector(A);
    DestroyDeviceVector(B);
    DestroyDeviceVector(C);
    DestroyDeviceVector(C1);


    return 0;
}