#include "common.h"

const int M = 256;
const int N = 128;
const int K = 64;

__global__
void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        for (int k = 0; k < K; k++) {
            C[m * N + n] += A[m * K + k] * B[k * N + n];
        }
    }
}

int main() {
    float *A = CreateDeviceVector(M * K, true);
    float *B = CreateDeviceVector(K * N, true);
    float *C = CreateDeviceVector(M * N);

    const int THREADS = 32;
    const int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    matmul<<<blocks, threads>>>(A, B, C, M, N, K);

    return 0;
}