#include <iostream>
#include "common.h"

const int M = 256;
const int N = 128;
const int K = 64;
const int REPEAT = 100;

// A: M x K
// B: K x N
// C: M x N
void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

void matmul_k_interchange(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

int main() {
    auto A = CreateHostVector(M * K, true);
    auto B = CreateHostVector(K * N, true);
    auto C = CreateHostVector(M * N);

    {
        HostTimer timer;
        timer.Start();
        for (int i = 0; i < REPEAT; i++) {
            matmul(A.data(), B.data(), C.data(), M, N, K);
        }
        double duration = timer.Stop();
        std::cerr << "Duration: " << duration / REPEAT << " ms" << std::endl;
    }

    {
        HostTimer timer;
        timer.Start();
        for (int i = 0; i < REPEAT; i++) {
            matmul_k_interchange(A.data(), B.data(), C.data(), M, N, K);
        }
        double duration = timer.Stop();
        std::cerr << "Duration: " << duration / REPEAT << " ms" << std::endl;
    }

    return 0;
}
