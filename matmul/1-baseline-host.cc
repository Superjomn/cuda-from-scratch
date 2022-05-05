#pragma once

#include <iostream>
#include "../common.h"

// A: M x K
// B: K x N
// C: M x N
void matmul_host(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float x{};
            for (int k = 0; k < K; k++) {
                x += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = x;
        }
    }
}

void matmul_host_k_interchanged(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

#ifndef NO_MAIN
#undef NO_MAIN

int main() {
    auto A = CreateHostVector(M * K, true);
    auto B = CreateHostVector(K * N, true);
    auto C = CreateHostVector(M * N);

    {
        HostTimer timer;
        timer.Start();
        for (int i = 0; i < REPEAT; i++) {
            matmul_host(A.data(), B.data(), C.data(), M, N, K);
        }
        double duration = timer.Stop();
        std::cerr << "Duration: " << duration / REPEAT << " ms" << std::endl;
    }

    {
        HostTimer timer;
        timer.Start();
        for (int i = 0; i < REPEAT; i++) {
            matmul_host_k_interchanged(A.data(), B.data(), C.data(), M, N, K);
        }
        double duration = timer.Stop();
        std::cerr << "Duration: " << duration / REPEAT << " ms" << std::endl;
    }

    return 0;
}

#endif // ifndef NO_MAIN
