#pragma once

#include <vector>
#include <algorithm>
#include <climits>
#include <sys/time.h>
#include <chrono>
#include <cuda_runtime.h>
#include <tuple>

const int M = 256;
const int N = 256;
const int K = 256;
const int REPEAT = 100;

std::vector<float> CreateHostVector(int n, bool rand = true) {
    std::vector<float> x(n, 0.f);

    if (rand) {
        std::generate(x.begin(), x.end(), [] { return std::rand() / INT_MAX; });
    }

    return x;
}

#ifdef __NVCC__

float *CreateDeviceVector(int n, std::vector<float> *host, bool rand = true) {
    float *x{};
    cudaMalloc(&x, n * sizeof(float));

    *host = CreateHostVector(n, rand);
    cudaMemcpy(x, host->data(), n * sizeof(float), cudaMemcpyHostToDevice);

    return x;
}

void DestroyDeviceVector(float *x) {
    cudaFree(x);
}

bool VerifyDeviceResult(const float *host_C, const float *dev_C, const int M, const int N) {
    auto C_ = CreateHostVector(M * N);
    cudaMemcpy(C_.data(), dev_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * N; i++) {
        if (std::abs(host_C[i] - C_[i]) > 1e-5) return false;
    }
    return true;
}

#endif


/**
 * Host-side Timer in ms.
 *
 * Usage:
 *   HostTimer timer;
 *   timer.Start();
 *   // do something
 *   double duration = timer.Stop();
 */
struct HostTimer {
    void Start() {
        start_ = clock();
    }

    double Stop() {
        return (clock() - start_) / CLOCKS_PER_SEC * 1e3;
    }


private:
    double start_{};
};
