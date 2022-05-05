#pragma once

#include <vector>
#include <algorithm>
#include <climits>
#include <sys/time.h>
#include <chrono>
#include <cuda_runtime.h>
#include <tuple>
#include <iostream>

const int M = 1024;
const int N = 1024;
const int K = 1024;
const int REPEAT = 100;

template<typename T=float>
std::vector<T> CreateHostVector(int n, bool rand = true) {
    std::vector<T> x(n, 0.f);

    if (rand) {
        std::generate(x.begin(), x.end(), [] { return std::rand() / INT_MAX; });
    }

    return x;
}

#ifdef __NVCC__

template<typename T=float>
T *CreateDeviceVector(int n, std::vector<T> *host, bool rand = true) {
    T *x{};
    cudaMalloc(&x, n * sizeof(T));

    *host = CreateHostVector<T>(n, rand);
    cudaMemcpy(x, host->data(), n * sizeof(T), cudaMemcpyHostToDevice);

    return x;
}

template<typename T=float>
void DestroyDeviceVector(T *x) {
    cudaFree(x);
}

__global__
void DeviceClearVector(float *x, const size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = 0.f;
    }
}

void ClearDeviceVector(float *x, const size_t size) {
    dim3 grid(size / 32);
    dim3 block(32);
    DeviceClearVector<<<grid, block>>>(x, size);
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
