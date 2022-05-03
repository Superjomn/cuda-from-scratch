#include <vector>
#include <algorithm>
#include <climits>
#include <sys/time.h>
#include <chrono>
#include <cuda_runtime.h>


std::vector<float> CreateHostVector(int n, bool rand = true) {
    std::vector<float> x(n, 0.f);

    if (rand) {
        std::generate(x.begin(), x.end(), [] { return std::rand() / INT_MAX; });
    }

    return x;
}

#ifdef __NVCC__
float *CreateDeviceVector(int n, bool rand = true) {
    float *x{};
    cudaMalloc(&x, n * sizeof(float));

    auto X = CreateHostVector(n, rand);
    cudaMemcpy(x, X.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    return x;
}

void DestroyDeviceVector(float *x) {
    cudaFree(x);
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
