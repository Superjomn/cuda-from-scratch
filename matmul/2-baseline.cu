#include "../common.h"

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
        for (int n = 0; n < N; n++) {
            // band conflict?
            C[m * N + n] += A[m * K + k] * B[k * N + n];
        }
    }
}

__global__
void matmul_dev_continuous_row0(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float x{};
        for (int k = 0; k < K; k += 4) {
            // all threads in the same wrap access the same elements each time.
            auto *aa = reinterpret_cast<const float4 *>(&A[m * K + k]);
            // unroll
            x += aa->x * B[k * N + n];
            x += aa->y * B[(k + 1) * N + n];
            x += aa->z * B[(k + 2) * N + n];
            x += aa->w * B[(k + 3) * N + n];
        }
    }
}

__global__
void matmul_dev_continuous_row1(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float x{};
        for (int k = 0; k < K; k += 4) {
            // all threads in the same wrap access the same elements each time.
            float4 aa = reinterpret_cast<const float4 *>(&A[m * K + k])[0];
            // unroll
            x += aa.x * B[k * N + n];
            x += aa.y * B[(k + 1) * N + n];
            x += aa.z * B[(k + 2) * N + n];
            x += aa.w * B[(k + 3) * N + n];
        }
    }
}


__global__
void matmul_dev_continuous_row2(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m < M && n < N) {
        float x{};
#pragma unroll
        for (int k = 0; k < K; k += 4) {
            // all threads in the same wrap access the same elements each time.
            float4 aa = reinterpret_cast<const float4 *>(&A[m * K + k])[0];
            // unroll
            x += aa.x * B[k * N + n];
            x += aa.y * B[(k + 1) * N + n];
            x += aa.z * B[(k + 2) * N + n];
            x += aa.w * B[(k + 3) * N + n];
        }
    }
}

// 16 x 16 floats
const int share_size = 16 * 16;

__global__
void matmul_dev_tile(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    __shared__ float a[share_size];
    __shared__ float b[share_size];

    // shorten for reuse
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int dim = blockDim.x;

    const int m = blockIdx.y * blockDim.y + ty; // row
    const int n = blockIdx.x * blockDim.x + tx; // column

    if (m >= M || n >= N) return;

    float tmp{};

    // i is the index of N's tile
    for (int i = 0; i < (N / dim); i++) {
        // Load a vector from A and B to a and b.
        a[ty * dim + tx] = A[m * N + (i * dim + tx)];
        b[ty * dim + tx] = B[(i*dim+ty) * N + n];
        // Ensure all the treads in the same block has finished loading data.
        __syncthreads();

        // Calculate the result of this tile.
        for (int k = 0; k < tx; k++) {
            tmp += a[ty * dim + k] * b[k * dim + tx];
        }
        __syncthreads();
    }

    // write back the result
    C[m * N + n] = tmp;
}


__global__
void matmul_dev_tile_B_collapse(const float *A, const float *B, float *C, const int M, const int N, const int K) {
    __shared__ float a[share_size];
    __shared__ float b[share_size];

    // shorten for reuse
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int dim = blockDim.x;

    const int m = blockIdx.y * blockDim.y + ty; // row
    const int n = blockIdx.x * blockDim.x + tx; // column

    if (m >= M || n >= N) return;

    float tmp{};

    // i is the index of N's tile
    for (int i = 0; i < (K / dim); i++) {
        // Load a vector from A and B to a and b.
        a[ty * dim + tx] = A[m * N + (i * dim + tx)];
        b[ty * dim + tx] = B[(i*dim+ty) * N + n];
        // Ensure all the treads in the same block has finished loading data.
        __syncthreads();

        // Calculate the result of this tile.
        for (int k = 0; k < tx; k++) {
            tmp += a[ty * dim + k] * b[k * dim + tx];
        }
        __syncthreads();
    }

    // write back the result
    C[m * N + n] = tmp;
}

// Transpose matrix from shape M x N to N x M
__global__
void TransposeMatrix(float* A, const int M, const int N) {

}

int main() {
    std::vector<float> A_host, B_host, C_host;
    auto A = CreateDeviceVector(M * K, &A_host, true);
    auto B = CreateDeviceVector(K * N, &B_host, true);
    auto C = CreateDeviceVector(M * N, &C_host);

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

        ClearDeviceVector(C, M * N);

        matmul_dev_k_interchanged<<<blocks, threads>>>(A, B, C, M, N, K);
        auto res = VerifyDeviceResult(C_host.data(), C, M, N);
        std::cerr << "res: " << res << std::endl;

        for (int i = 0; i < REPEAT; i++) {
            matmul_dev_k_interchanged<<<blocks, threads>>>(A, B, C, M, N, K);
        }
    }


#define PROFILE_KERNEL(kernel__) \
    { \
        const int THREADS = 32; \
        const int BLOCKS = K / THREADS; \
        dim3 threads(THREADS, THREADS);\
        dim3 blocks(BLOCKS, BLOCKS); \
\
        ClearDeviceVector(C, M * N);\
        matmul_dev_continuous_row1<<<blocks, threads>>>(A, B, C, M, N, K);\
        auto res = VerifyDeviceResult(C_host.data(), C, M, N);\
        std::cerr << "res: " << res << std::endl;\
\
        for (int i = 0; i < REPEAT; i++) {\
            kernel__ <<<blocks, threads>>>(A, B, C, M, N, K);\
        }\
    }

    PROFILE_KERNEL(matmul_dev_continuous_row0)
    PROFILE_KERNEL(matmul_dev_continuous_row1)
    PROFILE_KERNEL(matmul_dev_continuous_row2)


    {
        const int BLOCKS = 16;
        const int GRIDS = (N + BLOCKS - 1) / BLOCKS;
        const int GRIDS1 = (M + BLOCKS - 1) / BLOCKS;
        dim3 threads(BLOCKS, BLOCKS); // each block has 16 x 16 threads, that is same as shared_memory size
        dim3 blocks(GRIDS, GRIDS1);

        ClearDeviceVector(C, M * N);

        matmul_dev_tile<<<blocks, threads>>>(A, B, C, M, N, K);
        auto res = VerifyDeviceResult(C_host.data(), C, M, N);
        std::cerr << "res: " << res << std::endl;

        for (int i = 0; i < REPEAT; i++) {
            matmul_dev_tile<<<blocks, threads>>>(A, B, C, M, N, K);
        }
    }

    DestroyDeviceVector(A);
    DestroyDeviceVector(B);
    DestroyDeviceVector(C);


    return 0;
}