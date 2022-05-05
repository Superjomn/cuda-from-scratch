#include "../common.h"

// Referenced the blog https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

// We will launch blocks of [TILE_DIM=32, BLOCK_ROWS=8] threads.
// Each processes TILE_DIM x TILE_DIM elements each time, each thread processes (TILE_DIM / BLOCK_ROWS) elements
__global__
void CopyMatrix(float *dest, const float *src, const int M, const int N) {
    // (x,y) coordinates on TILE-wise
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    const int size = M * N;

    for (int j = 0; j < TILE_DIM && ((y + j) * width + x < size); j += BLOCK_ROWS) {
        dest[(y + j) * width + x] = src[(y + j) * width + x];
    }
}

__global__
void MatrixTranspose(float *dest, const float *src, const int M, const int N) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        dest[x * width + (y + j)] = src[(y + j) * width + x];
    }
}

__global__
void MatrixTransposeCoalesced(float *dest, const float *src, const int M, const int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Load continuous src memory to TILE
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * width + x];
    }
    __syncthreads();

    // After remapping the index, store the TILE to continuous dest memory.
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        dest[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// Void Shared Memory Bank Conflicts
__global__
void MatrixTransposeCoalescedBank(float *dest, const float *src, const int M, const int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // just padding
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    // Load continuous src memory to TILE
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        tile[threadIdx.y + j][threadIdx.x] = src[(y + j) * width + x];
    }
    __syncthreads();

    // After remapping the index, store the TILE to continuous dest memory.
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        dest[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}


__global__
void VerifyDeviceResult(const float *A, const float *B, const int N, int *diff) {
    __shared__ int mydiff;

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM && ((y + j) * width + x) < N; j += BLOCK_ROWS) {
        if (std::abs(A[(y + j) * width + x] - B[(y + j) * width + x]) >= 1e-5) {
            mydiff++;
        }
    }
    __syncthreads();
    *diff += mydiff;
}


int main() {
    {
        const int M = 1024;
        const int N = 1024;
        std::vector<float> A_host;
        auto *A = CreateDeviceVector(M * N, &A_host, true);
        auto *A_copied = CreateDeviceVector(M * N, &A_host);
        std::vector<int> diff_;
        auto *diffv = CreateDeviceVector<int>(1, &diff_);

        dim3 grid(N / TILE_DIM, M / TILE_DIM);
        dim3 thread(TILE_DIM, BLOCK_ROWS);

        {
            CopyMatrix<<<grid, thread>>>(A_copied, A, M, N);
            VerifyDeviceResult<<<grid, thread>>>(A_copied, A, M * N, diffv);
            cudaMemcpy(diff_.data(), diffv, sizeof(int), cudaMemcpyDeviceToHost);
            std::cerr << "diff: " << diff_[0] << std::endl;
        }

        for (int i = 0; i < REPEAT; i++) {
            MatrixTranspose<<<grid, thread>>>(A_copied, A, M, N);
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < REPEAT; i++) {
            MatrixTransposeCoalesced<<<grid, thread>>>(A_copied, A, M, N);
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < REPEAT; i++) {
            MatrixTransposeCoalescedBank<<<grid, thread>>>(A_copied, A, M, N);
        }
        cudaDeviceSynchronize();
    }

    return 0;
}