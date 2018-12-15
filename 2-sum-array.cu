#include "cuda.h"
#include "stdlib.h"
#include "time.h"
#include <glog/logging.h>

const size_t kDim = 10;

__global__ void checkIdx() {
  printf("threadIdx(%d,%d,%d) blockIdx(%d,%d,%d) blockDim(%d,%d,%d)\n",
         threadIdx.x, threadIdx.y, threadIdx.z, //
         blockIdx.x, blockIdx.y, blockIdx.z,    //
         blockDim.x, blockDim.y, blockDim.z);
}

// only works on 1-D block.
__global__ void sumVec(float *a, float *b) {
  auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  b[idx] += a[idx];
}

int main() {
  // Allocate 3 arrays

  float A[kDim];
  float B[kDim];
  float C[kDim];

  // random them

  for (size_t i = 0; i < kDim; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Allocate GPU memory

  float *dA, *dB;

  CHECK_EQ(cudaMalloc(&dA, kDim * sizeof(float)), cudaSuccess);
  CHECK_EQ(cudaMalloc(&dB, kDim * sizeof(float)), cudaSuccess);

  // Copy to GPU
  CHECK_EQ(cudaMemcpy(dA, A, kDim * sizeof(float), cudaMemcpyHostToDevice),
           cudaSuccess);
  CHECK_EQ(cudaMemcpy(dB, B, kDim * sizeof(float), cudaMemcpyHostToDevice),
           cudaSuccess);

  // Launch kernal
  dim3 block(3); // 1-D block, can also set to 3-D
  dim3 grid((kDim + block.x - 1) / block.x);

  checkIdx<<<grid, block>>>();

  sumVec<<<grid, block>>>(dA, dB);

  // copy back the result.
  CHECK_EQ(cudaMemcpy(C, dA, kDim * sizeof(float), cudaMemcpyDeviceToHost),
           cudaSuccess);

  // Release all the resource.
  cudaDeviceReset();

  return 0;
}
