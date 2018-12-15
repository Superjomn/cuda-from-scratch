#include "helper.h"

__global__ void sumMatrix(float *a, float *b, int nx, int ny) {
  unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = iy * nx + ix;
  if (ix < nx && iy < ny) {
    a[idx] += b[idx];
  }
}

int main() {
  int nx = 10000;
  int ny = 8000;

  const int bytes = nx * ny * sizeof(float);
  float *A = static_cast<float *>(malloc(bytes));
  float *B = static_cast<float *>(malloc(bytes));
  float *C = static_cast<float *>(malloc(bytes));
  memset(C, 0, bytes);

  randVec(A, nx * ny);
  randVec(B, nx * ny);

  // Create CUDA resources
  float *dA, *dB;
  NV_CHECK(cudaMalloc(&dA, bytes));
  NV_CHECK(cudaMalloc(&dB, bytes));
  NV_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
  NV_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

  // block(100, 80) takes 0.02ms
  // block(1000, 800) takes 0.025ms
  uint32_t blockX = 32*10;
  uint32_t blockY = 32*4;
  LOG(INFO) << "blockDim: " << blockX << "," << blockY;
  dim3 block(blockX, blockY);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  Timer timer;
  sumMatrix<<<grid, block>>>(dA, dB, nx, ny);
  LOG(INFO) << "takes " << timer.peek() * 1000;

  NV_CHECK(cudaMemcpy(C, dA, bytes, cudaMemcpyDeviceToHost));
  NV_CHECK(cudaFree(dA));
  NV_CHECK(cudaFree(dB));

  displaySummary("A ", A);
  displaySummary("B ", B);
  displaySummary("C ", C);

  return 0;
}
