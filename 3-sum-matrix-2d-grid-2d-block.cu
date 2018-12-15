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
  int nx = 1000;
  int ny = 800;

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

  dim3 block(nx / 100, ny / 100);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  sumMatrix<<<grid, block>>>(dA, dB, nx, ny);

  NV_CHECK(cudaMemcpy(C, dA, bytes, cudaMemcpyDeviceToHost));
  NV_CHECK(cudaFree(dA));
  NV_CHECK(cudaFree(dB));

  displaySummary("A ", A);
  displaySummary("B ", B);
  displaySummary("C ", C);

  return 0;
}
