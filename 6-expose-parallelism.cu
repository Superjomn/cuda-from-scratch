#include "helper.h"

// C = A + B
__global__ void sumMatrix2Dgrid(float *A, float *B, float *C, int nx, int ny) {
  unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t idy = blockDim.y * blockIdx.y + threadIdx.y;

  if (idx < nx && idy < ny) {
    int id = idy * nx + idx;
    // two memory loads, one memory store
    C[id] = A[id] + B[id];
  }
}

int main(int argc, char **argv) {
  const int nx = 4024;
  const int ny = 3024;

  float *A, *B, *C;
  A = new float[nx * ny];
  B = new float[nx * ny];
  C = new float[nx * ny];
  randVec(A, nx * ny);
  randVec(B, nx * ny);
  randVec(C, nx * ny);

  float *dA, *dB, *dC;
  int bytes = nx * ny * sizeof(float);
  NV_CHECK(cudaMalloc(&dA, bytes));
  NV_CHECK(cudaMalloc(&dB, bytes));
  NV_CHECK(cudaMalloc(&dC, bytes));

  NV_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
  NV_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

  // parse argument
  int dimx = atoi(argv[1]);
  int dimy = atoi(argv[2]);

  dim3 block(dimx, dimy);
  dim3 grid((nx + dimx - 1) / dimx, (ny + dimy - 1) / dimy);

  Timer timer;
  sumMatrix2Dgrid<<<grid, block>>>(dA, dB, dC, nx, ny);
  LOG(INFO) << "time " << timer.peek();
  NV_CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));

  cudaDeviceReset();
  return 0;
}
