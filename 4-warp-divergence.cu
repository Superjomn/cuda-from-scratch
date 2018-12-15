#include "helper.h"
/*
  With very large number of blocks, the magic 32 works.
  Make the threads in a warp run the same instructions(if-else branch) will
  accelerate the performace.
  e.g. A simple assignment operation, the warp divergent version takes 0.12ms,
  while the optimized one takes only 0.05ms, one-time acceleration.

  But with small number of blocks, the optimized is slower.
 */

__global__ void kernel0(float *c) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1-d block
  float a = 0., b = 0.;
  if (idx % 2 == 0) {
    a = 100.;
  } else {
    b = 100;
  }

  c[idx] = a + b;
}

__global__ void kernel1(float *c) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1-d block
  float a = 0., b = 0.;
  if (idx / kWarpSize % 2 == 0) {
    a = 100.;
  } else {
    b = 100;
  }

  c[idx] = a + b;
}

int main() {
  int dim = 12800;
  float *dC;
  NV_CHECK(cudaMalloc(&dC, dim * sizeof(float)));
  NV_CHECK(cudaMemset(dC, 0, dim * sizeof(float)));

  // warm up
  kernel0<<<dim, 1>>>(dC);
  kernel1<<<dim, 1>>>(dC);

  Timer timer;
  kernel0<<<dim, 1>>>(dC);
  LOG(INFO) << "kernel1: " << timer.peek();
  timer.peek();

  kernel1<<<dim, 1>>>(dC);
  LOG(INFO) << "kernel2: " << timer.peek();

  return 0;
}
