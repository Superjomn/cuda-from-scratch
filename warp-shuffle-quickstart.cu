#include "common.cuh"

__global__ void test_shlf() {
  int lane = threadIdx.x % 32;
  int value = lane;
  // The mask indicates which lanes are participating in the shuffle operation.
  value = __shfl_sync(0xffffffff, lane, 3);
  printf("shlf: lane %d, data %d\n", lane, value);
}

__global__ void test_shlf_up() {
  int lane = threadIdx.x % 32;
  int value = lane;
  // The mask indicates which lanes are participating in the shuffle operation.
  value = __shfl_up_sync(0xffffffff, lane, 3);
  printf("shlf_up: lane %d, data %d\n", lane, value);
}

__global__ void test_shlf_down() {
  int lane = threadIdx.x % 32;
  int value = lane;
  // The mask indicates which lanes are participating in the shuffle operation.
  value = __shfl_down_sync(0xffffffff, lane, 3);
  printf("shlf_down: lane %d, data %d\n", lane, value);
}

__global__ void test_shlf_xor() {
  int lane = threadIdx.x % 32;
  int value = lane;
  // The mask indicates which lanes are participating in the shuffle operation.
  value = __shfl_xor_sync(0xffffffff, lane, 1);
  printf("shlf_xor: lane %d, data %d\n", lane, value);
}

int main() {
  test_shlf<<<1, 32>>>();
  test_shlf_up<<<1, 32>>>();
  test_shlf_down<<<1, 32>>>();
  test_shlf_xor<<<1, 32>>>();

  cudaDeviceSynchronize();
  return 0;
}