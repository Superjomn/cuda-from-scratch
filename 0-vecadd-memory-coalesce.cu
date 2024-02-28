#include <gflags/gflags.h>
#include "./common.cuh"

template <typename T>
__global__ void add_coalesced0(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
__global__ void add_coalesced1(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int N) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int num_threads = blockDim.x * gridDim.x;
  while (tid < N) {
    c[tid] = a[tid] + b[tid];
    tid += num_threads;
  }
}

template <typename T>
__global__ void add_uncoalesced(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ c,
    int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int num_threads = blockDim.x * gridDim.x;
  int num_tasks = nvceil(n, num_threads);
  for (int i = 0; i < num_tasks; ++i) {
    int idx = tid * num_tasks + i;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
  }
}

DEFINE_int32(kernel, 0, "kernel");
DEFINE_int32(block_size, 256, "block size");
DEFINE_int32(num_tasks, 1, "number of tasks");
DEFINE_bool(profile, false, "profile");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const int M = 1024 * 1024;
  Vecotr<double> A(M);
  Vecotr<double> B(M);
  Vecotr<double> C(M);

  A.assign(1.f);
  B.assign(2.f);
  C.assign(0.f);

  cudaStream_t stream;
  NVCHECK(cudaStreamCreate(&stream));

  dim3 block(FLAGS_block_size);
  dim3 grid(ceil(ceil(M, block.x), FLAGS_num_tasks));

  std::cerr << "grid: " << grid << std::endl;
  std::cerr << "block: " << block << std::endl;
  std::cerr << "kernel: " << FLAGS_kernel << std::endl;

  auto add_wrapped = [&](cudaStream_t stream) -> void {
    switch (FLAGS_kernel) {
      case 0:
        add_coalesced0<<<grid, block, 0, stream>>>(A.data, B.data, C.data, M);
        break;
      case 1:
        add_coalesced1<<<grid, block, 0, stream>>>(A.data, B.data, C.data, M);
        break;
      case 2:
        add_uncoalesced<<<grid, block, 0, stream>>>(A.data, B.data, C.data, M);
        break;
      default:
        std::cerr << "ERROR: unknown kernel: " << FLAGS_kernel << std::endl;
    }
  };

  if (FLAGS_profile) {
    float time = measure_performance<void>(add_wrapped, stream);
    std::cerr << "time: " << time << std::endl;
  } else {
    add_wrapped(stream);
  }

  auto C_host = C.toHost();
  for (int i = 0; i < M; ++i) {
    if (C_host[i] != 3.f) {
      std::cerr << "ERROR: C_host[" << i << "] = " << C_host[i] << std::endl;
      return 1;
    }
  }

  return 0;
}