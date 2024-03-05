#include <gflags/gflags.h>
#include <cassert>
#include "./common.cuh"

template <typename T>
__global__ void transpose_read_coalesce(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // the contiguous tid
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) {
    output[i * m + j] = input[j * n + i];
  }
}

template <typename T>
__global__ void transpose_write_coalesce(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    int m) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // the contiguous tid
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < n && j < m) {
    output[j * n + i] = input[i * m + j];
  }
}

template <typename T, int TILE>
__global__ void transpose_tiled_coalesce0(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    int m) {
  assert(blockDim.x == blockDim.y && blockDim.x == TILE);

  // TILE + 1 to avoid
  __shared__ T tile[TILE][TILE + 1];

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    tile[threadIdx.x][threadIdx.y] = input[i * n + j];
  }
  __syncthreads();
  i = blockIdx.x * blockDim.x + threadIdx.y;
  j = blockIdx.y * blockDim.y + threadIdx.x;
  if (i < n && j < m) {
    output[i * m + j] = tile[threadIdx.y][threadIdx.x];
  }
}

template <typename T>
__global__ void transpose_tiled_coalesce1(
    const T* __restrict__ input,
    T* __restrict__ output,
    int n,
    int m) {
  const size_t TILE = blockDim.x;
  assert(blockDim.x == blockDim.y);

  extern __shared__ T tile[];

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m && j < n) {
    tile[threadIdx.x * (TILE + 1) + threadIdx.y] = input[i * n + j];
  }
  __syncthreads();
  i = blockIdx.x * blockDim.x + threadIdx.y;
  j = blockIdx.y * blockDim.y + threadIdx.x;
  if (i < n && j < m) {
    output[i * m + j] = tile[threadIdx.y * (TILE + 1) + threadIdx.x];
  }
}

DEFINE_int32(m, 1024, "The number of rows");
DEFINE_int32(n, 1024, "The number of cols");
DEFINE_int32(
    blockDim,
    32,
    "The size of the block dimension, both x and y shares the same value");
DEFINE_int32(kernel, 0, "The kernel to run (0: read, 1: write, 2: tiled)");
DEFINE_bool(profile, false, "Whether to profile the kernel");

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Matrix<uint64_t> input(FLAGS_m, FLAGS_n);
  Matrix<uint64_t> output(FLAGS_n, FLAGS_m);

  std::vector<uint64_t> input_data(FLAGS_m * FLAGS_n);
  for (int i = 0; i < FLAGS_m * FLAGS_n; ++i) {
    input_data[i] = i;
  }
  input.assign(input_data.data());

  dim3 blockSize(FLAGS_blockDim, FLAGS_blockDim);
  dim3 gridSize(
      (FLAGS_n + blockSize.x - 1) / blockSize.x,
      (FLAGS_m + blockSize.y - 1) / blockSize.y);

  auto fn = [&](cudaStream_t stream) {
    switch (FLAGS_kernel) {
      case 0:
        transpose_read_coalesce<<<gridSize, blockSize, 0, stream>>>(
            input.data, output.data, FLAGS_n, FLAGS_m);
        break;
      case 1:
        transpose_write_coalesce<<<gridSize, blockSize, 0, stream>>>(
            input.data, output.data, FLAGS_n, FLAGS_m);
        break;
      case 2: {
        assert(FLAGS_blockDim == 32 && "Tile size is fixed to 32");
        transpose_tiled_coalesce0<uint64_t, 32>
            <<<gridSize, blockSize, 0, stream>>>(
                input.data, output.data, FLAGS_n, FLAGS_m);
      } break;
      case 3: {
        size_t sharedSize = blockSize.x * blockSize.y * sizeof(uint64_t);
        transpose_tiled_coalesce1<uint64_t>
            <<<gridSize, blockSize, sharedSize, stream>>>(
                input.data, output.data, FLAGS_n, FLAGS_m);
      } break;
    }
  };

  cudaStream_t stream;
  NVCHECK(cudaStreamCreate(&stream));

  std::cerr << "Running kernel:\t" << FLAGS_kernel << "\tblock:\t"
            << FLAGS_blockDim << "x" << FLAGS_blockDim << "\tgrid:\t"
            << gridSize.x << "x" << gridSize.y << std::endl;

  if (FLAGS_profile) {
    float time = measure_performance<void>(fn, stream);
    std::cerr << "Time: " << time << "ms" << std::endl;
  } else {
    fn(stream);
  }

  auto outputHost = output.toHost();

  for (int i = 0; i < FLAGS_n; ++i) {
    for (int j = 0; j < FLAGS_m; ++j) {
      assert(outputHost[i * FLAGS_m + j] == input_data[j * FLAGS_n + i]);
    }
  }

  if (!FLAGS_profile)
    std::cerr << "PASS" << std::endl;

  return 0;
}
