#pragma once
#include <cuda_runtime.h>
#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#define NVCHECK(ret) \
  { gpuAssert((ret), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code,
                      const char* file,
                      int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
        stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define NVCHECK_LAST() check_last(__FILE__, __LINE__)

void check_last(const char* const file, const int line) {
  cudaError_t const err{cudaGetLastError()};
  if (err != cudaSuccess) {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// Borrowed from https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/
template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream,
                          int num_repeats = 100,
                          int num_warmups = 100) {
  cudaEvent_t start, stop;
  float time;

  NVCHECK(cudaEventCreate(&start));
  NVCHECK(cudaEventCreate(&stop));

  for (int i = 0; i < num_warmups; ++i) {
    bound_function(stream);
  }

  NVCHECK(cudaStreamSynchronize(stream));

  NVCHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < num_repeats; ++i) {
    bound_function(stream);
  }
  NVCHECK(cudaEventRecord(stop, stream));
  NVCHECK(cudaEventSynchronize(stop));
  NVCHECK_LAST();
  NVCHECK(cudaEventElapsedTime(&time, start, stop));
  NVCHECK(cudaEventDestroy(start));
  NVCHECK(cudaEventDestroy(stop));

  float const latency{time / num_repeats};

  return latency;
}

template <typename T>
class Matrix {
 public:
  using value_type = T;

  int rows{};
  int cols{};

  value_type* data{};

  Matrix(int rows, int cols) : rows(rows), cols(cols) {
    NVCHECK(cudaMalloc(&data, rows * cols * sizeof(value_type)));
  }

  void assign(const value_type* host_data) {
    NVCHECK(cudaMemcpy(data,
                       host_data,
                       rows * cols * sizeof(value_type),
                       cudaMemcpyHostToDevice));
  }

  void assign(const value_type& value) {
    std::vector<value_type> host_data(rows * cols, value);
    assign(host_data.data());
  }

  void randomize_int() {
    std::vector<int> host_data(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 100);
    for (int i = 0; i < rows * cols; ++i) {
      host_data[i] = dis(gen);
    }
    assign(reinterpret_cast<value_type*>(host_data.data()));
  }

  void randomize_float() {
    std::vector<float> host_data(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1);
    for (int i = 0; i < rows * cols; ++i) {
      host_data[i] = dis(gen);
    }
    assign(reinterpret_cast<value_type*>(host_data.data()));
  }

  void zero() {
    NVCHECK(cudaMemset(data, 0, rows * cols * sizeof(value_type)));
  }

  void ordered() {
    std::vector<value_type> host_data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
      host_data[i] = i;
    }
    assign(host_data.data());
  }

  void blocked(int max) {
    std::vector<value_type> host_data(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
      host_data[i] = i % max;
    }
    assign(host_data.data());
  }

  std::vector<value_type> toHost() const {
    std::vector<value_type> host_data(rows * cols);
    NVCHECK(cudaMemcpy(host_data.data(),
                       data,
                       rows * cols * sizeof(value_type),
                       cudaMemcpyDeviceToHost));
    return host_data;
  }

  size_t size() const { return rows * cols; }
};

template <typename T>
class Vector : public Matrix<T> {
 public:
  Vector(int size) : Matrix<T>(size, 1) {}
};

int ceil(int a, int b) { return (a + b - 1) / b; }

__device__ int nvceil(int a, int b) { return (a + b - 1) / b; }

std::ostream& operator<<(std::ostream& os, const dim3& d) {
  os << d.x << "x" << d.y << "x" << d.z;
  return os;
}
