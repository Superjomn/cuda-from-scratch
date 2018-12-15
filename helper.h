#include "cuda.h"
#include "glog/logging.h"
#include "stdio.h"
#include "sys/time.h"

#define NV_CHECK(call)                                                         \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));      \
      exit(1);                                                                 \
    }                                                                          \
  }

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void randVec(float *arr, size_t len) {
  for (size_t i = 0; i < len; i++) {
    arr[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}

void displaySummary(const char *header, float *arr, int len = 10) {
  printf(header);
  printf("data: ");
  for (int i = 0; i < len; i++) {
    printf("%f ", arr[i]);
  }
  printf("\n");
}

struct Timer {

  Timer() { start = cpuSecond(); }

  double peek() {
    cudaDeviceSynchronize();
    float time_span = cpuSecond() - start;
    start = cpuSecond();
    return time_span * 1e3;
  }

private:
  double start{0};
};


// magic number
const int kWarpSize = 32;
