#include "cuda.h"
#include "stdio.h"

__global__ void hello() { printf("Hello world from GPU\n"); }

int main() {
  // will print 5 hello world.
  hello<<<1, 5>>>();

  // reset all the resources in GPU for this process.
  // If no cudaDeviceReset(), no output will print, the program in CPU will just
  // quit without waiting for GPU response.
  cudaDeviceReset();
  return 0;
}
