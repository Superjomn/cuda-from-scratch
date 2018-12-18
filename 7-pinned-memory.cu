#include "helper.h"

int main() {
  float *m, dm;
  const int kSize = 10240;
  dm = new float[kSize];
  randVec(dm, kSize);

  {
    NV_CHECK(cudaMallocHost(&m, kSize));
    Timer timer;
    for (int i = 0; i < 100; i++) {
      NV_CHECK(cudaMemcpy(dm, m, kSize * sizeof(float)));
    }
    LOG(INFO) << "pinned copy take " << timer.pink();
    NV_CHECK(cudaFreeHost(dm));
  }

  {
    Timer timer;
    for (int i = 0; i < 100; i++) {
      NV_CHECK(cudaMemcpy(dm, m, kSize * sizeof(float)));
    }
    LOG(INFO) << "pagable copy take " << timer.pink();
    NV_CHECK(cudaFree(dm));
  }

  return 0;
}
