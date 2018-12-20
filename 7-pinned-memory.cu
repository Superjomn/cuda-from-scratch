#include "helper.h"
/*
  I1220 08:38:14.635007  2267 7-pinned-memory.cu:36] pinned copy take 0.554085
  I1220 08:38:14.636014  2267 7-pinned-memory.cu:41] pinned visit 0.956059
  I1220 08:38:14.639053  2267 7-pinned-memory.cu:36] pagable copy take 2.48313
  I1220 08:38:14.639241  2267 7-pinned-memory.cu:41] pagable visit 0.176907

  The visit takes more time, but still smaller compared to the memory copy time.
*/

const int kSize = 10240;

void profile(float *m, float *&dm, float *&dm1, bool pinned) {
  dim3 block(256, 1);
  dim3 grid((kSize + block.x - 1) / block.x, 1);

  if (pinned) {
    NV_CHECK(cudaMallocHost(&dm, kSize * sizeof(float)));
    NV_CHECK(cudaMallocHost(&dm1, kSize * sizeof(float)));
  } else {
    NV_CHECK(cudaMalloc(&dm, kSize * sizeof(float)));
    NV_CHECK(cudaMalloc(&dm1, kSize * sizeof(float)));
  }

  std::string header = pinned ? "pinned" : "pagable";

  {
    Timer timer;
    for (int i = 0; i < 100; i++) {
      NV_CHECK(
          cudaMemcpy(dm, m, kSize * sizeof(float), cudaMemcpyHostToDevice));
      NV_CHECK(
          cudaMemcpy(dm1, m, kSize * sizeof(float), cudaMemcpyHostToDevice));
    }
    LOG(INFO) << header << " copy take " << timer.peek();
    timer.peek();
    for (int i = 0; i < 100; i++) {
      vecAdd<<<grid, block>>>(dm, dm1, kSize);
    }
    LOG(INFO) << header << " visit " << timer.peek();
  }

  if (pinned) {
    NV_CHECK(cudaFreeHost(dm));
    NV_CHECK(cudaFreeHost(dm1));
  } else {
    NV_CHECK(cudaFree(dm));
    NV_CHECK(cudaFree(dm1));
  }
}

int main() {
  float *m, *dm, *dm1;
  m = new float[kSize];
  randVec(m, kSize);

  profile(m, dm, dm1, true);
  profile(m, dm, dm1, false);

  return 0;
}
