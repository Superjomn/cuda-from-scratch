/*
  I1223 11:07:55.462594  3677 8-cached-visit.cu:105] test offset 0 takes 82.8531
  I1223 11:07:55.546900  3677 8-cached-visit.cu:105] test offset 0 takes 84.235
  I1223 11:07:55.631135  3677 8-cached-visit.cu:105] test offset 0 takes 84.2221
  I1223 11:07:55.716351  3677 8-cached-visit.cu:105] test offset 11 takes
  85.2001
  I1223 11:07:55.802806  3677 8-cached-visit.cu:105] test offset 16 takes
  86.4291
  I1223 11:07:55.889264  3677 8-cached-visit.cu:105] test offset 128 takes
  86.4351
  I1223 11:07:55.891125  3677 8-cached-visit.cu:61] visit AofS
  I1223 11:07:56.008760  3677 8-cached-visit.cu:65] takes 117.631ms
  I1223 11:07:56.040253  3677 8-cached-visit.cu:87] visit SofA
  I1223 11:07:56.153906  3677 8-cached-visit.cu:92] takes 113.638ms
 */
#include "helper.h"

const int K = 1 << 20;

__global__ void readOffset(float *a, float *b, float *c, int n, int offset) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int k = idx + offset;
  if (k < n)
    c[k] = a[k] + b[k];
}

// AofS
struct SomeStruct {
  float x;
  float y;
};

// SofA
struct SomeStruct1 {
  float *x;
  float *y;
};

__global__ void visitStructs(SomeStruct *arr, SomeStruct *res, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    SomeStruct tmp = arr[idx];
    tmp.x += 1.f;
    tmp.y += 2.f;
    res[idx] = tmp;
  }
}

__global__ void visitStructs1(SomeStruct1 a, SomeStruct1 b, int n) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < n) {
    float x = a.x[idx];
    float y = a.y[idx];
    b.x[idx] = x + 1.f;
    b.y[idx] = y + 2.f;
  }
}

void testStructMain() {
  // visit struct
  size_t bytes = sizeof(SomeStruct) * K;
  SomeStruct *a = static_cast<SomeStruct *>(malloc(bytes));

  dim3 block(256);
  dim3 grid((K + block.x - 1) / block.x);

  SomeStruct *da, *db;

  NV_CHECK(cudaMalloc(&da, bytes));
  NV_CHECK(cudaMalloc(&db, bytes));
  NV_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));

  {
    Timer timer;

    LOG(INFO) << "visit AofS";
    for (int i = 0; i < 1000; i++) {
      visitStructs<<<grid, block>>>(da, db, K);
    }
    LOG(INFO) << "takes " << timer.peek() << "ms";
  }

  // visit struct1
  bytes = sizeof(float) * K;
  SomeStruct1 s;
  SomeStruct1 ds, res;

  s.x = static_cast<float *>(malloc(bytes));
  s.y = static_cast<float *>(malloc(bytes));
  randVec(s.x, K);
  randVec(s.y, K);

  NV_CHECK(cudaMalloc(&ds.x, bytes));
  NV_CHECK(cudaMalloc(&ds.y, bytes));
  NV_CHECK(cudaMalloc(&res.x, bytes));
  NV_CHECK(cudaMalloc(&res.y, bytes));

  NV_CHECK(cudaMemcpy(ds.x, s.x, bytes, cudaMemcpyHostToDevice));
  NV_CHECK(cudaMemcpy(ds.y, s.y, bytes, cudaMemcpyHostToDevice));

  {
    LOG(INFO) << "visit SofA";
    Timer timer;
    for (int i = 0; i < 1000; i++) {
      visitStructs1<<<grid, block>>>(ds, res, K);
    }
    LOG(INFO) << "takes " << timer.peek() << "ms";
  }
}

void testOffset(float *a, float *b, float *c, int n, int offset) {
  Timer timer;

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  for (int i = 0; i < 1000; i++) {
    readOffset<<<grid, block>>>(a, b, c, n, offset);
  }

  LOG(INFO) << "test offset " << offset << " takes " << timer.peek();
}

void testOffsetMain() {
  float *a, *b, *c;
  float *da, *db, *dc;
  int num_elements = 1 << 20;
  int bytes = num_elements * sizeof(float);

  a = static_cast<float *>(malloc(bytes));
  b = static_cast<float *>(malloc(bytes));
  c = static_cast<float *>(malloc(bytes));

  NV_CHECK(cudaMalloc(&da, bytes));
  NV_CHECK(cudaMalloc(&db, bytes));
  NV_CHECK(cudaMalloc(&dc, bytes));

  randVec(a, num_elements);
  randVec(b, num_elements);
  randVec(c, num_elements);

  NV_CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  NV_CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
  NV_CHECK(cudaMemcpy(dc, c, bytes, cudaMemcpyHostToDevice));

  testOffset(da, db, dc, num_elements, 0);
  testOffset(da, db, dc, num_elements, 0);
  testOffset(da, db, dc, num_elements, 0);
  testOffset(da, db, dc, num_elements, 11);
  testOffset(da, db, dc, num_elements, 16);
  testOffset(da, db, dc, num_elements, 128);

  testStructMain();
}

void testStruct() {}

int main(int argc, char **argv) {
  testOffsetMain();

  return 0;
}
