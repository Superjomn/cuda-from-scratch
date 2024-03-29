#include "common.cuh"
#include <cassert>
#include <gflags/gflags.h>

__global__ void reduce0(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Read a block of data into shared memory collectively
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // ISSUE: divergent warps
        if (tid % (2 * stride) == 0)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads(); // need to sync per level
    }

    // Write the result for this block to global memory
    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

using kernel_fn = void (*)(int*, int*, unsigned int);

// Avoid divergent warps
__global__ void reduce1(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            // Issue: bank conflict
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Avoid bank conflict
__global__ void reduce2(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce3(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Avoid divergent warps
__global__ void reduce1_1(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];
    const uint32_t block_size = blockDim.x * 2;

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * block_size + threadIdx.x;

    // sdata[tid] = (i < n) ? g_idata[i] : 0;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            // Issue: bank conflict
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__device__ void warpReduce(volatile int* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reduce4(int* g_idata, int* g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(sdata, tid);

    if (tid == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// using warp shuffle instruction
// From book <Professional CUDA C Programming>
__inline__ __device__ int warpReduce(int mySum)
{
    mySum += __shfl_xor(mySum, 16);
    mySum += __shfl_xor(mySum, 8);
    mySum += __shfl_xor(mySum, 4);
    mySum += __shfl_xor(mySum, 2);
    mySum += __shfl_xor(mySum, 1);
    return mySum;
}

__global__ void reduceShfl(int* g_idata, int* g_odata, unsigned int n)
{
    // Helps to share data between warps
    // size should be (blockDim.x / warpSize)
    extern __shared__ int sdata[];

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Necessary to make sure shfl instruction is not used with uninitialized data
    int mySum = idx < n ? g_idata[idx] : 0;

    int lane = threadIdx.x % warpSize;
    int warp = threadIdx.x / warpSize;

    mySum = warpReduce(mySum);

    if (lane == 0)
    {
        sdata[warp] = mySum;
    }

    __syncthreads();

    // last warp reduce
    mySum = (threadIdx.x < blockDim.x / warpSize) ? sdata[lane] : 0;
    if (warp == 0)
    {
        mySum = warpReduce(mySum);
    }

    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = mySum;
    }
}

DEFINE_int32(n, 1 << 20, "Number of elements in the input array");
DEFINE_int32(block_size, 512, "Number of threads per block");
DEFINE_int32(kernel, 0, "Kernel to run");
DEFINE_bool(profile, false, "Profile the kernel");

int launch_reduce(int* g_idata, int* g_odata, unsigned int n, int block_size, kernel_fn kernel, cudaStream_t stream,
    uint32_t num_blocks = 0, uint32_t smem_size = 0)
{
    int* idata = g_idata;
    int* odata = g_odata;
    if (smem_size == 0)
        smem_size = block_size * sizeof(int);

    // Calculate number of blocks
    num_blocks = (num_blocks > 0) ? num_blocks : (n + block_size - 1) / block_size;

    if (!FLAGS_profile)
        printf("- launching: num_blocks: %d, block_size:%d, n:%d\n", num_blocks, block_size, n);

    int level = 0;
    // Launch the kernel
    kernel<<<num_blocks, block_size, smem_size, stream>>>(idata, odata, n);
    if (!FLAGS_profile)
        cudaStreamSynchronize(stream);

    level++;

    // Recursively reduce the partial sums
    while (num_blocks > 1)
    {

        std::swap(idata, odata);
        n = num_blocks;
        num_blocks = (n + block_size - 1) / block_size;
        kernel<<<num_blocks, block_size, smem_size, stream>>>(idata, odata, n);
        if (!FLAGS_profile)
            cudaStreamSynchronize(stream);
    }

    // Copy the final result back to the host
    int h_out;
    NVCHECK(cudaMemcpyAsync(&h_out, odata, sizeof(int), cudaMemcpyDeviceToHost, stream));

    return h_out;
}

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    dim3 block(FLAGS_block_size);
    dim3 grid((FLAGS_n + block.x - 1) / block.x);

    Vector<int> input(FLAGS_n);
    Vector<int> output(FLAGS_n / block.x);
    input.blocked(4);

    // Both the input and output buffer will be altered, so we need to calculate
    // the expected value before running the kernel
    int expected = 0;
    auto host_input = input.toHost();
    for (int i = 0; i < FLAGS_n; i++)
    {
        expected += host_input[i];
    }

    const size_t smem_size = block.x * sizeof(int);

    int reduce_result = 0;

    auto add_wrapped = [&](cudaStream_t stream) -> void
    {
        switch (FLAGS_kernel)
        {
        case 0: reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce0, stream); break;
        case 1: reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce1, stream); break;
        case 2: reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce2, stream); break;
        case 3:
        {
            grid.x = (FLAGS_n + block.x * 2 - 1) / (block.x * 2);
            reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce3, stream, grid.x);
        }
        break;
        case 4:
        {
            grid.x = (FLAGS_n + block.x * 2 - 1) / (block.x * 2);
            reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce4, stream, grid.x);
        }
        break;
        case 5:
        {
            grid.x = (FLAGS_n + block.x * 2 - 1) / (block.x * 2);
            reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduce1_1, stream, grid.x);
        }
        break;

        case 6:
        {
            int numWarps = block.x / 32;
            reduce_result = launch_reduce(input.data, output.data, FLAGS_n, block.x, reduceShfl, stream, 0, numWarps);
        }
        break;
        }
    };

    cudaStream_t stream;
    NVCHECK(cudaStreamCreate(&stream));

    if (FLAGS_profile)
    {
        std::cerr << "GTX 3080 memory bandwidth: 760GB/s" << std::endl;
        float time = measure_performance<void>(add_wrapped, stream);
        float memory_bandwidth = FLAGS_n * sizeof(int) / time / 1e9 * 1e3;
        std::cerr << "Kernel: " << FLAGS_kernel << " time: " << time << " ms"
                  << " bandwidth: " << memory_bandwidth << " GB/s" << std::endl;
    }
    else
    {
        add_wrapped(stream);
    }

    cudaStreamSynchronize(stream);

    if (!FLAGS_profile)
    {
        // check result
        printf("Expected: %d, Got: %d\n", expected, reduce_result);
        assert(reduce_result == expected);
    }

    return 0;
}
