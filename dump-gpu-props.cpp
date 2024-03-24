#include "./common.cuh"
#include <cuda_runtime.h>
#include <gflags/gflags.h>

void dumpGpuProps()
{
    // Print GPU properties

    int dev;
    cudaGetDevice(&dev);

    // ==== grid and block size limits ====

    int xthreadlimit, ythreadlimit, zthreadlimit;
    int xgridlimit, ygridlimit, zgridlimit;
    NVCHECK(cudaDeviceGetAttribute(&xthreadlimit, cudaDevAttrMaxBlockDimX, dev));
    NVCHECK(cudaDeviceGetAttribute(&ythreadlimit, cudaDevAttrMaxBlockDimY, dev));
    NVCHECK(cudaDeviceGetAttribute(&zthreadlimit, cudaDevAttrMaxBlockDimZ, dev));
    NVCHECK(cudaDeviceGetAttribute(&xgridlimit, cudaDevAttrMaxGridDimX, dev));
    NVCHECK(cudaDeviceGetAttribute(&ygridlimit, cudaDevAttrMaxGridDimY, dev));
    NVCHECK(cudaDeviceGetAttribute(&zgridlimit, cudaDevAttrMaxGridDimZ, dev));

    std::cerr << "Device " << dev << " properties:" << std::endl;
    std::cerr << "  Max block dimensions: " << xthreadlimit << " x " << ythreadlimit << " x " << zthreadlimit
              << std::endl;
    std::cerr << "  Max grid dimensions: " << xgridlimit << " x " << ygridlimit << " x " << zgridlimit << std::endl;

    // ==== shared memory ====
    cudaSharedMemConfig sConfig;
    cudaDeviceGetSharedMemConfig(&sConfig);
    switch (sConfig)
    {
    case cudaSharedMemBankSizeDefault: std::cerr << "  Shared memory bank size: default" << std::endl; break;
    case cudaSharedMemBankSizeFourByte: std::cerr << "  Shared memory bank size: 4 bytes" << std::endl; break;
    case cudaSharedMemBankSizeEightByte: std::cerr << "  Shared memory bank size: 8 bytes" << std::endl; break;
    }

    int x;
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    std::cerr << "  Max shared memory per block: " << x << " bytes" << std::endl;
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMaxRegistersPerBlock, dev));
    std::cerr << "  Max registers per block: " << x << std::endl;

    // ==== warp and thread limits ====
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrWarpSize, dev));
    std::cerr << "  Warp size: " << x << std::endl;

    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMultiProcessorCount, dev));
    std::cerr << "  Multiprocessor count: " << x << std::endl;
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    std::cerr << "  Max resident threads per multiprocessor: " << x << " = " << x / 32 << " warps" << std::endl;

    // ==== L1 and L2 cache ====
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrL2CacheSize, dev));
    std::cerr << "  L2 cache size: " << x << " bytes" << std::endl;
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrGlobalL1CacheSupported, dev));
    std::cerr << "  Global L1 cache supported: " << (x ? "yes" : "no") << std::endl;

    // ==== memory ====
    NVCHECK(cudaDeviceGetAttribute(&x, cudaDevAttrGlobalMemoryBusWidth, dev));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "  Total global memory: " << deviceProp.totalGlobalMem / 1024 / 1024 / 1024 << " GB" << std::endl;

    std::cerr << "  Processor clock: " << deviceProp.clockRate / 1000 / 1000 << " MHZ" << std::endl;
    std::cerr << "  Memory clock: " << deviceProp.memoryClockRate / 1000 / 1000 << " MHZ" << std::endl;
}

int main()
{
    dumpGpuProps();
    return 0;
}
