cmake_minimum_required(VERSION 3.5)
project(cuda_from_scratch LANGUAGES CXX CUDA)
set(CUDA_NVCC_FLAGS "{CUDA_NVCC_FLAGS} -std=c++11}")

cuda_add_executable(hello 1-hello.cu)

