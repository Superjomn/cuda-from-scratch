cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

project(cuda_from_scratch LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
find_package(gflags REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

function(cuda_bin TARGET)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_bin "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  add_executable(${TARGET} ${cc_bin_SRCS})
  set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
  if(cc_bin_DEPS)
    add_dependencies(${TARGET} ${cc_bin_DEPS})
  endif()
  set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(${TARGET} gflags)
endfunction()

cuda_bin(0-add SRCS 0-vecadd-memory-coalesce.cu)
cuda_bin(1-transpose SRCS 1-matrix-transpose-coalesce.cu)
cuda_bin(2-reduce SRCS 2-reduce.cu)
cuda_bin(dump-cuda-props SRCS dump-gpu-props.cpp)
cuda_bin(warp-shuffle-quickstart SRCS warp-shuffle-quickstart.cu)

target_link_libraries(dump-cuda-props cuda cudart)
