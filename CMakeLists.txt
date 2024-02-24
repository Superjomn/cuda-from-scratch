cmake_minimum_required(VERSION 3.5)
enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

project(cuda_from_scratch LANGUAGES CXX CUDA)

find_package(gflags REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

function(cuda_bin TARGET)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_bin "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET} ${cc_bin_SRCS})
    set_property(TARGET ${TARGET} PROPERTY CUDA_ARCHITECTURES OFF)
    if (cc_bin_DEPS)
        add_dependencies(${TARGET} ${cc_bin_DEPS})
    endif()
    set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endfunction()

cuda_bin(0-add SRCS 0-vecadd-memory-coalesce.cu)
target_link_libraries(0-add gflags)