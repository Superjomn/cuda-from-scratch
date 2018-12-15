main: hello sum-array sum-matrix-2d-grid-2d-block

hello: 1-hello.cu
	nvcc 1-hello.cu -o hello

sum-array: 2-sum-array.cu
	nvcc 2-sum-array.cu -o sum-array -std=c++11 -lglog

sum-matrix-2d-grid-2d-block: 3-sum-matrix-2d-grid-2d-block.cu
	nvcc 3-sum-matrix-2d-grid-2d-block.cu -o sum-matrix-2d-grid-2d-block
