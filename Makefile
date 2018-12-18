main: hello sum-array sum-matrix-2d-grid-2d-block\
warp-divergence \
device-info \
expose-parallelism

flags = -lglog --ptxas-options=-v -std=c++11

hello: 1-hello.cu
	nvcc 1-hello.cu -o hello

sum-array: 2-sum-array.cu
	nvcc 2-sum-array.cu -o sum-array $(flags)

sum-matrix-2d-grid-2d-block: 3-sum-matrix-2d-grid-2d-block.cu
	nvcc 3-sum-matrix-2d-grid-2d-block.cu -o sum-matrix-2d-grid-2d-block -std=c++11 -lglog

warp-divergence: 4-warp-divergence.cu
	nvcc 4-warp-divergence.cu -o warp-divergence -std=c++11 -lglog --ptxas-options=-v

device-info: 5-device-info.cu
	nvcc 5-device-info.cu -o device-info -std=c++11 -lglog --ptxas-options=-v

expose-parallelism: target := expose-parallelism
expose-parallelism: 6-expose-parallelism.cu
	nvcc 6-$(target).cu -o $(target) $(flags)
