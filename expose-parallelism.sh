#!/bin/bash
set -x

nv_command="nvprof --metrics achieved_occupancy"

./expose-parallelism 256 256
./expose-parallelism 32 64
./expose-parallelism 32 32
./expose-parallelism 32 16
./expose-parallelism 16 32
./expose-parallelism 16 16

# found
:<<EOF
+ ./expose-parallelism 256 256
I1217 08:30:52.001938  4459 6-expose-parallelism.cu:45] time 0.0870228
+ ./expose-parallelism 32 64
I1217 08:30:52.133328  4466 6-expose-parallelism.cu:45] time 0.0839233
+ ./expose-parallelism 32 32
I1217 08:30:52.264940  4473 6-expose-parallelism.cu:45] time 0.175953
+ ./expose-parallelism 32 16
I1217 08:30:52.398440  4480 6-expose-parallelism.cu:45] time 0.176907
+ ./expose-parallelism 16 32
I1217 08:30:52.529139  4487 6-expose-parallelism.cu:45] time 0.176907
+ ./expose-parallelism 16 16
I1217 08:30:52.661077  4494 6-expose-parallelism.cu:45] time 0.180006

found that

the one with block size 32x64 is the best.
- the 1-th get the most thread number in a block, but not the best
- the last one get the least threads in a block, and is the worest.
EOF

# 1- Checking Active Warps with nvprof

$nv_command ./expose-parallelism 256 256
$nv_command ./expose-parallelism 32 64
$nv_command ./expose-parallelism 32 32
$nv_command ./expose-parallelism 32 16
$nv_command ./expose-parallelism 16 32
$nv_command ./expose-parallelism 16 16

:<<EOF
==4736== Profiling application: ./expose-parallelism 32 32
==4750== Profiling application: ./expose-parallelism 32 16
==4764== Profiling application: ./expose-parallelism 16 32
==4778== Profiling application: ./expose-parallelism 16 16
Invocations                               Metric Name                        Metric Description         Min         Max         Avg
Device "GeForce GTX 1060 6GB (0)"
    Kernel: sumMatrix2Dgrid(float*, float*, float*, int, int)
          1                        achieved_occupancy                        Achieved Occupancy    0.790099    0.790099    0.790099

          1                        achieved_occupancy                        Achieved Occupancy    0.814953    0.814953    0.814953
          1                        achieved_occupancy                        Achieved Occupancy    0.858136    0.858136    0.858136
          1                        achieved_occupancy                        Achieved Occupancy    0.893015    0.893015    0.893015

the occupancy of 32x32 is not the best, but it get the best performance, so some other factors also works.
EOF

# 2- Checking Memory Operations with nvprof
# there are two load and one store operations

# first profile the load performance
nv_command="nvprof --metrics gld_throughput"

$nv_command ./expose-parallelism 256 256
$nv_command ./expose-parallelism 32 64
$nv_command ./expose-parallelism 32 32
$nv_command ./expose-parallelism 32 16
$nv_command ./expose-parallelism 16 32
$nv_command ./expose-parallelism 16 16

nv_command="nvprof --metrics gld_efficiency"
$nv_command ./expose-parallelism 256 256
$nv_command ./expose-parallelism 32 64
$nv_command ./expose-parallelism 32 32
$nv_command ./expose-parallelism 32 16
$nv_command ./expose-parallelism 16 32
$nv_command ./expose-parallelism 16 16


:<<EOF
==5022== NVPROF is profiling process 5022, command: ./expose-parallelism 32 32
==5039== NVPROF is profiling process 5039, command: ./expose-parallelism 32 16
==5056== NVPROF is profiling process 5056, command: ./expose-parallelism 16 32
==5073== NVPROF is profiling process 5073, command: ./expose-parallelism 16 16

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
          1                            gld_throughput                    Global Load Throughput  91.668GB/s  91.668GB/s  91.667GB/s
          1                            gld_throughput                    Global Load Throughput  90.278GB/s  90.278GB/s  90.278GB/s
          1                            gld_throughput                    Global Load Throughput  90.647GB/s  90.647GB/s  90.647GB/s
          1                            gld_throughput                    Global Load Throughput  88.649GB/s  88.649GB/s  88.649GB/s

Invocations                               Metric Name                        Metric Description         Min         Max         Avg
          1                            gld_efficiency             Global Memory Load Efficiency     100.00%     100.00%     100.00%

The first get the best load performance.

All the tests takes 100% of the device's load bandwidth.
EOF
