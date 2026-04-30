cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(Wave_Propagation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_executable(app
        main.cu
        Wave_Kernel.cu
        Header.h
        Wave_Kernel.cu
)

set_target_properties(app PROPERTIES
        CUDA_ARCHITECTURES 120
)
