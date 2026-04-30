cmake_minimum_required(VERSION 3.24)


project(HeatDiffusion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_executable(app
        main.cu
        Heat_Kernel.cu
        Header.h
)

set_target_properties(app PROPERTIES
        CUDA_ARCHITECTURES 120
)
