cmake_minimum_required(VERSION 4.1)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(N_bodyProblem LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(N_bodyProblem
        main.cu
        main_kernel.cu
        Header.h
)

set_target_properties(N_bodyProblem PROPERTIES
        CUDA_ARCHITECTURES 120)

