cmake_minimum_required(VERSION 3.31)
project(2D_Wave_Spreading)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")

find_package(OpenMP REQUIRED)

add_executable(2D_Wave_Spreading main.cpp
        Functions.cpp
        Headers.h)

target_link_libraries(2D_Wave_Spreading PRIVATE OpenMP::OpenMP_CXX)