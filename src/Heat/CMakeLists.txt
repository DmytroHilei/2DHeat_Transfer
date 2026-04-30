cmake_minimum_required(VERSION 3.31)
project(2DheatTransfer)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-Og")

find_package(OpenMP REQUIRED)

add_executable(2DheatTransfer main.cpp Fucntions.cpp)

target_link_libraries(2DheatTransfer PRIVATE OpenMP::OpenMP_CXX)



