cmake_minimum_required(VERSION 3.20)

project(2DheatTransfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(2DheatTransfer
        main.cpp
        Fucntions.cpp
)

find_package(OpenMP REQUIRED)

target_link_libraries(2DheatTransfer PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(2DheatTransfer PRIVATE
        $<$<CONFIG:Debug>:-Og>
        $<$<CONFIG:Release>:-O3 -march=native -mtune=native>
)



