#include <chrono>
#include <iostream>
#include <omp.h>
#include <ostream>

#include "Headers.h"
#include <vector>

int main() {
    auto start = std::chrono::system_clock::now();

    auto T = data();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<real> diff = end - start;

    std::cout << "time used to run the program: " << diff.count() << " s" << std::endl;

#pragma omp parallel
    {
#pragma omp single
        std::cout << "Threads: " << omp_get_num_threads() << "\n";
    }

    return 0;
}