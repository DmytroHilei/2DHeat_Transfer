#include <chrono>
#include <iostream>
#include <omp.h>
#include <ostream>

#include "Headers.h"
int main() {
    auto start = std::chrono::high_resolution_clock::now(); //Start time before computing

    auto p = data(); //All calulations

    auto end = std::chrono::high_resolution_clock::now(); //End time after calculations
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << " seconds" << std::endl;



#pragma omp parallel
    {
#pragma omp single
        std::cout << "Threads: " << omp_get_num_threads() << "\n"; //I was checking if the omP works fine
    }
    return 0;
}