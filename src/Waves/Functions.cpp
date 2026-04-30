#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

#include "Headers.h"

real *data() {
    real* p = (real*)malloc(constants::size * constants::size * sizeof(real));
    real* p_new = (real*)malloc(constants::size * constants::size * sizeof(real));
    real* p_old = (real*)malloc(constants::size * constants::size * sizeof(real)); //I allocate a 3 big arrays (not 2D array, important for optimisation)


#pragma omp parallel for //Parallel loop
    for (int i = 0; i < constants::size; i++) {
#pragma omp simd
        for (int j = 0; j < constants::size; j++) {
            if ((i - 500) * (i - 500) + (j - 500) * (j - 500) <= 100 ){
                p[i * constants::size + j] = 50000.00f;
                p_old[i * constants::size + j] = 50000.00f; //Set central region with lower pressure to create the wave
            }
            else {
                p[i * constants::size + j] = constants::normal_pressure;
                p_old[i * constants::size + j] = constants::normal_pressure; //Set all other parts of the grid to a normal atmospheric pressure
            }

        }
    }

    if (constants::coeficient > 0.5f) {
        std::cerr << "CFL violated\n";
        std::exit(EXIT_FAILURE); //Here I check for the extreme conditions
    }

    int total_frames = 240;
    int saveInterval = constants::steps/total_frames;
    int frame_saved = 0; //set number of frames and how often I have to save the frame


    std::cout<<"Simulation has started "<<std::endl;
    std::cout<<"Amount of steps "<< constants::steps <<std::endl;
    std::cout<<"Amount of frames "<<total_frames<<std::endl;
    std::cout << "Interval " << saveInterval << std::endl; //Output of the necessery data

for (int n = 0; n < constants::steps; n++) {
#pragma omp parallel for
    for (int ii = 1; ii < constants::size - 1; ii+=constants::square_size) {
        for (int jj = 1; jj < constants::size - 1; jj+=constants::square_size) {
            int i_max = std::min(ii + constants::square_size, constants::size - 1);
            int j_max = std::min(jj + constants::square_size, constants::size - 1);
            for (int i = ii; i < i_max; i++) {
#pragma omp simd
                for (int j = jj; j < j_max; j++) { //2 more loops for the tiling (firstly I go by blocks, then inside of them)
                    p_new[i*constants::size + j] = 2.0f * p[i*constants::size + j] -
                        p_old[i*constants::size + j] + constants::coeficient *
                            (p[(i-1)*constants::size + j] + p[i*constants::size + j - 1]
                                + p[(i+1)*constants::size + j] + p[i*constants::size + j + 1]
                                -4*p[i*constants::size + j]); //Finite differences method
                }

            }
        }
    }
    for (int k = 0; k < constants::size; k++) {
        p_new[k] = constants::normal_pressure;
        p_new[(constants::size - 1)*constants::size + k] = constants::normal_pressure; //set top and bottom edges to normal pressure
    }
    for (int k = 1; k < constants::size - 1; k++) {
        p_new[k*constants::size] = constants::normal_pressure;
        p_new[k*constants::size + constants::size - 1] = constants::normal_pressure; //Set the left and right edges to normal pressure
    }
    real* temp = p_old;
    p_old = p;
    p = p_new;
    p_new = temp; //swap the pressures 
    // p_old -> p, p - p_new, p_new -> p_old


    if (n%saveInterval == 0) {
        std::string filename = R"(C:\Users\giley\CLionProjects\2D_Wave_spreading\Bins\frame_)" + std::to_string(frame_saved) + ".bin";
        saveToBin(p, filename);
        frame_saved++;
    }
}

    free(p_new);
    free(p_old); //free alocated memory
    return p;
}

void saveToBin(real *p, const std::string &filename) {
    std::ofstream file (filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    file.write(reinterpret_cast<char*>(p), constants::size * constants::size * sizeof(real));
    if (!file) {
        std::cerr << "Error writing to file " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }
    file.close();

}