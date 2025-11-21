#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Headers.h"

int N = 1000;
real dh = 0.01f;
real alpha = 0.005f;
real t = 200.00f;
real dt = 0.0032f;
int32_t step = static_cast<int32_t>(t / dt);
real coefficient = alpha * dt / (dh * dh);
int B = 4;

real *data() {
    real* T = (real*)malloc(N * N * sizeof(real));
    real* Tnew = (real*)malloc(N * N * sizeof(real));


    std::fill(T, T + N*N, 0.0f);
    std::fill(Tnew, Tnew + N*N, 0.0f);
#pragma omp parallel for
    for (int i = 300; i < 700; i++) {
#pragma omp simd
        for (int j = 300; j < 700; j++) {
            if ((i - 500)*(i - 500) + (j - 500)*(j - 500) <= 40000) {
                T[i*N + j] = 100.0f;
            }
        }
    }
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
#pragma omp simd
        for (int j = 0; j < N; j++) {
            if (i <= 100 && j >= 200 && j <= 800) {
                T[i*N + j] = 100.0f;
            }
        }
    }

    real r = alpha * dt / (dh * dh);
    std::cout << "Stability parameter r = " << r << std::endl;
    if (r > 0.25f) {
        std::cout << "WARNING: Unstable! r should be <= 0.25" << std::endl;
    }

    int totalFrames = 240;
    int saveInterval = step/totalFrames;
    int frameSaved = 0;

    std::cout<<"Simulation has started "<<std::endl;
    std::cout<<"Amount of steps "<< step <<std::endl;
    std::cout<<"Amount of frames "<<totalFrames<<std::endl;
    std::cout << "Interval " << saveInterval << std::endl;

    // Compute new temperatures

    for (int n = 0; n < step; n++) {
        // Compute new temperatures with proper boundary handling
#pragma omp parallel for
        for (int ii = 1; ii < N - 1; ii+=B) {
            for (int jj = 1; jj < N - 1; jj+=B) {
                int i_max = std::min(ii + B, N - 1);
                int j_max = std::min(jj + B, N - 1);
                for (int i = ii; i < i_max; i++) {
#pragma omp simd
                    for (int j = jj; j < j_max; j++) {
                        Tnew[i*N + j] = T[i*N + j] + coefficient *
                        (T[(i+1)*N + j] + T[(i-1)*N + j] + T[i*N +j + 1] + T[i*N + j - 1] - 4.0f * T[i*N + j]);
                    }

                }
            }// 32 x 32 tiling
            //We just divide the matrix into smaller ones

        }
        for (int k = 0; k < N; k++) {
            Tnew[k]     = 0.0f;
            Tnew[(N - 1)*N + k]   = 0.0f;
        }
        for (int k = 1; k < N - 1; k++) {
            Tnew[k*N] = 0.0f;
            Tnew[(N - 1)*N + k] = 0.0f;
        }
        real* temp = T;
        T = Tnew;
        Tnew = temp;

        if (n % saveInterval == 0) {
#pragma omp barrier
            std::string filename = R"(C:\Users\giley\2DHeatTransferBins\frame_)"
                     + std::to_string(frameSaved) + ".bin";
            saveToBin(T, filename);
            frameSaved++;
        }
    }

    free(Tnew);
    return T;
}
// I have to do the checking for steady elements to reduce the number of the iterations

void SaveToCSV(real *T, const std::string &filename)
{
    std::ofstream file (filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    file << "x,y,Temperature\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            file << i * dh << "," << j * dh << "," << T[i*N + j] << "\n";
        }
    }
    file.close();
}

void saveToBin(real *T, const std::string &filename) {
    std::ofstream file (filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(1);
    }

    file.write(reinterpret_cast<char*>(T), N * N *sizeof(real));
    if (!file) {
        std::cout << "Error writing to file " << filename << std::endl;
        exit(1);
    }
}
