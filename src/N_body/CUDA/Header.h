//
// Created by dmytro-hilei on 5/5/26.
//

#ifndef N_BODYPROBLEM_HEADER_H
#define N_BODYPROBLEM_HEADER_H
#include <cstdio>
#include <ctime>


#define filename "log.txt"

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float mass;
} Particle;

typedef struct {
    Particle *particles;
    int n;
} System;

namespace Constants {
    //Physics
    inline constexpr float G = 39.4784176f;  // 4*pi*pi
    inline constexpr float dt = 0.001f;      // years ≈ 8.76 hours
    inline constexpr float eps2 = 1e-8f;     // AU^2

    //cuda settings
    inline constexpr int NUM_BODIES = 100000;
    inline constexpr int BLOCK_SIZE = 256;
    inline constexpr int NUM_BLOCKS = (NUM_BODIES + BLOCK_SIZE - 1) / BLOCK_SIZE;

}

namespace HelperFunctions {
    inline void writeTimestamp(FILE* f) {
        std::time_t now = std::time(nullptr);
        std::tm* local = std::localtime(&now);

        if (local != nullptr) {
            char buffer[32];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local);
            std::fprintf(f, "[%s] ", buffer);
        }
    }


    inline void checkCudaErrors(
        cudaError_t status,
        const char *filename_log,
        const char *label) {
        if (status != cudaSuccess) {
            const char* msg = cudaGetErrorString(status);

            std::fprintf(stderr, "CUDA error at %s: %s\n", label, msg);

            FILE* f_error = std::fopen(filename_log, "a");
            if (f_error != nullptr) {
                writeTimestamp(f_error);
                std::fprintf(f_error, "CUDA error at %s: %s\n", label, msg);
                std::fclose(f_error);
            }

            std::exit(EXIT_FAILURE);
        }
    }
}



#endif //N_BODYPROBLEM_HEADER_H