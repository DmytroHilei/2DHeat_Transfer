#include <ctime>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <string>

#include "Header.h"

namespace {
    constexpr int H = 1000;
    constexpr int W = 1000;

    constexpr int BLOCK_SIZE_X = 32;
    constexpr int BLOCK_SIZE_Y = 16;

    constexpr int NUM_BLOCKS_H = (H + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
    constexpr int NUM_BLOCKS_W = (W + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;

    constexpr float sound_velocity = 340.0f;
    constexpr float dx = 0.01f;
    constexpr float time = 0.1f;
    constexpr float dt = 0.00001f;
    constexpr float coefficient = (sound_velocity * dt / dx) * (sound_velocity * dt / dx);
    //constexpr float norm_pressure = 100000.00f;
    //constexpr float lower_pressure = 50000.00f;
    constexpr float num_steps = time / dt;

    constexpr int num_frames = 240;
    constexpr int save_interval = num_steps/num_frames;
    void writeTimestamp(FILE* f) {
        std::time_t now = std::time(nullptr);
        std::tm* local = std::localtime(&now);

        if (local != nullptr) {
            char buffer[32];
            std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local);
            std::fprintf(f, "[%s] ", buffer);
        }
    }

    void checkCUDA(
    cudaError_t status,
    const char* filename_error,
    const char* label
    )
        {
        if (status != cudaSuccess) {
            const char* msg = cudaGetErrorString(status);

            std::fprintf(stderr, "CUDA error at %s: %s\n", label, msg);

            FILE* f_error = std::fopen(filename_error, "a");
            if (f_error != nullptr) {
                writeTimestamp(f_error);
                std::fprintf(f_error, "CUDA error at %s: %s\n", label, msg);
                std::fclose(f_error);
            }

            std::exit(EXIT_FAILURE);
        }
    }

    int checkPhysicsConstants(const char* filename_error, const char* filename_success) {
        float cfl = sound_velocity * dt / dx;
        float cfl_limit = 1.0f / sqrtf(2.0f);  // ~0.707 для 2D
        float coef = sound_velocity * sound_velocity * dt * dt / (dx * dx);

        if (cfl > cfl_limit) {
            std::fprintf(stderr,
                "Unstable: CFL = %.4f > %.4f (limit)\n",
                cfl, cfl_limit
            );

            FILE* f_err = std::fopen(filename_error, "a");
            if (f_err != nullptr) {
                writeTimestamp(f_err);
                std::fprintf(f_err,
                    "Unstable: CFL = %.4f > %.4f, c=%.4f, dt=%.6f, dx=%.4f\n",
                    cfl, cfl_limit, sound_velocity, dt, dx
                );
                std::fclose(f_err);
            }

            return 0;
        }

        FILE* f_succ = std::fopen(filename_success, "a");
        if (f_succ != nullptr) {
            writeTimestamp(f_succ);
            std::fprintf(f_succ,
                "Physics OK: CFL = %.4f, coefficient = %.4f\n",
                cfl, coef
            );
            std::fclose(f_succ);
        }

        return 1;
    }


    void saveFrameBin(float *T, const std::string &filename) {
        std::ofstream file (filename, std::ios::binary);
        if (!file.is_open()) {
            std::fprintf(stderr, "Could not open %s\n", filename.c_str());
            std::exit(EXIT_FAILURE);
        }


        file.write(reinterpret_cast<char*>(T),
            W * H * sizeof(float));

        if (!file) {
            std::fprintf(stderr, "Could not write to %s\n", filename.c_str());
            std::exit(EXIT_FAILURE);
        }
    }

}

int main() {
    if (!checkPhysicsConstants(FILENAME_ERR, FILENAME_SUCC)) {
        return EXIT_FAILURE;
    }

    const auto bytes_grid = static_cast<size_t>(H * W * sizeof(float));
    auto* h_p     = (float*)malloc(bytes_grid);
    auto* h_p_old = (float*)malloc(bytes_grid);

    const float center_y = (H - 1) / 2.0f;
    const float center_x = (W - 1) / 2.0f;
    const float sigma = 20.0f;

    for (size_t i = 0; i < H; i++) {
        for (size_t j = 0; j < W; j++) {
            size_t idx = i * W + j;
            float dy = static_cast<float>(i) - center_y;
            float dx = static_cast<float>(j) - center_x;
            float val = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            h_p[idx]     = val;
            h_p_old[idx] = val;
        }
    }

    // GPU буфери — три шари
    float *d_p_old = nullptr;
    float *d_p     = nullptr;
    float *d_p_new = nullptr;

    checkCUDA(cudaMalloc(&d_p_old, bytes_grid), FILENAME_ERR, "cudaMalloc p_old");
    checkCUDA(cudaMalloc(&d_p,     bytes_grid), FILENAME_ERR, "cudaMalloc p");
    checkCUDA(cudaMalloc(&d_p_new, bytes_grid), FILENAME_ERR, "cudaMalloc p_new");

    checkCUDA(cudaMemcpy(d_p_old, h_p_old, bytes_grid, cudaMemcpyHostToDevice), FILENAME_ERR, "cudaMemcpy p_old");
    checkCUDA(cudaMemcpy(d_p,     h_p,     bytes_grid, cudaMemcpyHostToDevice), FILENAME_ERR, "cudaMemcpy p");
    checkCUDA(cudaMemcpy(d_p_new, h_p,     bytes_grid, cudaMemcpyHostToDevice), FILENAME_ERR, "cudaMemcpy p_new");

    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid(NUM_BLOCKS_W, NUM_BLOCKS_H);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int frame = 0;

    for (size_t t = 0; t < num_steps; t++) {
        Wave_Kernel<<<grid, block>>>(d_p_old, d_p_new, d_p, H, W, coefficient);

        checkCUDA(cudaGetLastError(), FILENAME_ERR, "kernel launch");

        // три шари: old <- current <- new
        std::swap(d_p_old, d_p);
        std::swap(d_p,     d_p_new);

        int target_step = frame * save_interval;
        if (frame < num_frames && t == target_step) {
            checkCUDA(cudaDeviceSynchronize(), FILENAME_ERR, "cudaDeviceSynchronize");
            checkCUDA(cudaMemcpy(h_p, d_p, bytes_grid, cudaMemcpyDeviceToHost), FILENAME_ERR, "cudaMemcpy");
            std::string filename = "bins/frame_" + std::to_string(frame) + ".bin";
            saveFrameBin(h_p, filename);
            frame++;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::printf("GPU elapsed time: %f milliseconds\n", milliseconds);

    checkCUDA(cudaFree(d_p_old), FILENAME_ERR, "cudaFree p_old");
    checkCUDA(cudaFree(d_p),     FILENAME_ERR, "cudaFree p");
    checkCUDA(cudaFree(d_p_new), FILENAME_ERR, "cudaFree p_new");
    d_p_old = d_p = d_p_new = nullptr;

    free(h_p);
    free(h_p_old);
    h_p = h_p_old = nullptr;

    return 0;
}

