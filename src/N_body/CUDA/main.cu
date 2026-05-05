//
// Created by dmytro-hilei on 5/5/26.
//

#include "Header.h"

#include <cstdlib>
#include <cmath>
#include <algorithm>

__global__ void N_Body_problem(
    Particle *particles_old,
    Particle *particles_new,
    int N_particles,
    float dt,
    float G,
    float eps2
);

void initParticles(Particle* particles, int N) {
    // Sun-like central massive body
    particles[0].x = 0.0f;
    particles[0].y = 0.0f;
    particles[0].z = 0.0f;

    particles[0].vx = 0.0f;
    particles[0].vy = 0.0f;
    particles[0].vz = 0.0f;

    particles[0].mass = 1.0f;

    // Other particles roughly on circular-ish orbits
    for (int i = 1; i < N; i++) {
        float angle = 2.0f * 3.1415926535f * static_cast<float>(i) / static_cast<float>(N);

        // Spread particles from ~0.4 AU to ~30 AU
        float r = 0.4f + 29.6f * static_cast<float>(i) / static_cast<float>(N);

        particles[i].x = r * std::cos(angle);
        particles[i].y = r * std::sin(angle);
        particles[i].z = 0.0f;

        // Circular orbit velocity around central mass M = 1
        float v = std::sqrt(Constants::G / r);

        particles[i].vx = -v * std::sin(angle);
        particles[i].vy =  v * std::cos(angle);
        particles[i].vz = 0.0f;

        // Tiny mass, asteroid-like
        particles[i].mass = 1e-12f;
    }
}

void saveSample(const Particle* particles, int N, const char* filename_out) {
    FILE* f = std::fopen(filename_out, "w");

    if (f == nullptr) {
        std::fprintf(stderr, "Could not open output file: %s\n", filename_out);
        return;
    }

    std::fprintf(f, "x,y,z,vx,vy,vz,mass\n");

    int sampleN = std::min(N, 1000);

    for (int i = 0; i < sampleN; i++) {
        std::fprintf(
            f,
            "%f,%f,%f,%f,%f,%f,%e\n",
            particles[i].x,
            particles[i].y,
            particles[i].z,
            particles[i].vx,
            particles[i].vy,
            particles[i].vz,
            particles[i].mass
        );
    }

    std::fclose(f);
}

int main() {
    using namespace Constants;
    using namespace HelperFunctions;

    const int N = NUM_BODIES;
    const int steps = 100;
    const size_t bytes = static_cast<size_t>(N) * sizeof(Particle);

    std::printf("N-body simulation\n");
    std::printf("Particles: %d\n", N);
    std::printf("Steps: %d\n", steps);
    std::printf("Memory per buffer: %.2f MB\n", bytes / 1024.0 / 1024.0);

    Particle* h_particles = static_cast<Particle*>(std::malloc(bytes));

    if (h_particles == nullptr) {
        std::fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    initParticles(h_particles, N);

    Particle* d_particles_old = nullptr;
    Particle* d_particles_new = nullptr;

    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void**>(&d_particles_old), bytes),
        filename,
        "cudaMalloc d_particles_old"
    );

    checkCudaErrors(
        cudaMalloc(reinterpret_cast<void**>(&d_particles_new), bytes),
        filename,
        "cudaMalloc d_particles_new"
    );

    checkCudaErrors(
        cudaMemcpy(d_particles_old, h_particles, bytes, cudaMemcpyHostToDevice),
        filename,
        "cudaMemcpy host to d_particles_old"
    );

    checkCudaErrors(
        cudaMemcpy(d_particles_new, h_particles, bytes, cudaMemcpyHostToDevice),
        filename,
        "cudaMemcpy host to d_particles_new"
    );

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start), filename, "cudaEventCreate start");
    checkCudaErrors(cudaEventCreate(&stop), filename, "cudaEventCreate stop");

    checkCudaErrors(cudaEventRecord(start), filename, "cudaEventRecord start");

    for (int step = 0; step < steps; step++) {
        N_Body_problem<<<NUM_BLOCKS, BLOCK_SIZE>>>(
            d_particles_old,
            d_particles_new,
            N,
            dt,
            G,
            eps2
        );

        checkCudaErrors(
            cudaGetLastError(),
            filename,
            "N_Body_problem kernel launch"
        );

        checkCudaErrors(
            cudaDeviceSynchronize(),
            filename,
            "N_Body_problem kernel execution"
        );

        std::swap(d_particles_old, d_particles_new);
    }

    checkCudaErrors(cudaEventRecord(stop), filename, "cudaEventRecord stop");
    checkCudaErrors(cudaEventSynchronize(stop), filename, "cudaEventSynchronize stop");

    float ms = 0.0f;

    checkCudaErrors(
        cudaEventElapsedTime(&ms, start, stop),
        filename,
        "cudaEventElapsedTime"
    );

    std::printf("GPU simulation time: %.3f ms\n", ms);
    std::printf("Time per step: %.3f ms\n", ms / static_cast<float>(steps));

    checkCudaErrors(
        cudaMemcpy(h_particles, d_particles_old, bytes, cudaMemcpyDeviceToHost),
        filename,
        "cudaMemcpy device to host"
    );

    saveSample(h_particles, N, "particles_sample.csv");

    checkCudaErrors(cudaEventDestroy(start), filename, "cudaEventDestroy start");
    checkCudaErrors(cudaEventDestroy(stop), filename, "cudaEventDestroy stop");

    checkCudaErrors(cudaFree(d_particles_old), filename, "cudaFree d_particles_old");
    checkCudaErrors(cudaFree(d_particles_new), filename, "cudaFree d_particles_new");

    std::free(h_particles);

    std::printf("Saved sample to particles_sample.csv\n");

    return EXIT_SUCCESS;
}