#include <ctime>
#include <iostream>
#include "Header.h"
#include <fstream>
#include <string>

namespace {
	constexpr int H = 1000;
	constexpr int W = 1000;

	constexpr int BLOCK_SIZE_X = 32;
	constexpr int BLOCK_SIZE_Y = 16;

	constexpr int NUM_BLOCKS_H = (H + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
	constexpr int NUM_BLOCKS_W = (W + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;

	constexpr float alpha = 0.005f;
	constexpr float dt = 0.0032f;
	constexpr float dx = 0.01f;
	constexpr float T_max = 100.0f;
	constexpr float T_min = 0.0f;
	constexpr float Time = 200.0f;
	constexpr int32_t num_steps = Time / dt;
	constexpr int16_t num_frames = 240;
	constexpr int32_t save_interval = num_steps/num_frames;
	constexpr float r = alpha * dt / (dx * dx);

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
) {
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
		float r = alpha * dt / (dx * dx);

		if (r > 0.25f) {
			std::fprintf(stderr,
				"Unstable physics constants: r = %f, must be <= 0.25\n",
				r
			);

			FILE* f_err = std::fopen(filename_error, "a");
			if (f_err != nullptr) {
				writeTimestamp(f_err);
				std::fprintf(f_err,
					"Unstable physics constants: r = %f, must be <= 0.25\n",
					r
				);
				std::fclose(f_err);
			}

			return 0;
		}

		FILE* f_succ = std::fopen(filename_success, "a");
		if (f_succ != nullptr) {
			writeTimestamp(f_succ);
			std::fprintf(f_succ, "Physics constants OK: r = %f\n", r);
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
	checkPhysicsConstants(FILENAME_ERR, FILENAME_SUCC);

	const auto bytes_grid = static_cast<size_t>(H * W * sizeof(float));
	const float R = 100.0f;
	auto *h_T = (float*)malloc(bytes_grid);

	const float center_y = (H - 1) / 2.0f;
	const float center_x = (W - 1) / 2.0f;

	for (size_t i = 0; i < H; i++) {
    	for (size_t j = 0; j < W; j++) {
        	size_t index = i * W + j;

        	float dy = static_cast<float>(i) - center_y;
        	float dx = static_cast<float>(j) - center_x;

        	if (dx * dx + dy * dy <= R * R) {
            	h_T[index] = T_max;
				continue;
        } 	else {
            h_T[index] = T_min;
        	}
    	}
	}



	float *d_T_old = nullptr;
	float *d_T_new = nullptr;

	checkCUDA(cudaMalloc(&d_T_old, bytes_grid), FILENAME_ERR, "cudaMalloc");
	checkCUDA(cudaMalloc(&d_T_new, bytes_grid), FILENAME_ERR, "cudaMalloc");

	checkCUDA(cudaMemcpy(d_T_old, h_T, bytes_grid, cudaMemcpyHostToDevice), FILENAME_ERR, "cudaMemcpy");
	checkCUDA(cudaMemcpy(d_T_new, h_T, bytes_grid, cudaMemcpyHostToDevice), FILENAME_ERR, "cudaMemcpy");

	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 grid(NUM_BLOCKS_W, NUM_BLOCKS_H);

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	int frame = 0;
	size_t shared_bytes = (block.x + 2) * (block.y + 2) * sizeof(float);

	for (size_t t = 0; t < num_steps; t++) {
		naive_heat_diffusion<<<grid, block>>>(d_T_old, d_T_new, W, H, alpha, dt, dx);

		//heat_diffusion_optimised<<<grid, block, shared_bytes>>>(d_T_old, d_T_new, W, H, r);

		checkCUDA(cudaGetLastError(), FILENAME_ERR, "cudaGetLastError");
		//checkCUDA(cudaDeviceSynchronize(), FILENAME_ERR, FILENAME_SUCC, "cudaDeviceSynchronize");

		checkCUDA(cudaGetLastError(), FILENAME_ERR, "kernel launch");


		std::swap(d_T_old, d_T_new);

		int target_step = frame * save_interval;
		if (frame < num_frames && t == target_step) {
			checkCUDA(cudaDeviceSynchronize(), FILENAME_ERR, "kernel execution");
			checkCUDA(cudaMemcpy(h_T, d_T_old, bytes_grid, cudaMemcpyDeviceToHost), FILENAME_ERR, "cudaMemcpy");
			std::string filename = "bins/frame_" + std::to_string(frame) + ".bin";
			saveFrameBin(h_T, filename);

			frame++;
		}

	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::printf("GPU elapsed time: %f milliseconds\n", milliseconds);


	/* std::printf("Before free:\n");
	std::printf("d_T_old = %p\n", static_cast<void*>(d_T_old));
	std::printf("d_T_new = %p\n", static_cast<void*>(d_T_new));
	std::printf("h_T     = %p\n", static_cast<void*>(h_T)); */

	checkCUDA(cudaFree(d_T_old), FILENAME_ERR, "cudaFree");
	d_T_old = nullptr;

	checkCUDA(cudaFree(d_T_new), FILENAME_ERR, "cudaFree");
	d_T_new = nullptr;

	free(h_T);
	h_T = nullptr;



    return 0;
}