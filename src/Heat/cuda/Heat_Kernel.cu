//
// Created by dmytro-hilei on 4/30/26.
//



__global__ void naive_heat_diffusion(
    const float *T_old,
    float *T_new,
    int W,
    int H,
    float alpha, float dt, float dx) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; //Unsigned is a bit safer

    if (x >= W || y >= H) {
        return;
    }

    int idx = y * W + x;

    if (x == 0 || y == 0 || x == W - 1 || y == H - 1) {
        T_new[idx] = T_old[idx];
        return;
    }

    float center = T_old[idx];
    float left = T_old[idx - 1];
    float right = T_old[idx + 1];
    float top = T_old[idx - W];
    float bottom = T_old[idx + W];

    T_new[idx] = center + alpha * dt/(dx*dx) * (left + right + top + bottom - 4 * center);
}


__global__ void heat_diffusion_optimised(
    const float *T_old,
    float *T_new,
    int W,
    int H,
    float r
    ) {
    extern __shared__ float smem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    int tileW = blockDim.x + 2;

    int local_x = threadIdx.x + 1;
    int local_y = threadIdx.y + 1;

    bool inside = (x < W && y < H);
    bool interior = (x > 0 && x < W - 1 && y > 0 && y < H - 1);

    int local_idx = local_y * tileW + local_x;

    if (inside) {
        smem[local_idx] = T_old[y * W + x];
    }
    if (interior) {
        if (threadIdx.x == 0) {
            smem[local_y * tileW + local_x - 1] = T_old[y * W + x - 1];
        }

        if (threadIdx.x == blockDim.x - 1) {
            smem[local_y * tileW + local_x + 1] = T_old[y * W + x + 1];
        }

        if (threadIdx.y == 0) {
            smem[(local_y - 1) * tileW + local_x] = T_old[(y - 1) * W + x];
        }

        if (threadIdx.y == blockDim.y - 1) {
            smem[(local_y + 1) * tileW + local_x] = T_old[(y + 1) * W + x];
        }
    }

    __syncthreads();

    if (!inside) {
        return;
    }

    if (!interior) {
        T_new[y * W + x] = T_old[y * W + x];
        return;
    }

    float center = smem[local_idx];
    float left = smem[local_y * tileW + local_x - 1];
    float right = smem[local_y * tileW + local_x + 1];
    float top = smem[(local_y - 1) * tileW + local_x];
    float bottom = smem[(local_y + 1) * tileW + local_x];

    T_new[y * W + x] = center + r * (left + right + top + bottom - 4.0f * center);
}