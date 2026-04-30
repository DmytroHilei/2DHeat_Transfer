//
// Created by dmytro-hilei on 4/30/26.
//

__global__ void Wave_Kernel(
    float *p_old,
    float *p_new,
    float *p,
    int H,
    int W,
    float coefficient
   ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W && y >= H) {
        return;
    }

    int index = y * W + x;

    if (x == 0 || y == 0 || x == W - 1 || y == H - 1) {
        p_new[index] = p[index];
        return;
    }

    float center = p[index];

    p_new[index] = 2.0f * center
               - p_old[index]
               + coefficient * (
                     p[(x-1) * W + y]
                   + p[(x+1) * H + y]
                   + p[x * W + y - 1]
                   + p[x * W + y + 1]
                   - 4.0f * center
               );
}