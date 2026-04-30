//
// Created by dmytro-hilei on 4/30/26.
//

#ifndef HEATDIFFUSION_HEADER_H
#define HEATDIFFUSION_HEADER_H

#define FILENAME_ERR "cuda_error.log"
#define FILENAME_SUCC "cuda_success.log"

__global__ void naive_heat_diffusion(
    const float *T_old,
    float *T_new,
    int W,
    int H,
    float alpha, float dt, float dx);

__global__ void heat_diffusion_optimised(
    const float *T_old,
    float *T_new,
    int W,
    int H,
    float r
    );

#endif //HEATDIFFUSION_HEADER_H