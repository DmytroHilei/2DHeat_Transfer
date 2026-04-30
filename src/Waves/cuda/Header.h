//
// Created by dmytro-hilei on 4/30/26.
//

#ifndef WAVE_PROPAGATION_HEADER_H
#define WAVE_PROPAGATION_HEADER_H

#define FILENAME_ERR "wave_error.log"
#define FILENAME_SUCC "wave_success.log"

__global__ void Wave_Kernel(
    float *p_old,
    float *p_new,
    float *p,
    int H,
    int W,
    float coefficient
   );


#endif //WAVE_PROPAGATION_HEADER_H