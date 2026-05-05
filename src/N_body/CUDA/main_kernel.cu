//
// Created by dmytro-hilei on 5/5/26.
//
#include "Header.h"

__global__ void N_Body_problem(
    Particle *particles_old,
    Particle *particles_new,
    int N_particles,
    float dt,
    float G,
    float eps2
    ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_particles) {
        return;
    }

    float pos_old_x = particles_old[i].x;
    float pos_old_y = particles_old[i].y;
    float pos_old_z = particles_old[i].z;

    float vel_old_x = particles_old[i].vx;
    float vel_old_y = particles_old[i].vy;
    float vel_old_z = particles_old[i].vz;

    float3 acc = make_float3(0.0f, 0.0f, 0.0f);

    for (int j = 0; j < N_particles; j++) {
        if (i == j) {
            continue;
        }
        float3 r = make_float3(particles_old[j].x - pos_old_x, particles_old[j].y - pos_old_y, particles_old[j].z - pos_old_z);
        float dist2 = r.x * r.x + r.y * r.y + r.z * r.z + eps2;

        float invdist = rsqrtf(dist2);
        float invdist3 = invdist * invdist * invdist;

        float s = G * particles_old[j].mass * invdist3;
        acc.x += s * r.x;
        acc.y += s * r.y;
        acc.z += s * r.z;


    }

    vel_old_x += acc.x * dt;
    vel_old_y += acc.y * dt;
    vel_old_z += acc.z * dt;

    pos_old_x += vel_old_x * dt;
    pos_old_y += vel_old_y * dt;
    pos_old_z += vel_old_z * dt;

    particles_new[i].x = pos_old_x;
    particles_new[i].y = pos_old_y;
    particles_new[i].z = pos_old_z;
    particles_new[i].vx = vel_old_x;
    particles_new[i].vy = vel_old_y;
    particles_new[i].vz = vel_old_z;

    particles_new[i].mass = particles_old[i].mass;
}