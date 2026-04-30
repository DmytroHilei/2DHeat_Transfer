//
// Created by giley on 1/25/2026.
//

#ifndef HEADERS_H
#define HEADERS_H


using real = float;
namespace constants {
    constexpr int size {1000};
    constexpr real sound_velocity {340};
    constexpr real space_step {0.01f};
    constexpr real time_of_simulation {0.1f};
    constexpr real time_step {0.00001f};
    constexpr real coeficient {(sound_velocity * time_step / space_step)*(sound_velocity * time_step/space_step)};
    constexpr int square_size {16};
    constexpr real normal_pressure {100000.00f};
    constexpr int32_t steps = static_cast<int32_t>(time_of_simulation/time_step); //Here I set namespace of constants for the simulation
}

real *data();
void saveToBin(real *p, const std::string &filename);

#endif //HEADERS_H
