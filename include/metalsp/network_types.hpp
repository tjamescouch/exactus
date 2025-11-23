#pragma once

#include <cstdint>
#include <vector>

namespace metalsp {

// Must match the struct used in the Metal kernel
struct Meta {
    std::uint32_t D;             // Input dimension
    std::uint32_t M;             // Total coefficients
    std::uint32_t m;             // Monte Carlo sample size
    float learning_rate;         // eta
};

// Expose the debug output vector globally so main.cpp can access it
extern std::vector<float> debug_output_host;

} // namespace metalsp