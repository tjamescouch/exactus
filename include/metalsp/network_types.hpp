// include/metalsp/network_types.hpp

#pragma once

#include <cstdint>

namespace metalsp {

// Must match the struct used in the Metal kernel
struct Meta {
    std::uint32_t D;             // Input dimension (2)
    std::uint32_t M;             // Total coefficients (6)
    std::uint32_t m;             // Monte Carlo sample size (e.g., 2)
    float learning_rate;         // eta
};

} // namespace metalsp