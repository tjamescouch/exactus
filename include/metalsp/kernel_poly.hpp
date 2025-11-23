#pragma once
#include <vector>
#include <cstdint>

namespace metalsp {

// Compute v = K u, K_ij = (1 + x_i^T x_j)^degree
// X_flat: size N*D (row-major)
// u: size N
// returns v size N
std::vector<float> kernel_poly_matvec(const std::vector<float>& X_flat,
                                      int32_t N, int32_t D,
                                      const std::vector<float>& u,
                                      int32_t degree);

} // namespace metalsp
