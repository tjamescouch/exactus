// include/metalsp/spmv.hpp
#pragma once

#include <cstddef> // for std::size_t

namespace metalsp {

// New declaration for the GPU pipeline test
void mc_network_process(const float* in_data,
                        float* out_data,
                        std::size_t count);

// You may keep CsrMatrix and the old spmv declarations for later if you want to reuse them,
// but they are unnecessary for the current pipeline test.
// If removed, ensure they are also removed from the CMakeLists.txt build list.

} // namespace metalsp