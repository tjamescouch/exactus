#pragma once

#include <cstddef>
#include "metalsp/csr_matrix.hpp"

namespace metalsp {

// Reference CPU implementation: y = A * x
void spmv_cpu(const CsrMatrix& A,
              const float* x,
              float* y);

// Metal GPU implementation (single device).
// For now this is a stub; implementation will come later.
void spmv_metal(const CsrMatrix& A,
                const float* x,
                float* y);

} // namespace metalsp
