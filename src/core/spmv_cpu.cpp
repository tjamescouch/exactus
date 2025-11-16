#include "metalsp/spmv.hpp"

#include <algorithm>
#include <stdexcept>

namespace metalsp {

void spmv_cpu(const CsrMatrix& A,
              const float* x,
              float* y)
{
    if (!x || !y) {
        throw std::runtime_error("spmv_cpu: null x or y pointer");
    }

    // y = A * x
    for (std::size_t i = 0; i < A.numRows; ++i) {
        float sum = 0.0f;
        const std::size_t rowStart = A.rowPtr[i];
        const std::size_t rowEnd   = A.rowPtr[i + 1];

        for (std::size_t idx = rowStart; idx < rowEnd; ++idx) {
            const std::size_t j = A.colInd[idx];
            sum += A.values[idx] * x[j];
        }

        y[i] = sum;
    }
}

} // namespace metalsp
