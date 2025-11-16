#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>

namespace metalsp {

// Simple CSR matrix (row-major compressed sparse row).
struct CsrMatrix {
    std::size_t numRows{};
    std::size_t numCols{};
    std::size_t nnz{}; // number of non-zeros

    std::vector<std::size_t> rowPtr;  // size = numRows + 1
    std::vector<std::size_t> colInd;  // size = nnz
    std::vector<float>       values;  // size = nnz

    CsrMatrix() = default;

    CsrMatrix(std::size_t rows,
              std::size_t cols,
              std::vector<std::size_t> rowPtr_,
              std::vector<std::size_t> colInd_,
              std::vector<float> values_);

    void validate() const;
};

} // namespace metalsp
