#include "metalsp/csr_matrix.hpp"

namespace metalsp {

CsrMatrix::CsrMatrix(std::size_t rows,
                     std::size_t cols,
                     std::vector<std::size_t> rowPtr_,
                     std::vector<std::size_t> colInd_,
                     std::vector<float> values_)
    : numRows(rows),
      numCols(cols),
      nnz(values_.size()),
      rowPtr(std::move(rowPtr_)),
      colInd(std::move(colInd_)),
      values(std::move(values_))
{
    validate();
}

void CsrMatrix::validate() const {
    if (rowPtr.size() != numRows + 1) {
        throw std::runtime_error("CsrMatrix: rowPtr.size() must be numRows + 1");
    }
    if (colInd.size() != nnz || values.size() != nnz) {
        throw std::runtime_error("CsrMatrix: colInd/values size must equal nnz");
    }
    if (rowPtr.front() != 0) {
        throw std::runtime_error("CsrMatrix: rowPtr[0] must be 0");
    }
    if (rowPtr.back() != nnz) {
        throw std::runtime_error("CsrMatrix: rowPtr[numRows] must equal nnz");
    }
    for (std::size_t i = 0; i + 1 < rowPtr.size(); ++i) {
        if (rowPtr[i] > rowPtr[i + 1]) {
            throw std::runtime_error("CsrMatrix: rowPtr must be non-decreasing");
        }
    }
    for (std::size_t k = 0; k < nnz; ++k) {
        if (colInd[k] >= numCols) {
            throw std::runtime_error("CsrMatrix: colInd out of range");
        }
    }
}

} // namespace metalsp
