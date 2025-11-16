#include <metal_stdlib>
using namespace metal;

// Simple metadata passed from the host.
// Extend later with numCols, etc., if we need it.
struct CsrMeta
{
    uint numRows;
};

// CSR SpMV kernel:
//   y = A * x
// where A is in CSR format:
//
//   rowPtr : size numRows + 1
//   colInd : size nnz
//   values : size nnz
//
// One thread handles one row.
kernel void spmv_csr(device const uint   *rowPtr  [[buffer(0)]],
                     device const uint   *colInd  [[buffer(1)]],
                     device const float  *values  [[buffer(2)]],
                     device const float  *x       [[buffer(3)]],
                     device       float  *y       [[buffer(4)]],
                     constant     CsrMeta& meta   [[buffer(5)]],
                     uint                  row    [[thread_position_in_grid]])
{
    if (row >= meta.numRows) {
        return;
    }

    const uint rowStart = rowPtr[row];
    const uint rowEnd   = rowPtr[row + 1];

    float sum = 0.0f;
    for (uint idx = rowStart; idx < rowEnd; ++idx) {
        const uint j = colInd[idx];
        sum += values[idx] * x[j];
    }

    y[row] = sum;
}
