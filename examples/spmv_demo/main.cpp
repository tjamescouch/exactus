#include "metalsp/csr_matrix.hpp"
#include "metalsp/spmv.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

using metalsp::CsrMatrix;

CsrMatrix make_random_csr(std::size_t rows,
                          std::size_t cols,
                          float density)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> valDist(0.0f, 1.0f);

    std::vector<std::size_t> rowPtr(rows + 1);
    std::vector<std::size_t> colInd;
    std::vector<float>       vals;

    std::bernoulli_distribution keep(density);

    std::size_t nnz = 0;
    for (std::size_t i = 0; i < rows; ++i) {
        rowPtr[i] = nnz;
        for (std::size_t j = 0; j < cols; ++j) {
            if (keep(rng)) {
                colInd.push_back(j);
                vals.push_back(valDist(rng));
                nnz++;
            }
        }
    }
    rowPtr[rows] = nnz;

    return CsrMatrix(rows, cols, rowPtr, colInd, vals);
}

int main() {
    const std::size_t N = 4096;     // square matrix
    const float density = 0.01f;    // 1% non-zero

    std::cout << "Generating random " << N << "x" << N
              << " CSR with density " << density << "...\n";

    CsrMatrix A = make_random_csr(N, N, density);

    std::cout << "nnz = " << A.nnz << "\n";

    std::vector<float> x(N, 1.0f);
    std::vector<float> y_cpu(N, 0.0f);
    std::vector<float> y_gpu(N, 0.0f);

    // --- CPU ---
    auto t0 = std::chrono::high_resolution_clock::now();
    metalsp::spmv_cpu(A, x.data(), y_cpu.data());
    auto t1 = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- GPU ---
    auto t2 = std::chrono::high_resolution_clock::now();
    metalsp::spmv_metal(A, x.data(), y_gpu.data());
    auto t3 = std::chrono::high_resolution_clock::now();

    double gpu_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- correctness ---
    float maxDiff = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        maxDiff = std::max(maxDiff, std::abs(y_cpu[i] - y_gpu[i]));
    }

    std::cout << "CPU time = " << cpu_ms << " ms\n";
    std::cout << "GPU time = " << gpu_ms << " ms\n";
    std::cout << "max diff = " << maxDiff << "\n";

    return 0;
}
