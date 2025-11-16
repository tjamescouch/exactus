#!/usr/bin/env python3
import os
import sys
from textwrap import dedent

PROJECT_FILES = {
    "CMakeLists.txt": dedent(
        r"""
        cmake_minimum_required(VERSION 3.20)
        project(metalsp LANGUAGES CXX OBJCXX)

        set(CMAKE_CXX_STANDARD 17)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)

        # Library
        add_library(metalsp
            src/core/csr_matrix.cpp
            src/core/spmv_cpu.cpp
            src/gpu/metal/spmv_metal.mm
        )

        target_include_directories(metalsp PUBLIC include)

        # Link Metal frameworks on Apple platforms.
        if(APPLE)
            find_library(METAL_FRAMEWORK Metal)
            find_library(FOUNDATION_FRAMEWORK Foundation)
            if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
                target_link_libraries(metalsp
                    PUBLIC
                        ${METAL_FRAMEWORK}
                        ${FOUNDATION_FRAMEWORK}
                )
            endif()
        endif()

        # Demo executable
        add_executable(metalsp_demo
            examples/spmv_demo/main.cpp
        )

        target_link_libraries(metalsp_demo PRIVATE metalsp)
        """
    ).lstrip(),
    ".gitignore": dedent(
        r"""
        build/
        out/
        .vscode/
        .DS_Store
        """
    ).lstrip(),
    "include/metalsp/csr_matrix.hpp": dedent(
        r"""
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
        """
    ).lstrip(),
    "include/metalsp/spmv.hpp": dedent(
        r"""
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
        """
    ).lstrip(),
    "src/core/csr_matrix.cpp": dedent(
        r"""
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
        """
    ).lstrip(),
    "src/core/spmv_cpu.cpp": dedent(
        r"""
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
        """
    ).lstrip(),
    "src/gpu/metal/spmv_metal.mm": dedent(
        r"""
        #include "metalsp/spmv.hpp"

        #include <stdexcept>

        namespace metalsp {

        // Stub implementation for now.
        // We will later:
        //  - create Metal buffers for rowPtr, colInd, values, x, y
        //  - encode a compute pipeline using kernels/spmv.metal
        //  - dispatch one thread per row (or a more optimized scheme)
        void spmv_metal(const CsrMatrix&,
                        const float*,
                        float*)
        {
            throw std::runtime_error("spmv_metal: not implemented yet");
        }

        } // namespace metalsp
        """
    ).lstrip(),
    "kernels/spmv.metal": dedent(
        r"""
        #include <metal_stdlib>
        using namespace metal;

        // Placeholder kernel. We will replace this with a proper CSR SpMV kernel.
        kernel void spmv_csr_placeholder(device const float*  x      [[buffer(0)]],
                                         device       float*  y      [[buffer(1)]],
                                         uint                   gid  [[thread_position_in_grid]])
        {
            // For now, just copy x to y (if in range) so we can smoke-test the pipeline.
            // This is NOT a real SpMV.
            y[gid] = x[gid];
        }
        """
    ).lstrip(),
    "examples/spmv_demo/main.cpp": dedent(
        r"""
        #include "metalsp/csr_matrix.hpp"
        #include "metalsp/spmv.hpp"

        #include <iostream>
        #include <vector>

        using metalsp::CsrMatrix;

        int main() {
            // Simple 3x3 matrix:
            // [ 10  0  0 ]
            // [  0 20 30 ]
            // [ 40  0 50 ]
            std::vector<std::size_t> rowPtr = {0, 1, 3, 5};
            std::vector<std::size_t> colInd = {0, 1, 2, 0, 2};
            std::vector<float>       vals   = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

            CsrMatrix A(3, 3, rowPtr, colInd, vals);

            float x[3] = {1.0f, 2.0f, 3.0f};
            float y[3] = {0.0f, 0.0f, 0.0f};

            metalsp::spmv_cpu(A, x, y);

            std::cout << "y = [ ";
            for (float v : y) {
                std::cout << v << " ";
            }
            std::cout << "]\n";

            return 0;
        }
        """
    ).lstrip(),
}


def ensure_dirs():
    dirs = set(os.path.dirname(path) for path in PROJECT_FILES.keys() if os.path.dirname(path))
    for d in sorted(dirs):
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def check_existing():
    existing = [p for p in PROJECT_FILES.keys() if os.path.exists(p)]
    if existing:
        print("Refusing to overwrite existing files:")
        for p in existing:
            print(f"  {p}")
        print("Move or delete these files and run again.")
        sys.exit(1)


def write_files():
    for path, content in PROJECT_FILES.items():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created {path}")


def main():
    check_existing()
    ensure_dirs()
    write_files()
    print("\nBootstrap complete.")
    print("Next steps:")
    print("  mkdir -p build && cd build")
    print("  cmake .. && cmake --build .")
    print("  ./metalsp_demo (or ./Debug/metalsp_demo, depending on generator)")

if __name__ == "__main__":
    main()

