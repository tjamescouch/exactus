#include "metalsp/spmv.hpp" // Contains the declaration for mc_network_process

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Forward declaration of the GPU function (declared in metalsp/spmv.hpp)
namespace metalsp {
    void mc_network_process(const float* in_data,
                            float* out_data,
                            std::size_t count);
}

int main() {
    const std::size_t N = 10; // Use a small, fixed size for simple testing

    std::cout << "--- mc-network GPU Pipeline Test (Add 1) ---\n";
    std::cout << "Testing buffer size: " << N << " elements\n";

    // 1. Setup Input and Expected Output
    std::vector<float> input_data(N);
    std::vector<float> output_data(N, 0.0f);
    std::vector<float> expected_data(N);

    // Initialize input_data: [1.0, 2.0, 3.0, ..., 10.0]
    for (std::size_t i = 0; i < N; ++i) {
        input_data[i] = static_cast<float>(i + 1);
        // Kernel output should be: input + 1.0f
        expected_data[i] = input_data[i] + 1.0f;
    }

    // 2. Run GPU Function
    try {
        std::cout << "Input data: [";
        for (std::size_t i = 0; i < N; ++i) std::cout << input_data[i] << (i < N - 1 ? ", " : "");
        std::cout << "]\n";

        metalsp::mc_network_process(input_data.data(), output_data.data(), N);

        std::cout << "GPU output: [";
        for (std::size_t i = 0; i < N; ++i) std::cout << output_data[i] << (i < N - 1 ? ", " : "");
        std::cout << "]\n";

    } catch (const std::exception& e) {
        std::cerr << "GPU process failed: " << e.what() << "\n";
        return 1;
    }


    // 3. Verification
    float max_diff = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        max_diff = std::max(max_diff, std::abs(output_data[i] - expected_data[i]));
    }

    if (max_diff < 1e-6) {
        std::cout << "\n✅ Pipeline test PASSED! Max difference: " << max_diff << "\n";
        return 0;
    } else {
        std::cerr << "\n❌ Pipeline test FAILED! Max difference: " << max_diff << "\n";
        return 1;
    }
}