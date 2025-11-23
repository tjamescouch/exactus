// examples/spmv_demo/main.cpp

#include "metalsp/network_types.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>

// Function declaration from mc_network_metal.mm (now using the full signature)
namespace metalsp {
    void mc_network_process_step(
        const std::vector<float>& x,
        float y,
        std::vector<float>& coefficients,
        float learning_rate,
        std::vector<uint32_t>& update_indices
    );
}

// Global variable for reproducibility in Monte Carlo sampling
std::mt19937 rng(42);

// Function to sample 'm' unique indices from 0 to M-1
std::vector<uint32_t> sample_indices(uint32_t M, uint32_t m) {
    if (m >= M) {
        std::vector<uint32_t> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    
    std::vector<uint32_t> pool(M);
    std::iota(pool.begin(), pool.end(), 0);
    
    std::shuffle(pool.begin(), pool.end(), rng);
    
    std::vector<uint32_t> sample(pool.begin(), pool.begin() + m);
    return sample;
}

int main() {
    std::cout << "--- Monte Carlo Polynomial Network Test (D=2, N=2) ---\n";

    // M=6 coefficients correspond to: [1, x1, x2, x1^2, x2^2, x1*x2]
    const uint32_t M = 6;
    const uint32_t m = 2; // Monte Carlo sample size: update only 2 coefficients per step
    const float LEARNING_RATE = 0.01f;
    const int NUM_STEPS = 10;

    // --- Training Data (Simple target function: y = 5*x1 + 3*x2) ---
    // The network should learn coefficients 1 and 2 (corresponding to x1, x2)
    std::vector<float> input_x = {2.0f, 3.0f}; // x1=2.0, x2=3.0
    float target_y = 5.0f * 2.0f + 3.0f * 3.0f; // Target y = 10.0 + 9.0 = 19.0f

    // --- Initialization ---
    // Initialize all coefficients to a small random value (except the bias a[0]=0)
    std::vector<float> coefficients(M, 0.1f); 
    
    std::cout << "Target function: y = 5*x1 + 3*x2\n";
    std::cout << "Input x: [" << input_x[0] << ", " << input_x[1] << "], Target y: " << target_y << "\n";
    std::cout << "Initial coefficients: [";
    for(float c : coefficients) std::cout << c << ", ";
    std::cout << "]\n\n";

    // --- Training Loop ---
    for (int step = 1; step <= NUM_STEPS; ++step) {
        // 1. Monte Carlo Sampling
        std::vector<uint32_t> sampled_indices = sample_indices(M, m);
        
        std::cout << "Step " << step << " (Sampled: ";
        for(uint32_t idx : sampled_indices) std::cout << idx << " ";
        std::cout << ")\n";
        
        // 2. GPU Processing Step
        try {
            metalsp::mc_network_process_step(
                input_x, 
                target_y, 
                coefficients, 
                LEARNING_RATE, 
                sampled_indices
            );
        } catch (const std::exception& e) {
            std::cerr << "GPU process failed at step " << step << ": " << e.what() << "\n";
            return 1;
        }

        // 3. Log current coefficients
        std::cout << "  > Coeffs: [";
        for (uint32_t i = 0; i < M; ++i) {
            std::cout << i << ":" << coefficients[i] << (i < M - 1 ? ", " : "");
        }
        std::cout << "]\n";
    }

    std::cout << "\nTraining finished.\n";
    return 0;
}