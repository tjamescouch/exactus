#include "metalsp/network_types.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>

// --- FORWARD DECLARATION ---
namespace metalsp {
    void mc_network_process_step(
        const std::vector<float>& x,
        float y,
        std::vector<float>& coefficients,
        float learning_rate,
        std::vector<uint32_t>& update_indices,
        const std::vector<uint32_t>& pascal_table,
        bool sync_weights 
    );
    extern std::vector<float> debug_output_host;
}

std::vector<uint32_t> load_pascal_table(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("CRITICAL ERROR: Could not find '" + filename + "'");
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> buffer(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

long long nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0 || r == n) return 1;
    if (r > n / 2) r = n - r;
    long long res = 1;
    for (int i = 1; i <= r; ++i) {
        long long next_val;
        if (__builtin_mul_overflow(res, (n - i + 1), &next_val)) throw std::overflow_error("Overflow");
        res = next_val / i;
    }
    return res;
}

std::mt19937 rng(42);
std::vector<uint32_t> sample_indices(uint32_t M, uint32_t m) {
    if (m >= M) {
        std::vector<uint32_t> indices(M);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }
    std::vector<uint32_t> sample;
    sample.reserve(m);
    std::uniform_int_distribution<uint32_t> dist(0, M - 1);
    for(size_t i=0; i<m; ++i) sample.push_back(dist(rng));
    return sample;
}

int main() {
    try {
        const uint32_t D = 500; 
        const uint32_t N = 4;   
        
        long long M_long = nCr(D + N - 1, N);
        if (M_long > 4294967295) throw std::overflow_error("M exceeds uint32 limit!");
        uint32_t M = static_cast<uint32_t>(M_long);

        const uint32_t m_samples = 1000000; 
        const float INITIAL_LR = 5.0e-7f; 
        const int NUM_STEPS = 100;

        std::cout << "--- Bitwise Monte Carlo (Final Exam) ---\n";
        std::cout << "Dims: D=" << D << ", N=" << N << "\n";
        std::cout << "Features: " << M << " (2.6 Billion)\n";

        std::vector<uint32_t> pascal_table = load_pascal_table("pascal.bin");
        std::vector<float> input_x(D, 1.0f); 
        input_x[0] = 2.0f; input_x[1] = 3.0f; input_x[2] = 1.0f; 
        float target_y = 21.6f; 
        
        std::cout << "Allocating Weights (10GB)... ";
        std::cout.flush();
        std::vector<float> coefficients(M, 0.0f);
        std::cout << "Done.\n\n";

        float current_lr = INITIAL_LR;
        double total_gpu_time = 0.0;

        // --- TRAINING LOOP ---
        for (int step = 1; step <= NUM_STEPS; ++step) {
            // Adjusted Annealing for 100 steps
            if (step == 30) { // Give it 30 steps to climb
                current_lr = INITIAL_LR * 0.1f;
                std::cout << " [System] Learning Rate Decayed to " << std::scientific << current_lr << std::defaultfloat << "\n";
            }
            if (step == 70) { // Fine tuning late in the game
                current_lr = INITIAL_LR * 0.01f;
                std::cout << " [System] Learning Rate Decayed to " << std::scientific << current_lr << std::defaultfloat << "\n";
            }

            std::vector<uint32_t> update_indices = sample_indices(M, m_samples);
            
            auto start = std::chrono::high_resolution_clock::now();

            metalsp::mc_network_process_step(
                input_x, target_y, coefficients, 
                current_lr, update_indices, pascal_table,
                false 
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = end - start;
            
            if (step > 1) total_gpu_time += ms_double.count();

            if (step % 5 == 0 || step == 1) {
                float y_hat = metalsp::debug_output_host[0];
                std::cout << "Step " << std::setw(2) << step 
                          << " | Time: " << std::fixed << std::setprecision(2) << ms_double.count() << " ms"
                          << " | y_hat (Est): " << std::setprecision(4) << y_hat 
                          << " | Error: " << (y_hat - target_y) << "\n";
            }
        }

        // --- FINAL EXAM: HIGH-FIDELITY VALIDATION ---
        std::cout << "\n[System] Running High-Fidelity Validation (10M Samples)...\n";
        
        // 1. Increase sample size by 100x for the test
        uint32_t validation_m = 10000000; 
        std::vector<uint32_t> val_indices = sample_indices(M, validation_m);
        
        // 2. Run with LR = 0.0 (No updates, just inference)
        auto v_start = std::chrono::high_resolution_clock::now();
        metalsp::mc_network_process_step(
            input_x, target_y, coefficients, 
            0.0f, // Zero Learning Rate
            val_indices, pascal_table,
            false 
        );
        auto v_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> v_ms = v_end - v_start;

        float final_y_hat = metalsp::debug_output_host[0];

        std::cout << "Validation Time: " << v_ms.count() << " ms\n";
        std::cout << "------------------------------------------------\n";
        std::cout << "TARGET      : " << target_y << "\n";
        std::cout << "PREDICTION  : " << final_y_hat << "\n";
        std::cout << "FINAL ERROR : " << (final_y_hat - target_y) << "\n";
        std::cout << "------------------------------------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << "\n";
        return 1;
    }
    return 0;
}