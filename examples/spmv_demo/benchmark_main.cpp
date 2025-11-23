#include "metalsp/network_types.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip> // Added for cleaner output formatting

// --- FORWARD DECLARATION ---
// Matches the Persistent GPU signature
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

// ---------------------------------------------------------
// HELPER 1: Load Pascal Triangle
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// HELPER 2: Safe nCr
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// HELPER 3: Random Sampler
// ---------------------------------------------------------
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
    
    // Fast sampling (collisions allowed for benchmark speed)
    for(size_t i=0; i<m; ++i) sample.push_back(dist(rng));
    return sample;
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    try {
        // --- 1. CONFIGURATION ---
        const uint32_t D = 500; 
        const uint32_t N = 4;   
        
        long long M_long = nCr(D + N - 1, N);
        if (M_long > 4294967295) throw std::overflow_error("M exceeds uint32 limit!");
        uint32_t M = static_cast<uint32_t>(M_long);

        // --- TUNING ---
        const uint32_t m_samples = 100000; 
        
        // START FAST: High LR to get near the target quickly
        const float INITIAL_LR = 5.0e-7f; 
        
        const int NUM_STEPS = 50;

        std::cout << "--- Bitwise Monte Carlo (Annealing Mode) ---\n";
        std::cout << "Dims: D=" << D << ", N=" << N << "\n";
        std::cout << "Features (M): " << M << " (2.6 Billion)\n";
        std::cout << "Initial LR: " << INITIAL_LR << "\n";

        // --- 2. RESOURCES ---
        std::vector<uint32_t> pascal_table = load_pascal_table("pascal.bin");

        // --- 3. INITIALIZATION ---
        std::vector<float> input_x(D, 1.0f); 
        input_x[0] = 2.0f; input_x[1] = 3.0f; input_x[2] = 1.0f; 

        float target_y = 21.6f; 
        
        // Allocate large host buffer (10GB)
        std::cout << "Allocating Weights... ";
        std::cout.flush();
        std::vector<float> coefficients(M, 0.0f);
        std::cout << "Done.\n";

        std::cout << "Starting Training Loop...\n\n";

        // --- 4. TRAINING LOOP ---
        double total_gpu_time = 0.0;
        float current_lr = INITIAL_LR;

        for (int step = 1; step <= NUM_STEPS; ++step) {
            
            // --- ANNEALING SCHEDULE ---
            if (step == 15) {
                current_lr = INITIAL_LR * 0.1f;
                std::cout << " [System] Learning Rate Decayed to " << current_lr << "\n";
            }
            if (step == 35) {
                current_lr = INITIAL_LR * 0.01f;
                std::cout << " [System] Learning Rate Decayed to " << current_lr << "\n";
            }

            std::vector<uint32_t> update_indices = sample_indices(M, m_samples);
            
            auto start = std::chrono::high_resolution_clock::now();

            // call with sync_weights = false (Persistent Mode)
            metalsp::mc_network_process_step(
                input_x, target_y, coefficients, 
                current_lr, update_indices, pascal_table,
                false 
            );

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> ms_double = end - start;
            
            // Exclude first step (upload overhead) from timing average
            if (step > 1) total_gpu_time += ms_double.count();

            if (step % 5 == 0 || step == 1) {
                float y_hat = metalsp::debug_output_host[0];
                std::cout << "Step " << std::setw(2) << step 
                          << " | Time: " << std::fixed << std::setprecision(2) << ms_double.count() << " ms"
                          << " | y_hat: " << std::setprecision(4) << y_hat 
                          << " | Error: " << (y_hat - target_y) << "\n";
            }
        }

        std::cout << "\n--- Results ---\n";
        double avg_time = total_gpu_time / (NUM_STEPS - 1);
        std::cout << "Avg Time (steady state): " << avg_time << " ms\n";
        std::cout << "Final Prediction: " << metalsp::debug_output_host[0] << "\n";
        
        double ops = (double)m_samples * 2.0; 
        double throughput = (ops / (avg_time / 1000.0)) / 1e6; 
        std::cout << "Throughput: " << throughput << " MOps/sec\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << "\n";
        return 1;
    }
    return 0;
}