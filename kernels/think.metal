// kernels/think.metal (STOCHASTIC OPTIMIZATION)

#include <metal_stdlib>
using namespace metal;

struct Meta {
    uint D;
    uint M;
    uint m;
    float learning_rate;
};

constant uint PASCAL_COLS = 5; // MAX_N + 1

// --- BITWISE GENERATOR (Same as before) ---
inline uint nCr(uint n, uint k, constant uint* pascal_table) {
    return pascal_table[n * PASCAL_COLS + k];
}

// kernels/think.metal (Binary Search Optimization)

// ... (Includes and nCr helper remain the same) ...

// --- BINARY SEARCH SOLVER ---
// Replaces Linear Scan. Solves for c in nCr(c, k) <= remainder
// Complexity: O(log D) instead of O(D)
void get_monomial_indices(uint linear_idx, uint degree, uint max_d,
                          constant uint* pascal_table, thread uint* out_indices) {
    uint remainder = linear_idx;
    
    // Upper bound can be tighter, but max_d + degree is safe
    // Lower bound is always k (since n must be >= k)
    
    for (int k = degree; k > 0; k--) {
        // We want to find the LARGEST 'c' such that nCr(c, k) <= remainder.
        
        int low = k; 
        int high = max_d + k; 
        if (high >= 512) high = 511; // Clamp to table limits
        
        int c = low;
        
        // Binary Search
        while (low <= high) {
            int mid = low + (high - low) / 2;
            uint val = nCr(mid, k, pascal_table);
            
            if (val <= remainder) {
                c = mid;      // This 'mid' is a candidate (it fits)
                low = mid + 1; // Try to find a larger one
            } else {
                high = mid - 1; // 'mid' is too big
            }
        }
        
        // 'c' is now the largest index satisfying the condition
        out_indices[k-1] = c - (k - 1); 
        remainder -= nCr(c, k, pascal_table);
    }
}

// ... (Rest of the file: compute_feature and think_kernel remain exactly the same) ...

float compute_feature(uint k_idx, uint degree, uint D, 
                      constant uint* pascal_table, device const float* x) {
    uint indices[8]; 
    get_monomial_indices(k_idx, degree, D, pascal_table, indices);
    float val = 1.0f;
    for(int i = 0; i < degree; ++i) {
        val *= x[indices[i]];
    }
    return val;
}

// --- STOCHASTIC KERNEL ---
kernel void think_kernel(
    device float *coefficients              [[buffer(0)]],
    const device float *input_x             [[buffer(1)]],
    const device float *target_y            [[buffer(2)]],
    const device uint  *update_indices      [[buffer(3)]],
    constant Meta      *meta                [[buffer(4)]],
    device float       *debug_output        [[buffer(5)]],
    constant uint      *pascal_table        [[buffer(6)]],
    uint tid_tg                             [[thread_position_in_threadgroup]],
    uint tgid                               [[threadgroup_position_in_grid]],
    uint threadsPerGroup                    [[threads_per_threadgroup]]
) {
    uint N = 4; 
    
    // Shared memory for reduction
    threadgroup float tg_sum_buffer[64]; // Assumes max 64 threads/group
    threadgroup float tg_loss_grad;

    // --- PHASE 1: STOCHASTIC FORWARD PASS ---
    // All threads compute partial sums of the SAMPLE batch (not the full M)
    float local_y_part = 0.0f;

    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, input_x);
        local_y_part += coefficients[k] * phi;
    }

    // Store in shared memory
    if (tid_tg < 64) tg_sum_buffer[tid_tg] = local_y_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel Reduction (Simple linear scan by leader for 64 items is fast enough)
    if (tid_tg == 0) {
        float batch_sum = 0.0f;
        // Sum up results from all threads
        // (Note: In production, a tree-reduction is faster, but for 64 threads linear is fine)
        uint active_threads = min(threadsPerGroup, 64u); 
        for (uint t = 0; t < active_threads; ++t) {
            batch_sum += tg_sum_buffer[t];
        }

        // SCALE the sum: (M / m) * batch_sum
        // This is the Monte Carlo Estimator
        float scale_factor = float(meta->M) / float(meta->m);
        float y_hat_est = batch_sum * scale_factor;

        float error = y_hat_est - target_y[0];
        tg_loss_grad = 2.0f * error;

        // Write debug info
        if (tgid == 0) {
            debug_output[0] = y_hat_est;
            debug_output[1] = tg_loss_grad;
        }
    }

    // Wait for Leader to compute gradient
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- PHASE 2: SPARSE UPDATES ---
    // Re-run the loop to update weights
    // (We re-compute phi to save shared memory space - ALU is cheap!)
    float learning_rate = meta->learning_rate;
    float loss_grad = tg_loss_grad; // Read from shared

    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, input_x);
        
        // Update
        coefficients[k] -= learning_rate * loss_grad * phi;
    }
}