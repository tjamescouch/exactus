#include <metal_stdlib>
using namespace metal;

struct Meta {
    uint D;
    uint M;
    uint m;
    float learning_rate;
};

constant uint PASCAL_COLS = 5; // MAX_N + 1

// --- HELPER: PASCAL LOOKUP ---
inline uint nCr(uint n, uint k, constant uint* pascal_table) {
    return pascal_table[n * PASCAL_COLS + k];
}

// --- HELPER: BINARY SEARCH SOLVER (O(log D)) ---
void get_monomial_indices(uint linear_idx, uint degree, uint max_d,
                          constant uint* pascal_table, thread uint* out_indices) {
    uint remainder = linear_idx;
    
    for (int k = degree; k > 0; k--) {
        int low = k; 
        int high = max_d + k; 
        if (high >= 512) high = 511; 
        
        int c = low;
        
        // Binary Search for largest c where nCr(c, k) <= remainder
        while (low <= high) {
            int mid = low + (high - low) / 2;
            uint val = nCr(mid, k, pascal_table);
            if (val <= remainder) {
                c = mid;      
                low = mid + 1; 
            } else {
                high = mid - 1;
            }
        }
        
        out_indices[k-1] = c - (k - 1); 
        remainder -= nCr(c, k, pascal_table);
    }
}

// --- HELPER: COMPUTE FEATURE ---
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

// --- KERNEL: STOCHASTIC UPDATE ---
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
    
    threadgroup float tg_sum_buffer[64]; 
    threadgroup float tg_loss_grad;

    // 1. Stochastic Forward Pass (Estimate y_hat)
    float local_y_part = 0.0f;

    // Grid-stride loop
    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, input_x);
        local_y_part += coefficients[k] * phi;
    }

    if (tid_tg < 64) tg_sum_buffer[tid_tg] = local_y_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduction (Leader Thread)
    if (tid_tg == 0) {
        float batch_sum = 0.0f;
        uint active_threads = min(threadsPerGroup, 64u); 
        for (uint t = 0; t < active_threads; ++t) {
            batch_sum += tg_sum_buffer[t];
        }

        // Scale Estimate: (M / m) * sum
        float scale_factor = float(meta->M) / float(meta->m);
        float y_hat_est = batch_sum * scale_factor;

        float error = y_hat_est - target_y[0];
        tg_loss_grad = 2.0f * error;

        if (tgid == 0) {
            debug_output[0] = y_hat_est;
            debug_output[1] = tg_loss_grad;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Sparse Backward Pass
    float learning_rate = meta->learning_rate;
    float loss_grad = tg_loss_grad;

    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, input_x);
        coefficients[k] -= learning_rate * loss_grad * phi;
    }
}