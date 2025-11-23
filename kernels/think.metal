// kernels/think.metal

#include <metal_stdlib>
using namespace metal;

struct Meta {
    uint D;
    uint M;
    uint m;
    float learning_rate;
};

inline float get_feature(uint k, const device float *x) {
    switch (k) {
        case 0: return 1.0f;
        case 1: return x[0];
        case 2: return x[1];
        case 3: return x[0]*x[0];
        case 4: return x[1]*x[1];
        case 5: return x[0]*x[1];
        default: return 0.0f;
    }
}

inline float forward_pass(const device float *coefficients,
                          const device float *x,
                          uint M) {
    float y_hat = 0.0f;
    for (uint k = 0; k < M; ++k) {
        y_hat += coefficients[k] * get_feature(k, x);
    }
    return y_hat;
}

kernel void think_kernel(
    device float *coefficients              [[buffer(0)]],
    const device float *input_x             [[buffer(1)]],
    const device float *target_y            [[buffer(2)]],
    const device uint  *update_indices      [[buffer(3)]],
    constant Meta      *meta                [[buffer(4)]],
    device float       *debug_output        [[buffer(5)]],
    uint tid_tg                               [[thread_position_in_threadgroup]],
    uint tgid                                  [[threadgroup_position_in_grid]]
) {
    // Share forward-pass results within the threadgroup.
    threadgroup float tg_y_hat;
    threadgroup float tg_loss_grad;

    // --- 1) Forward pass + loss gradient by one lane per threadgroup ---
    if (tid_tg == 0) {
        float y_hat = forward_pass(coefficients, input_x, meta->M);
        float error = y_hat - target_y[0];
        float loss_gradient = 2.0f * error;

        tg_y_hat     = y_hat;
        tg_loss_grad = loss_gradient;

        // Optional: write to device buffer for host inspection
        if (tgid == 0) {            // only first TG writes, to avoid races if >1 TG
            debug_output[0] = y_hat;
            debug_output[1] = loss_gradient;
        }
    }

    // Make the threadgroup variables visible to all lanes in this TG
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- 2) Sparse updates: first m threads in the TG update coefficients ---
    if (tid_tg < meta->m) {
        float loss_gradient = tg_loss_grad;

        uint k_update = update_indices[tid_tg];
        float phi_k   = get_feature(k_update, input_x);
        float grad_ak = loss_gradient * phi_k;

        coefficients[k_update] -= meta->learning_rate * grad_ak;
    }
}
