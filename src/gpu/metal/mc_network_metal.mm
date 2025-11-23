#include "metalsp/network_types.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <random>
#include <algorithm>
#include <iomanip>

namespace metalsp {

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pipeline = nil;
static id<MTLBuffer> g_coefficientsBuf = nil;

std::vector<float> debug_output_host(2, 0.0f);

// --- SHADER (Same as before) ---
const char* METAL_SOURCE = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Meta {
    uint D;
    uint M;
    uint m;
    float learning_rate;
    uint use_lut; 
};

constant uint PASCAL_COLS = 5; 

inline uint nCr(uint n, uint k, constant uint* pascal_table) {
    return pascal_table[n * PASCAL_COLS + k];
}

void get_monomial_indices_math(uint linear_idx, uint degree, uint max_d,
                          constant uint* pascal_table, thread uint* out_indices) {
    uint remainder = linear_idx;
    for (int k = degree; k > 0; k--) {
        int low = k; int high = max_d + k; if (high >= 512) high = 511; 
        int c = low;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            uint val = nCr(mid, k, pascal_table);
            if (val <= remainder) { c = mid; low = mid + 1; } else { high = mid - 1; }
        }
        out_indices[k-1] = c - (k - 1); 
        remainder -= nCr(c, k, pascal_table);
    }
}

void get_monomial_indices_lut(uint k_idx, uint degree, constant uint* lut, thread uint* out_indices) {
    uint base = k_idx * degree;
    for (int i = 0; i < degree; ++i) {
        out_indices[i] = lut[base + i];
    }
}

float compute_feature(uint k_idx, uint degree, uint D, 
                      constant uint* pascal_table, device const float* sample_x,
                      constant uint* lut, uint use_lut) {
    uint indices[8]; 
    if (use_lut == 1) {
        get_monomial_indices_lut(k_idx, degree, lut, indices);
    } else {
        get_monomial_indices_math(k_idx, degree, D, pascal_table, indices);
    }
    float val = 1.0f;
    for(int i = 0; i < degree; ++i) { val *= sample_x[indices[i]]; }
    return val;
}

kernel void think_kernel(
    device float *coefficients [[buffer(0)]],
    const device float *all_input_x [[buffer(1)]],
    const device float *all_target_y [[buffer(2)]],
    const device uint  *update_indices [[buffer(3)]],
    constant Meta      *meta [[buffer(4)]],
    device float       *debug_output [[buffer(5)]],
    constant uint      *pascal_table [[buffer(6)]],
    constant uint      *indices_lut  [[buffer(7)]],
    uint tid_tg [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]], 
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    uint N = 4; 
    threadgroup float tg_sum_buffer[64]; 
    threadgroup float tg_loss_grad;

    // 1. Identify Sample
    uint sample_idx = tgid; 
    device const float* my_x = all_input_x + (sample_idx * meta->D);
    float my_y = all_target_y[sample_idx];

    // 2. Forward
    float local_y_part = 0.0f;
    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, my_x, indices_lut, meta->use_lut);
        local_y_part += coefficients[k] * phi;
    }
    if (tid_tg < 64) tg_sum_buffer[tid_tg] = local_y_part;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_tg == 0) {
        float batch_sum = 0.0f;
        uint active_threads = min(threadsPerGroup, 64u); 
        for (uint t = 0; t < active_threads; ++t) { batch_sum += tg_sum_buffer[t]; }
        float scale_factor = float(meta->M) / float(meta->m);
        float y_hat_est = batch_sum * scale_factor;
        float error = y_hat_est - my_y;
        tg_loss_grad = 2.0f * error;
        
        if (sample_idx == 0) { debug_output[0] = y_hat_est; debug_output[1] = tg_loss_grad; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Backward
    float learning_rate = meta->learning_rate;
    float loss_grad = tg_loss_grad;
    for (uint i = tid_tg; i < meta->m; i += threadsPerGroup) {
        uint k = update_indices[i];
        float phi = compute_feature(k, N, meta->D, pascal_table, my_x, indices_lut, meta->use_lut);
        coefficients[k] -= learning_rate * loss_grad * phi;
    }
}
)METAL";

// --- STRUCTS ---
struct MetaGPU {
    uint32_t D; uint32_t M; uint32_t m; float learning_rate; uint32_t use_lut;
};

// --- HELPER FUNCTIONS ---
long long nCr_cpu(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0 || r == n) return 1;
    if (r > n / 2) r = n - r;
    long long res = 1;
    for (int i = 1; i <= r; ++i) res = res * (n - i + 1) / i;
    return res;
}

std::vector<uint32_t> generate_indices_lut(uint32_t M, uint32_t N, uint32_t D) {
    std::vector<uint32_t> lut;
    lut.reserve(M * N);
    for (uint32_t k = 0; k < M; ++k) {
        uint32_t remainder = k;
        uint32_t c = D + N; 
        for (int r = N; r > 0; r--) {
            while (true) {
                long long val = nCr_cpu(c, r);
                if (val <= remainder) { lut.push_back(c - (r - 1)); remainder -= val; c--; break; }
                c--;
            }
        }
    }
    return lut;
}

void init_metal_context() {
    if (g_device != nil) return;
    NSError *error = nil;
    g_device = MTLCreateSystemDefaultDevice();
    g_queue = [g_device newCommandQueue];
    NSString *source = [NSString stringWithUTF8String:METAL_SOURCE];
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> library = [g_device newLibraryWithSource:source options:options error:&error];
    if (!library) { NSLog(@"Error: %@", error); throw std::runtime_error("Shader Compile Failed"); }
    id<MTLFunction> func = [library newFunctionWithName:@"think_kernel"];
    g_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];
}

void mc_network_process_step(
    const std::vector<float>& x, float y, std::vector<float>& coefficients,
    float learning_rate, std::vector<uint32_t>& update_indices,
    const std::vector<uint32_t>& pascal_table, bool sync_weights 
) {}

// --- MINI-BATCH FIT ---
void mc_network_fit(
    const std::vector<float>& X_flat, 
    const std::vector<float>& y,      
    std::vector<float>& coefficients,
    float learning_rate,
    int epochs,
    const std::vector<uint32_t>& pascal_table
) {
    init_metal_context();
    
    uint32_t N_SAMPLES = static_cast<uint32_t>(y.size());
    uint32_t D = static_cast<uint32_t>(X_flat.size() / N_SAMPLES);
    uint32_t M = static_cast<uint32_t>(coefficients.size());

    // 1. Buffers
    if (g_coefficientsBuf == nil || [g_coefficientsBuf length] != coefficients.size() * sizeof(float)) {
        g_coefficientsBuf = [g_device newBufferWithBytes:coefficients.data() length:coefficients.size() * sizeof(float) options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> allXBuf = [g_device newBufferWithBytes:X_flat.data() length:X_flat.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> allYBuf = [g_device newBufferWithBytes:y.data()      length:y.size() * sizeof(float)      options:MTLResourceStorageModeShared];
    id<MTLBuffer> pascalBuf = [g_device newBufferWithBytes:pascal_table.data() length:pascal_table.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> debugBuf = [g_device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];

    // 2. LUT & Dense Setup
    bool is_dense = (M < 20000); 
    uint32_t use_lut = 0;
    id<MTLBuffer> indicesBuf;
    id<MTLBuffer> lutBuf;
    uint32_t m_updates;
    
    if (is_dense) {
        std::cout << "[Metal] Dense Mode (M=" << M << ")\n";
        m_updates = M;
        std::vector<uint32_t> dense_indices(M);
        std::iota(dense_indices.begin(), dense_indices.end(), 0);
        indicesBuf = [g_device newBufferWithBytes:dense_indices.data() length:M * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        
        std::cout << "[Metal] Pre-computing Index LUT... ";
        std::vector<uint32_t> lut_data = generate_indices_lut(M, 4, D); 
        lutBuf = [g_device newBufferWithBytes:lut_data.data() length:lut_data.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        use_lut = 1;
        std::cout << "Done.\n";
    } else {
        std::cout << "[Metal] Sparse Mode\n";
        m_updates = 10000;
        indicesBuf = [g_device newBufferWithLength:m_updates * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        lutBuf = [g_device newBufferWithLength:4 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        use_lut = 0;
    }

    MetaGPU meta_struct{ 
        .D = D, .M = M, .m = m_updates, .learning_rate = learning_rate, .use_lut = use_lut 
    };

    // --- MINI-BATCH CONFIG ---
    // Fixes Race Condition. 256 is a sweet spot for M1 Max.
    uint32_t BATCH_SIZE = 256; 
    std::cout << "[Metal] Training on " << N_SAMPLES << " samples using Mini-Batch: " << BATCH_SIZE << "\n";
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pipeline];
            
            // Set Static Buffers
            [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0];
            [enc setBuffer:indicesBuf offset:0 atIndex:3];
            [enc setBytes:&meta_struct length:sizeof(MetaGPU) atIndex:4];
            [enc setBuffer:debugBuf offset:0 atIndex:5];
            [enc setBuffer:pascalBuf offset:0 atIndex:6];
            [enc setBuffer:lutBuf offset:0 atIndex:7];

            // MINI-BATCH LOOP
            for (uint32_t offset = 0; offset < N_SAMPLES; offset += BATCH_SIZE) {
                uint32_t current_batch = std::min(BATCH_SIZE, N_SAMPLES - offset);
                
                // Set Offsets into the Main Data Buffers
                // The kernel uses 'tgid' (0..255) to index. 
                // We add 'offset' to the pointer inside the kernel? 
                // NO, we use setBufferOffset so the kernel thinks it starts at 0.
                
                [enc setBuffer:allXBuf offset:(offset * D * sizeof(float)) atIndex:1];
                [enc setBuffer:allYBuf offset:(offset * sizeof(float)) atIndex:2];
                
                // Dispatch only the batch size
                [enc dispatchThreads:MTLSizeMake(current_batch * 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }
        
        if (epoch % 10 == 0) {
             std::memcpy(debug_output_host.data(), [debugBuf contents], 2 * sizeof(float));
             // std::cout << "Epoch " << epoch << "\n";
        }
    }
    std::cout << "[Metal] Syncing weights...\n";
    std::memcpy(coefficients.data(), [g_coefficientsBuf contents], coefficients.size() * sizeof(float));
}

void free_gpu_memory() { g_coefficientsBuf = nil; }
}