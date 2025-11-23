// src/gpu/metal/mc_network_metal.mm

#include "metalsp/network_types.hpp"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <iostream>

namespace metalsp {

// GLOBAL PERSISTENT BUFFERS
// This keeps the 10GB allocation alive on the GPU between function calls.
static id<MTLBuffer> g_coefficientsBuf = nil;

std::vector<float> debug_output_host(2, 0.0f);

id<MTLComputePipelineState> getPipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> pipeline = nil;
    if (!pipeline) {
        NSError *error = nil;
        NSString *path = @"../kernels/think.metal";
        NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (!source) throw std::runtime_error("Failed to read think.metal");

        id<MTLLibrary> library = [device newLibraryWithSource:source options:[[MTLCompileOptions alloc] init] error:&error];
        if (!library) throw std::runtime_error("Failed to compile Metal library");

        id<MTLFunction> func = [library newFunctionWithName:@"think_kernel"];
        pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    }
    return pipeline;
}

// UPDATED SIGNATURE: Added 'sync_weights' boolean
void mc_network_process_step(
    const std::vector<float>& x,
    float y,
    std::vector<float>& coefficients, // Only used for init or sync
    float learning_rate,
    std::vector<uint32_t>& update_indices,
    const std::vector<uint32_t>& pascal_table,
    bool sync_weights // <--- NEW CONTROL FLAG
)
{
    // 1. Setup Metal
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLComputePipelineState> pipeline = getPipeline(device);

    // 2. Check/Create Persistent Coefficient Buffer
    if (g_coefficientsBuf == nil) {
        // FIRST RUN ONLY: Allocate 10GB and Upload
        // We use StorageModeShared on M1 so CPU can read it later without explicit blit,
        // but we avoid re-creating this object every frame.
        std::cout << "[Metal] Allocating Persistent GPU Buffer (" 
                  << (coefficients.size() * sizeof(float) / 1024.0 / 1024.0) << " MB)... ";
        
        g_coefficientsBuf = [device newBufferWithBytes:coefficients.data() 
                                    length:coefficients.size() * sizeof(float) 
                                    options:MTLResourceStorageModeShared];
        std::cout << "Done.\n";
    }

    // 3. Create Transient Buffers (Input X, Indices, etc. are small -> Create every frame is fine)
    id<MTLBuffer> inputXBuf       = [device newBufferWithBytes:x.data() length:x.size() * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> targetYBuf      = [device newBufferWithBytes:&y length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> updateIndicesBuf= [device newBufferWithBytes:update_indices.data() length:update_indices.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> debugOutputBuf  = [device newBufferWithLength:2 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> pascalBuf       = [device newBufferWithBytes:pascal_table.data() length:pascal_table.size() * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    
    // Meta Struct
    Meta meta{
        .D = static_cast<uint32_t>(x.size()),
        .M = static_cast<uint32_t>(coefficients.size()),
        .m = static_cast<uint32_t>(update_indices.size()),
        .learning_rate = learning_rate
    };
    id<MTLBuffer> metaBuf = [device newBufferWithBytes:&meta length:sizeof(Meta) options:MTLResourceStorageModeShared];

    // 4. Encode
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    
    [enc setBuffer:g_coefficientsBuf offset:0 atIndex:0]; // PERSISTENT BUFFER
    [enc setBuffer:inputXBuf        offset:0 atIndex:1];
    [enc setBuffer:targetYBuf       offset:0 atIndex:2];
    [enc setBuffer:updateIndicesBuf offset:0 atIndex:3];
    [enc setBuffer:metaBuf          offset:0 atIndex:4];
    [enc setBuffer:debugOutputBuf   offset:0 atIndex:5];
    [enc setBuffer:pascalBuf        offset:0 atIndex:6];

    // Dispatch (Max 64 threads per group to match kernel logic)
    [enc dispatchThreads:MTLSizeMake(64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // 5. Read Back Results
    
    // Always read debug info (tiny)
    std::memcpy(debug_output_host.data(), [debugOutputBuf contents], 2 * sizeof(float));

    // ONLY read back the 10GB coefficients if explicitly requested
    if (sync_weights) {
        std::cout << "[Metal] Syncing Weights back to CPU... ";
        std::memcpy(coefficients.data(), [g_coefficientsBuf contents], coefficients.size() * sizeof(float));
        std::cout << "Done.\n";
    }
}

// Helper to clear GPU memory if needed (optional)
void free_gpu_memory() {
    g_coefficientsBuf = nil;
}

}