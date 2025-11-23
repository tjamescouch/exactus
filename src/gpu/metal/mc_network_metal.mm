// src/gpu/metal/mc_network_metal.mm

#include "metalsp/network_types.hpp"
#include "metalsp/spmv.hpp" // Keep for function declaration reference

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <random>

namespace metalsp {

// Helper to set up the pipeline state (modified from previous versions)
id<MTLComputePipelineState> getPipeline(id<MTLDevice> device)
{
    static id<MTLComputePipelineState> pipeline = nil;
    static bool initialized = false;

    if (!initialized) {
        initialized = true;

        NSError *error = nil;
        NSString *path = @"../kernels/think.metal";
        NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
        if (!source || error) {
            throw std::runtime_error("mc_network_process: failed to read ../kernels/think.metal");
        }

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library || error) {
            // --- NEW DEBUGGING CODE START ---
            if (error) {
                NSLog(@"Metal Compilation Error: %@", error.localizedDescription);
                NSLog(@"User Info: %@", error.userInfo);
            }
            // --- NEW DEBUGGING CODE END ---
            throw std::runtime_error("mc_network_process: failed to compile Metal library from source. Check logs for details.");
        }

        id<MTLFunction> func = [library newFunctionWithName:@"think_kernel"];
        if (!func) {
            throw std::runtime_error("mc_network_process: kernel function 'think_kernel' not found");
        }

        pipeline = [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline || error) {
            throw std::runtime_error("mc_network_process: failed to create compute pipeline state");
        }
    }

    return pipeline;
}

// Function to run the full forward/backward step
void mc_network_process_step(
    const std::vector<float>& x,       // Input (D=2)
    float y,                           // Target (1)
    std::vector<float>& coefficients,  // Coefficients (M=6, R/W)
    float learning_rate,               // eta
    std::vector<uint32_t>& update_indices // Sampled indices (m)
)
{
    // --- Configuration ---
    if (x.size() != 2 || coefficients.size() != 6) {
        throw std::runtime_error("mc_network_process_step: D or M size mismatch for test case (D=2, M=6)");
    }
    
    // Debug buffer to store y_hat and loss_gradient from the GPU
    // [0]: y_hat, [1]: loss_gradient
    std::vector<float> debug_output_host(2); 

    Meta meta{};
    meta.D = 2;
    meta.M = 6;
    meta.m = static_cast<uint32_t>(update_indices.size());
    meta.learning_rate = learning_rate;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("mc_network_process: Failed to create Metal device.");
    id<MTLComputePipelineState> pipeline = getPipeline(device);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) throw std::runtime_error("mc_network_process: Failed to create command queue.");

    const MTLResourceOptions opts = MTLResourceStorageModeShared;
    
    // --- 1. Buffer Setup ---

    id<MTLBuffer> coefficientsBuf =
        [device newBufferWithBytes:coefficients.data()
                            length:coefficients.size() * sizeof(float)
                           options:opts];

    id<MTLBuffer> inputXBuf =
        [device newBufferWithBytes:x.data()
                            length:x.size() * sizeof(float)
                           options:opts];

    id<MTLBuffer> targetYBuf =
        [device newBufferWithBytes:&y
                            length:sizeof(float)
                           options:opts];

    id<MTLBuffer> updateIndicesBuf =
        [device newBufferWithBytes:update_indices.data()
                            length:update_indices.size() * sizeof(uint32_t)
                           options:opts];

    id<MTLBuffer> metaBuf =
        [device newBufferWithBytes:&meta
                            length:sizeof(Meta)
                           options:opts];

    id<MTLBuffer> debugOutputBuf =
        [device newBufferWithLength:debug_output_host.size() * sizeof(float)
                            options:opts];


    // --- 2. Encode and Dispatch ---

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

// --- 2. Encode and Dispatch (CORRECTED BINDINGS) ---

    // ... (encoder setup code) ...

    [enc setComputePipelineState:pipeline];
    
    // Bindings must match kernel signature:
    [enc setBuffer:coefficientsBuf  offset:0 atIndex:0]; // Buffer 0: Coefficients (R/W)
    [enc setBuffer:inputXBuf        offset:0 atIndex:1]; // Buffer 1: Input X
    [enc setBuffer:targetYBuf       offset:0 atIndex:2]; // Buffer 2: Target Y
    [enc setBuffer:updateIndicesBuf offset:0 atIndex:3]; // Buffer 3: Update Indices
    [enc setBuffer:metaBuf          offset:0 atIndex:4]; // Buffer 4: Metadata
    [enc setBuffer:debugOutputBuf   offset:0 atIndex:5]; // Buffer 5: Debug Output

    // ... (dispatch code) ...

    // Dispatch a grid where the number of threads equals the Monte Carlo sample size (m)
    const NSUInteger threadsPerThreadgroup =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 64);
    
    MTLSize tgSize   = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    NSUInteger groups = (meta.m + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    MTLSize gridSize = MTLSizeMake(groups * threadsPerThreadgroup, 1, 1);

    [enc dispatchThreads:gridSize
  threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // --- 3. Read Results Back ---

    // Copy updated coefficients back to host
    std::memcpy(coefficients.data(), [coefficientsBuf contents], coefficients.size() * sizeof(float));
    
    // Copy debug info back
    std::memcpy(debug_output_host.data(), [debugOutputBuf contents], debug_output_host.size() * sizeof(float));
    
    // Log for verification
    std::cout << "  > y_hat: " << debug_output_host[0] << ", Target: " << y 
              << ", Error: " << debug_output_host[0] - y << "\n";
}

} // namespace metalsp