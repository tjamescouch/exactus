// src/gpu/metal/mc_network_metal.mm

#include "metalsp/spmv.hpp" // Keep for general structure, but not used directly here.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace metalsp {

// The new function that orchestrates the GPU execution.
void mc_network_process(const float* in_data, // Input buffer pointer
                        float* out_data,       // Output buffer pointer
                        std::size_t count)     // Number of elements in the buffer
{
    if (!in_data || !out_data) {
        throw std::runtime_error("mc_network_process: null input or output pointer");
    }

    if (count == 0) {
        return; // nothing to do
    }
    
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("mc_network_process: MTLCreateSystemDefaultDevice() returned nil");
    }

    NSError *error = nil;

    // --- 1. Load and Compile Kernel ---

    // Note the path change for the new kernel file name
    NSString *path = @"../kernels/think.metal"; 
    NSString *source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&error];
    if (!source || error) {
        throw std::runtime_error("mc_network_process: failed to read ../kernels/think.metal");
    }

    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (!library || error) {
        throw std::runtime_error("mc_network_process: failed to compile Metal library from source");
    }

    id<MTLFunction> func = [library newFunctionWithName:@"think_kernel"]; // New kernel name
    if (!func) {
        throw std::runtime_error("mc_network_process: kernel function 'think_kernel' not found");
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:func error:&error];
    if (!pipeline || error) {
        throw std::runtime_error("mc_network_process: failed to create compute pipeline state");
    }
    
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        throw std::runtime_error("mc_network_process: failed to create command queue");
    }
    
    // --- 2. Create and Populate Metal Buffers ---

    const MTLResourceOptions opts = MTLResourceStorageModeShared;
    const NSUInteger bufferLength = static_cast<NSUInteger>(count * sizeof(float));

    // Input buffer: Copy host data to GPU-accessible buffer
    id<MTLBuffer> inBuf =
        [device newBufferWithBytes:(void*)in_data
                            length:bufferLength
                           options:opts];

    // Output buffer: Allocate memory for result
    id<MTLBuffer> outBuf =
        [device newBufferWithLength:bufferLength
                            options:opts];

    if (!inBuf || !outBuf) {
        throw std::runtime_error("mc_network_process: failed to allocate one or more Metal buffers");
    }

    // --- 3. Encode, Dispatch, and Commit ---

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        throw std::runtime_error("mc_network_process: failed to create command buffer");
    }

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        throw std::runtime_error("mc_network_process: failed to create compute encoder");
    }

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:inBuf offset:0 atIndex:0];
    [enc setBuffer:outBuf offset:0 atIndex:1];

    // Setup Grid and Threadgroup sizes
    const NSUInteger threadsPerThreadgroup =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    
    MTLSize tgSize   = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    NSUInteger numThreads = static_cast<NSUInteger>(count);
    
    // Grid size must cover all elements.
    MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);

    [enc dispatchThreads:gridSize
  threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // --- 4. Read Results Back ---

    // Copy the result from the Metal buffer back to the host output pointer
    std::memcpy(out_data, [outBuf contents], bufferLength);
}

} // namespace metalsp