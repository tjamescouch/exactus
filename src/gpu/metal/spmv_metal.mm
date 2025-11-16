// src/gpu/metal/spmv_metal.mm

#include "metalsp/spmv.hpp"
#include "metalsp/csr_matrix.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <limits>

namespace metalsp {

namespace {

// Must match the struct in kernels/spmv.metal
struct CsrMetaCpu {
    uint32_t numRows;
};

id<MTLDevice> getDevice()
{
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("spmv_metal: MTLCreateSystemDefaultDevice() returned nil");
    }
    return device;
}

id<MTLComputePipelineState> getPipeline(id<MTLDevice> device)
{
    static id<MTLComputePipelineState> pipeline = nil;
    static bool initialized = false;

    if (!initialized) {
        initialized = true;

        NSError *error = nil;

        // Load the Metal source from disk.
        // Assumes you run from build/ and kernels/ is at ../kernels/spmv.metal
        NSString *path = @"../kernels/spmv.metal";
        NSString *source =
            [NSString stringWithContentsOfFile:path
                                      encoding:NSUTF8StringEncoding
                                         error:&error];
        if (!source || error) {
            throw std::runtime_error("spmv_metal: failed to read ../kernels/spmv.metal");
        }

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library =
            [device newLibraryWithSource:source options:options error:&error];
        if (!library || error) {
            throw std::runtime_error("spmv_metal: failed to compile Metal library from source");
        }

        id<MTLFunction> func = [library newFunctionWithName:@"spmv_csr"];
        if (!func) {
            throw std::runtime_error("spmv_metal: kernel function 'spmv_csr' not found");
        }

        pipeline = [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline || error) {
            throw std::runtime_error("spmv_metal: failed to create compute pipeline state");
        }
    }

    return pipeline;
}

} // namespace

void spmv_metal(const CsrMatrix& A,
                const float* x,
                float* y)
{
    if (!x || !y) {
        throw std::runtime_error("spmv_metal: null x or y pointer");
    }

    if (A.numRows == 0 || A.numCols == 0) {
        return; // nothing to do
    }

    if (A.numRows > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()) ||
        A.nnz     > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("spmv_metal: matrix too large for 32-bit indices");
    }

    id<MTLDevice> device   = getDevice();
    id<MTLComputePipelineState> pipeline = getPipeline(device);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        throw std::runtime_error("spmv_metal: failed to create command queue");
    }

    // ---- Host-side staging buffers ----

    std::vector<uint32_t> rowPtr32(A.rowPtr.size());
    std::vector<uint32_t> colInd32(A.colInd.size());

    for (std::size_t i = 0; i < A.rowPtr.size(); ++i) {
        rowPtr32[i] = static_cast<uint32_t>(A.rowPtr[i]);
    }
    for (std::size_t i = 0; i < A.colInd.size(); ++i) {
        colInd32[i] = static_cast<uint32_t>(A.colInd[i]);
    }

    std::vector<float> xHost(A.numCols);
    std::memcpy(xHost.data(), x, A.numCols * sizeof(float));

    std::vector<float> yHost(A.numRows, 0.0f);

    CsrMetaCpu meta{};
    meta.numRows = static_cast<uint32_t>(A.numRows);

    const MTLResourceOptions opts = MTLResourceStorageModeShared;

    id<MTLBuffer> rowPtrBuf =
        [device newBufferWithBytes:rowPtr32.data()
                            length:rowPtr32.size() * sizeof(uint32_t)
                           options:opts];

    id<MTLBuffer> colIndBuf =
        [device newBufferWithBytes:colInd32.data()
                            length:colInd32.size() * sizeof(uint32_t)
                           options:opts];

    id<MTLBuffer> valuesBuf =
        [device newBufferWithBytes:A.values.data()
                            length:A.values.size() * sizeof(float)
                           options:opts];

    id<MTLBuffer> xBuf =
        [device newBufferWithBytes:xHost.data()
                            length:xHost.size() * sizeof(float)
                           options:opts];

    id<MTLBuffer> yBuf =
        [device newBufferWithLength:yHost.size() * sizeof(float)
                            options:opts];

    id<MTLBuffer> metaBuf =
        [device newBufferWithBytes:&meta
                            length:sizeof(CsrMetaCpu)
                           options:opts];

    if (!rowPtrBuf || !colIndBuf || !valuesBuf || !xBuf || !yBuf || !metaBuf) {
        throw std::runtime_error("spmv_metal: failed to allocate one or more Metal buffers");
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) {
        throw std::runtime_error("spmv_metal: failed to create command buffer");
    }

    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) {
        throw std::runtime_error("spmv_metal: failed to create compute encoder");
    }

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:rowPtrBuf offset:0 atIndex:0];
    [enc setBuffer:colIndBuf offset:0 atIndex:1];
    [enc setBuffer:valuesBuf offset:0 atIndex:2];
    [enc setBuffer:xBuf      offset:0 atIndex:3];
    [enc setBuffer:yBuf      offset:0 atIndex:4];
    [enc setBuffer:metaBuf   offset:0 atIndex:5];

    const NSUInteger numRows = static_cast<NSUInteger>(A.numRows);
    const NSUInteger maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger threadsPerThreadgroup =
        std::min<NSUInteger>(maxThreads, 64);

    MTLSize tgSize   = MTLSizeMake(threadsPerThreadgroup, 1, 1);
    NSUInteger groups =
        (numRows + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    MTLSize gridSize = MTLSizeMake(groups * threadsPerThreadgroup, 1, 1);

    [enc dispatchThreads:gridSize
  threadsPerThreadgroup:tgSize];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    std::memcpy(yHost.data(), [yBuf contents], yHost.size() * sizeof(float));

    for (std::size_t i = 0; i < A.numRows; ++i) {
        y[i] = yHost[i];
    }
}

} // namespace metalsp
