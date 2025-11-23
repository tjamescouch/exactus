#import <Metal/Metal.h>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "metalsp/kernel_poly.hpp"

namespace {
id<MTLDevice> dev() {
    static id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) throw std::runtime_error("Metal device not available.");
    return d;
}
id<MTLLibrary> lib_from_source(NSString* src) {
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> L = [dev() newLibraryWithSource:src options:opts error:&err];
    if (!L) {
        std::string msg = "Metal compile error: ";
        msg += err ? [[err localizedDescription] UTF8String] : "unknown";
        throw std::runtime_error(msg);
    }
    return L;
}
id<MTLComputePipelineState> pipeline_poly() {
    static std::once_flag once;
    static id<MTLComputePipelineState> pso = nil;
    static id<MTLLibrary> L = nil;
    std::call_once(once, ^{
        // Load the .metal file at build time; CMake sets it as a resource we can read.
        // Simpler: embed small source string here. Safer for drop-in.
        NSString* src =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "struct Params { uint N; uint D; uint degree; };\n"
         "kernel void poly_kernel_mv(device const float* X [[buffer(0)]],\n"
         "                           device const float* u [[buffer(1)]],\n"
         "                           device float* v [[buffer(2)]],\n"
         "                           constant Params& P [[buffer(3)]],\n"
         "                           uint i [[thread_position_in_grid]]) {\n"
         "  if (i >= P.N) return;\n"
         "  const device float* xi = X + (size_t)i * P.D;\n"
         "  float acc = 0.0f;\n"
         "  for (uint j = 0; j < P.N; ++j) {\n"
         "    const device float* xj = X + (size_t)j * P.D;\n"
         "    float dotv = 0.0f;\n"
         "    for (uint k = 0; k < P.D; ++k) dotv += xi[k]*xj[k];\n"
         "    float kij = powr(1.0f + dotv, (float)P.degree);\n"
         "    acc += kij * u[j];\n"
         "  }\n"
         "  v[i] = acc;\n"
         "}\n";
        L = lib_from_source(src);
        NSError* err = nil;
        id<MTLFunction> fn = [L newFunctionWithName:@"poly_kernel_mv"];
        pso = [dev() newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) {
            std::string msg = "PSO creation failed: ";
            msg += err ? [[err localizedDescription] UTF8String] : "unknown";
            throw std::runtime_error(msg);
        }
    });
    return pso;
}
} // anon

namespace metalsp {

std::vector<float> kernel_poly_matvec(const std::vector<float>& X_flat,
                                      int32_t N, int32_t D,
                                      const std::vector<float>& u,
                                      int32_t degree)
{
    if (N <= 0 || D <= 0) throw std::invalid_argument("N,D must be >0");
    if ((int64_t)X_flat.size() != (int64_t)N * (int64_t)D)
        throw std::invalid_argument("X_flat size != N*D");
    if ((int)u.size() != N)
        throw std::invalid_argument("u size != N");

    id<MTLCommandQueue> Q = [dev() newCommandQueue];
    id<MTLBuffer> bx = [dev() newBufferWithBytes:X_flat.data()
                                          length:sizeof(float)*X_flat.size()
                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> bu = [dev() newBufferWithBytes:u.data()
                                          length:sizeof(float)*u.size()
                                         options:MTLResourceStorageModeShared];
    std::vector<float> v(N, 0.0f);
    id<MTLBuffer> bv = [dev() newBufferWithBytes:v.data()
                                          length:sizeof(float)*v.size()
                                         options:MTLResourceStorageModeShared];
    struct { uint32_t N, D, degree; } P = { (uint32_t)N, (uint32_t)D, (uint32_t)degree };
    id<MTLBuffer> bp = [dev() newBufferWithBytes:&P length:sizeof(P)
                                         options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipeline_poly()];
    [enc setBuffer:bx offset:0 atIndex:0];
    [enc setBuffer:bu offset:0 atIndex:1];
    [enc setBuffer:bv offset:0 atIndex:2];
    [enc setBuffer:bp offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake((NSUInteger)N, 1, 1);
    NSUInteger tg = pipeline_poly().maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    MTLSize tgs = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    // copy back
    std::memcpy(v.data(), [bv contents], sizeof(float)*v.size());
    return v;
}

} // namespace metalsp
