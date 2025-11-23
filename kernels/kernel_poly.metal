#include <metal_stdlib>
using namespace metal;

struct Params {
    uint N;
    uint D;
    uint degree;
};

kernel void poly_kernel_mv(
    device const float*   X      [[buffer(0)]],   // N*D
    device const float*   u      [[buffer(1)]],   // N
    device float*         v      [[buffer(2)]],   // N
    constant Params&      P      [[buffer(3)]],
    uint                  i      [[thread_position_in_grid]])
{
    if (i >= P.N) return;

    // Row i pointer
    const device float* xi = X + (size_t)i * P.D;

    float acc = 0.0f;
    for (uint j = 0; j < P.N; ++j) {
        const device float* xj = X + (size_t)j * P.D;

        // dot(xi, xj)
        float dotv = 0.0f;
        // unroll a bit in case D is big – keep it simple for now
        for (uint k = 0; k < P.D; ++k) {
            dotv += xi[k] * xj[k];
        }
        float kij = powr(1.0f + dotv, (float)P.degree);
        acc += kij * u[j];
    }
    v[i] = acc;
}
