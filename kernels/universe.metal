#include <metal_stdlib>
using namespace metal;

// Parameters mirrored from SimParams (keep in sync).
struct Params {
  uint    width;
  uint    height;
  float   dt;
  float   eta;
  float   m2;
  float   lambda_;
  float   alpha0;
  float   alpha1;
  float   alpha2;
  float   W00, W01, W10, W11;
};

// Wrap-around indexing (toroidal world).
inline uint2 wrap(uint2 p, uint W, uint H) {
  return uint2((p.x + W) % W, (p.y + H) % H);
}

// Sample single-channel slice with wrap.
inline float sread(texture2d_array<float, access::read> tex, uint2 p, uint slice) {
  uint2 q = wrap(p, tex.get_width(), tex.get_height());
  return tex.read(uint2(q.x, q.y), slice).r;
}

// Separable 5-tap band-pass via (hi - lo) approximations.
// Taps chosen to be compact and stable; three scales differ by stride.
constant float T5[5] = { -1.0f/16.0f, -1.0f/4.0f, 0.0f, 1.0f/4.0f, 1.0f/16.0f }; // high-ish
constant float S5[5] = {  1.0f/16.0f,  1.0f/4.0f, 3.0f/8.0f, 1.0f/4.0f, 1.0f/16.0f }; // smooth
inline float sep_band(texture2d_array<float, access::read> tex, uint2 p, uint slice, uint stride) {
  // DoG-like: conv with T5 horizontally+vertically + subtract a smoothed version.
  float h = 0.0f;
  for (int k=-2; k<=2; ++k) {
    h += T5[k+2] * sread(tex, uint2(p.x + k*stride, p.y), slice);
  }
  float v = 0.0f;
  for (int k=-2; k<=2; ++k) {
    v += T5[k+2] * sread(tex, uint2(p.x, p.y + k*stride), slice);
  }
  float hi = 0.5f*(h + v);

  float hs=0.0f, vs=0.0f;
  for (int k=-2; k<=2; ++k) {
    hs += S5[k+2] * sread(tex, uint2(p.x + k*stride, p.y), slice);
    vs += S5[k+2] * sread(tex, uint2(p.x, p.y + k*stride), slice);
  }
  float lo = 0.5f*(hs + vs);
  return hi - lo;
}

kernel void wavelet_step(
  texture2d_array<float, access::read>   U_t     [[texture(0)]],
  texture2d_array<float, access::write>  U_next  [[texture(1)]],
  constant Params& P                               [[buffer(0)]],
  uint2 gid                                        [[thread_position_in_grid]])
{
  if (gid.x >= P.width || gid.y >= P.height) return;

  // Read current field (two channels).
  float u0 = sread(U_t, gid, 0);
  float u1 = sread(U_t, gid, 1);

  // Propagation: 3 scales, different strides.
  float p0_0 = sep_band(U_t, gid, 0, 1);
  float p1_0 = sep_band(U_t, gid, 0, 2);
  float p2_0 = sep_band(U_t, gid, 0, 4);
  float prop0 = P.alpha0*p0_0 + P.alpha1*p1_0 + P.alpha2*p2_0;

  float p0_1 = sep_band(U_t, gid, 1, 1);
  float p1_1 = sep_band(U_t, gid, 1, 2);
  float p2_1 = sep_band(U_t, gid, 1, 4);
  float prop1 = P.alpha0*p0_1 + P.alpha1*p1_1 + P.alpha2*p2_1;

  // Local nonlinear interaction (φ^4-like + channel coupling).
  float mix0 = P.W00 * u0 + P.W01 * u1;
  float mix1 = P.W10 * u0 + P.W11 * u1;

  float nonlin0 = P.m2 * u0 + P.lambda_ * (u0*u0*u0) + (u0 * mix0);
  float nonlin1 = P.m2 * u1 + P.lambda_ * (u1*u1*u1) + (u1 * mix1);

  float next0 = (1.0f - P.eta) * (prop0 - P.dt * nonlin0);
  float next1 = (1.0f - P.eta) * (prop1 - P.dt * nonlin1);

  U_next.write(next0, gid, 0);
  U_next.write(next1, gid, 1);
}
