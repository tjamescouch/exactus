// NOTE: keep this as ObjC++ (.mm) but pure C++ right now.
// DO NOT include any pybind11 headers in this file.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>  // harmless to include; we aren't using it yet

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include "metalsp/network_types.hpp"

namespace metalsp {

// ---- globals ----
std::vector<float> debug_output_host = {0.f, 0.f};
TrainHyper g_hparams;

// ---- combinatorics ----
static inline uint64_t nck_u64(uint32_t n, uint32_t k) {
    if (k > n) return 0;
    if (k == 0 || k == n) return 1;
    k = std::min(k, n - k);
    long double acc = 1.0;
    for (uint32_t i = 1; i <= k; ++i) {
        acc *= (long double)(n - (k - i));
        acc /= (long double)i;
    }
    return (uint64_t)std::llround(acc);
}

static inline uint32_t monomial_count(uint32_t D, uint32_t deg) {
    return (uint32_t)nck_u64(D + deg - 1, deg);
}

// decode k-th multiset combination of size 'deg' from [0..D-1]
static void decode_monomial_indices(uint32_t k, uint32_t D, uint32_t deg, std::vector<uint32_t>& out) {
    out.clear();
    if (deg == 0) return;
    uint32_t prev = 0;
    uint32_t rem = k;
    for (uint32_t i = 0; i < deg; ++i) {
        for (uint32_t v = prev; v < D; ++v) {
            uint64_t rest = nck_u64((D - v) + (deg - i - 1) - 1, (deg - i - 1));
            if (rest <= rem) {
                rem -= (uint32_t)rest;
                continue;
            }
            out.push_back(v);
            prev = v;
            break;
        }
    }
}

// compute phi(x) for degree=deg
static float eval_monomial(const float* x, const std::vector<uint32_t>& idx) {
    float p = 1.f;
    for (uint32_t j : idx) p *= x[j];
    return p;
}

// yhat = Phi * w (streamed)
static void predict_deg(const std::vector<float>& X_flat, uint32_t N, uint32_t D,
                        uint32_t deg, const std::vector<float>& w,
                        std::vector<float>& yhat)
{
    yhat.assign(N, 0.f);
    const uint32_t M = monomial_count(D, deg);
    std::vector<uint32_t> idx; idx.reserve(deg);

    for (uint32_t n = 0; n < N; ++n) {
        const float* x = &X_flat[n * D];
        float sum = 0.f;
        for (uint32_t k = 0; k < M; ++k) {
            decode_monomial_indices(k, D, deg, idx);
            sum += w[k] * eval_monomial(x, idx);
        }
        yhat[n] = sum;
    }
}

// SGD with L2 and gradient clipping
static void fit_deg(const std::vector<float>& X_flat,
                    const std::vector<float>& y,
                    uint32_t N, uint32_t D, uint32_t deg,
                    std::vector<float>& w, float lr,
                    int epochs, int bs)
{
    const uint32_t M = monomial_count(D, deg);
    w.assign(M, 0.f);

    std::vector<uint32_t> idx; idx.reserve(deg);
    std::vector<float> grad(M, 0.f);
    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0U);

    auto l2 = [](const std::vector<float>& v) {
        long double s = 0.0;
        for (float a : v) s += (long double)a * (long double)a;
        return (double)std::sqrt((double)s);
    };

    for (int ep = 0; ep < epochs; ++ep) {
        // simple epoch order; optional shuffle can be added
        for (uint32_t off = 0; off < N; off += (uint32_t)bs) {
            uint32_t end = std::min(off + (uint32_t)bs, N);
            std::fill(grad.begin(), grad.end(), 0.f);

            for (uint32_t ii = off; ii < end; ++ii) {
                uint32_t n = order[ii];
                const float* x = &X_flat[n * D];

                // forward
                float yh = 0.f;
                for (uint32_t k = 0; k < M; ++k) {
                    decode_monomial_indices(k, D, deg, idx);
                    yh += w[k] * eval_monomial(x, idx);
                }

                float err = yh - y[n];
                if (n == 0 && ep == 0) {
                    debug_output_host[0] = yh;
                    debug_output_host[1] = 2.f * err; // d/dy of squared loss
                }

                const float lg = 2.f * err; // dL/dy
                for (uint32_t k = 0; k < M; ++k) {
                    decode_monomial_indices(k, D, deg, idx);
                    float phi = eval_monomial(x, idx);
                    grad[k] += lg * phi;
                }
            }

            // average + L2 + clip
            float invB = 1.f / float(end - off);
            for (uint32_t k = 0; k < M; ++k) grad[k] = invB * grad[k] + g_hparams.lambda * w[k];

            double gn = l2(grad);
            if (gn > g_hparams.gmax && gn > 0.0) {
                float s = (float)(g_hparams.gmax / gn);
                for (uint32_t k = 0; k < M; ++k) grad[k] *= s;
            }

            for (uint32_t k = 0; k < M; ++k) w[k] -= lr * grad[k];
        }
    }
}

// --------- public C++ API used by bindings.cpp ---------

// Configure training hyperparameters
void set_train_hyperparams(float lambda, float gmax) {
    g_hparams.lambda = lambda;
    g_hparams.gmax   = gmax;
}

// Fit a single-degree model on already-expanded polynomial family (via combinadics inside).
// X_flat: N*D, y: N, coefficients_in: M (ignored here; starting from zeros), degree: K
std::vector<float> mc_network_fit(const std::vector<float>& X_flat,
                                  const std::vector<float>& y,
                                  const std::vector<float>& /*coefficients_in*/,
                                  float learning_rate,
                                  int epochs,
                                  int batch_size,
                                  int degree,
                                  const void* /*pascal_ignored*/)
{
    if (y.empty()) return {};
    const uint32_t D = (uint32_t)(X_flat.size() / y.size());
    const uint32_t N = (uint32_t)y.size();
    std::vector<float> w;
    fit_deg(X_flat, y, N, D, (uint32_t)degree, w, learning_rate, epochs, batch_size);
    return w;
}

// Predict using learned weights
std::vector<float> mc_network_predict(const std::vector<float>& X_flat,
                                      const std::vector<float>& w,
                                      int D,
                                      int degree,
                                      const void* /*pascal_ignored*/)
{
    const uint32_t N = (uint32_t)(X_flat.size() / (uint32_t)D);
    std::vector<float> yhat; yhat.reserve(N);
    predict_deg(X_flat, N, (uint32_t)D, (uint32_t)degree, w, yhat);
    return yhat;
}

// Expose debug
const std::vector<float>& get_debug_output() {
    return debug_output_host;
}

} // namespace metalsp
