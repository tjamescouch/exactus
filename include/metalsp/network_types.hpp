#pragma once
#include <vector>
#include <cstdint>

namespace metalsp {

std::vector<float> mc_network_fit(
    const std::vector<float>& X_flat,
    const std::vector<float>& y,
    const std::vector<float>& coefficients_in,
    float learning_rate,
    int   epochs,
    int   batch_size,
    int   degree,
    ... /* pybind passes numpy uint32 Pascal */);

std::vector<float> mc_network_predict(
    const std::vector<float>& X_flat,
    const std::vector<float>& coefficients,
    int input_dim,
    int degree,
    ... /* pybind passes numpy uint32 Pascal */);

void free_gpu_memory();

/** Set training hyperparams used by the Metal trainer (global, thread-safe for single process):
 *  lambda ≥ 0 (ridge / weight decay), gmax > 0 (per-batch gradient clamp in apply pass).
 */
void set_train_hyperparams(float lambda, float gmax);

// host-visible debug mirror [yhat0, grad0]
extern std::vector<float> debug_output_host;

} // namespace metalsp
