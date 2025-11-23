#pragma once
#include <vector>
#include <cstdint>

namespace metalsp {

struct Meta {
    uint32_t D;
    uint32_t M;
    uint32_t m;
    float    learning_rate;
};

// host-visible debug buffer (y_hat, grad0)
extern std::vector<float> debug_output_host;

// training hyperparameters (L2 + grad clip)
struct TrainHyper {
    float lambda = 1e-3f;
    float gmax   = 10.0f;
};
extern TrainHyper g_hparams;

} // namespace metalsp
