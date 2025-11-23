#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "metalsp/network_types.hpp"

namespace py = pybind11;

namespace metalsp {
    // Existing single-step function
    void mc_network_process_step(
        const std::vector<float>& x,
        float y,
        std::vector<float>& coefficients,
        float learning_rate,
        std::vector<uint32_t>& update_indices,
        const std::vector<uint32_t>& pascal_table,
        bool sync_weights 
    );
    
    // NEW: The "Fit" function for Batch Processing
    void mc_network_fit(
        const std::vector<float>& X_flat, // Flattened 2D array (N * D)
        const std::vector<float>& y,      // Target vector (N)
        std::vector<float>& coefficients,
        float learning_rate,
        int epochs,
        const std::vector<uint32_t>& pascal_table
    );

    void free_gpu_memory(); 
    extern std::vector<float> debug_output_host;
}

PYBIND11_MODULE(mc_network, m) {
    m.doc() = "Bitwise Monte Carlo Network - Metal Backend";

    m.def("process_step", &metalsp::mc_network_process_step, "Run one training step");
    
    // NEW BINDING
    m.def("fit", &metalsp::mc_network_fit, 
          "Train the model on a full dataset",
          py::arg("X_flat"),
          py::arg("y"),
          py::arg("coefficients"),
          py::arg("learning_rate"),
          py::arg("epochs"),
          py::arg("pascal_table")
    );

    m.def("reset_gpu", &metalsp::free_gpu_memory, "Free persistent GPU memory");
    m.def("get_debug_output", []() { return metalsp::debug_output_host; });
}