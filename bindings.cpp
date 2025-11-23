#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "metalsp/network_types.hpp"

namespace py = pybind11;

namespace metalsp {
    void mc_network_process_step(const std::vector<float>& x, float y, std::vector<float>& coefficients, float learning_rate, std::vector<uint32_t>& update_indices, const std::vector<uint32_t>& pascal_table, bool sync_weights);
    
    std::vector<float> mc_network_fit(
        const std::vector<float>& X_flat, const std::vector<float>& y,
        const std::vector<float>& coefficients, 
        float learning_rate, int epochs, int batch_size, int degree, // <--- Added degree
        py::array_t<uint32_t>& pascal_table
    );

    std::vector<float> mc_network_predict(
        const std::vector<float>& X_flat, 
        const std::vector<float>& coefficients,
        int input_dim, // <--- Added
        int degree,    // <--- Added
        py::array_t<uint32_t>& pascal_table
    );

    void free_gpu_memory(); 
    extern std::vector<float> debug_output_host;
}

PYBIND11_MODULE(mc_network, m) {
    m.doc() = "Bitwise Monte Carlo Network";
    m.def("process_step", &metalsp::mc_network_process_step);
    
    m.def("fit", &metalsp::mc_network_fit, 
          py::arg("X_flat"), py::arg("y"), py::arg("coefficients"),
          py::arg("learning_rate"), py::arg("epochs"), py::arg("batch_size"), py::arg("degree"),
          py::arg("pascal_table")
    );

    m.def("predict", &metalsp::mc_network_predict,
          py::arg("X_flat"), py::arg("coefficients"), py::arg("input_dim"), py::arg("degree"),
          py::arg("pascal_table")
    );

    m.def("reset_gpu", &metalsp::free_gpu_memory);
    m.def("get_debug_output", []() { return metalsp::debug_output_host; });
}