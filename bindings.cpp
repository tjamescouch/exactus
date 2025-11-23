#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "metalsp/network_types.hpp"

namespace py = pybind11;

namespace metalsp {
  std::vector<float> mc_network_fit(
    const std::vector<float>& X_flat,
    const std::vector<float>& y,
    const std::vector<float>& coefficients_in,
    float learning_rate,
    int   epochs,
    int   batch_size,
    int   degree,
    py::array_t<uint32_t>& pascal_table_np);

  std::vector<float> mc_network_predict(
    const std::vector<float>& X_flat,
    const std::vector<float>& coefficients,
    int input_dim,
    int degree,
    py::array_t<uint32_t>& pascal_table_np);

  void free_gpu_memory();
  void set_train_hyperparams(float lambda, float gmax);

  extern std::vector<float> debug_output_host;
}

PYBIND11_MODULE(mc_network, m) {
  // Wrap in lambdas so pybind can deduce Func (handles numpy arg cleanly)
  m.def("fit",
        [](std::vector<float> X_flat,
           std::vector<float> y,
           std::vector<float> coefficients_in,
           float learning_rate,
           int   epochs,
           int   batch_size,
           int   degree,
           py::array_t<uint32_t> pascal_table_np) {
          return metalsp::mc_network_fit(
              X_flat, y, coefficients_in, learning_rate, epochs, batch_size, degree, pascal_table_np);
        },
        py::arg("X_flat"), py::arg("y"), py::arg("coefficients_in"),
        py::arg("learning_rate"), py::arg("epochs"), py::arg("batch_size"),
        py::arg("degree"), py::arg("pascal_table_np"));

  m.def("predict",
        [](std::vector<float> X_flat,
           std::vector<float> coefficients,
           int input_dim,
           int degree,
           py::array_t<uint32_t> pascal_table_np) {
          return metalsp::mc_network_predict(
              X_flat, coefficients, input_dim, degree, pascal_table_np);
        },
        py::arg("X_flat"), py::arg("coefficients"),
        py::arg("input_dim"), py::arg("degree"),
        py::arg("pascal_table_np"));

  m.def("free_gpu_memory", &metalsp::free_gpu_memory);

  // Expose ridge + clamp knobs; avoid Python keyword conflict by naming lambda_
  m.def("set_train_hyperparams",
        [](float lambda_, float gmax) { metalsp::set_train_hyperparams(lambda_, gmax); },
        py::arg("lambda_"), py::arg("gmax"));

  m.def("debug_output", []() {
    const py::ssize_t n = static_cast<py::ssize_t>(metalsp::debug_output_host.size());
    return py::array_t<float>(n, metalsp::debug_output_host.data());
  });
}
