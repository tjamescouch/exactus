#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "metalsp/network_types.hpp"

namespace py = pybind11;

namespace metalsp {
// C++ functions implemented in mc_network_metal.mm
std::vector<float> mc_network_fit(const std::vector<float>& X_flat,
                                  const std::vector<float>& y,
                                  const std::vector<float>& coefficients_in,
                                  float learning_rate,
                                  int epochs,
                                  int batch_size,
                                  int degree,
                                  const void* pascal_ignored);

std::vector<float> mc_network_predict(const std::vector<float>& X_flat,
                                      const std::vector<float>& w,
                                      int D,
                                      int degree,
                                      const void* pascal_ignored);

void set_train_hyperparams(float lambda, float gmax);
const std::vector<float>& get_debug_output();
} // namespace metalsp

PYBIND11_MODULE(mc_network, m) {
    m.doc() = "Polynomial logic/feature model (CPU path) with a stable Python API";

    // keep the signature exactly as you were calling from Python
    m.def("fit",
          [](const std::vector<float>& X_flat,
             const std::vector<float>& y,
             const std::vector<float>& coefficients_in,
             float learning_rate,
             int epochs,
             int batch_size,
             int degree,
             py::array_t<uint32_t> /*pascal*/){
              // pascal currently unused in CPU path
              return metalsp::mc_network_fit(X_flat, y, coefficients_in,
                                             learning_rate, epochs, batch_size, degree, nullptr);
          },
          py::arg("X_flat"),
          py::arg("y"),
          py::arg("coefficients_in"),
          py::arg("learning_rate"),
          py::arg("epochs"),
          py::arg("batch_size"),
          py::arg("degree"),
          py::arg("pascal_table_np"));

    m.def("predict",
          [](const std::vector<float>& X_flat,
             const std::vector<float>& w,
             int D,
             int degree,
             py::array_t<uint32_t> /*pascal*/){
              return metalsp::mc_network_predict(X_flat, w, D, degree, nullptr);
          },
          py::arg("X_flat"),
          py::arg("w"),
          py::arg("D"),
          py::arg("degree"),
          py::arg("pascal_table_np"));

    m.def("set_train_hyperparams",
          [](double lambda, double gmax){
              metalsp::set_train_hyperparams((float)lambda, (float)gmax);
          },
          py::arg("lambda_") = 1e-3, py::arg("gmax") = 10.0);

    m.def("debug_output",
          [](){
              const auto& v = metalsp::get_debug_output();
              return v;
          });
}
