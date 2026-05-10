#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(portfolio_engine, m) {
    py::class_<PortfolioEngine>(m, "PortfolioEngine")
        .def(py::init<int, double, double, double>(), 
             py::arg("num_assets"), 
             py::arg("initial_nav"), 
             py::arg("taker_fee_bps"), 
             py::arg("impact_coeff"))
        .def("step", &PortfolioEngine::step, 
             py::arg("target_weights"), 
             py::arg("asset_returns"))
        .def("get_nav", &PortfolioEngine::get_nav)
        .def("get_weights", &PortfolioEngine::get_weights);
}
