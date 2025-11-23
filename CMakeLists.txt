cmake_minimum_required(VERSION 3.20)
project(mc-network LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Required for Python modules

# --- 1. CORE LIBRARY (Metal Backend) ---
add_library(mc-network-core STATIC
    src/gpu/metal/mc_network_metal.mm
)
target_include_directories(mc-network-core PUBLIC include)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(mc-network-core PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# --- 2. SETUP PYBIND11 (Automatic Download) ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1 # Stable version
)
# This makes pybind11 available to the rest of the script
FetchContent_MakeAvailable(pybind11)

# --- 3. PYTHON MODULE ---
# pybind11_add_module handles all the complex setup (suffixes, includes, flags) for us
pybind11_add_module(mc_network bindings.cpp)

# Link the core library to the python module
target_link_libraries(mc_network PRIVATE mc-network-core)

# --- 4. C++ DEMO (Optional) ---
add_executable(mc_network_demo examples/spmv_demo/main.cpp)
target_link_libraries(mc_network_demo PRIVATE mc-network-core)