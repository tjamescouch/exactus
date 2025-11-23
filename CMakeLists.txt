cmake_minimum_required(VERSION 3.20)
project(mc-network LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 1. SETUP PYBIND11 (Fetch FIRST) ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# --- 2. CORE LIBRARY (Metal Backend) ---
add_library(mc-network-core STATIC
    src/gpu/metal/mc_network_metal.mm
)
target_include_directories(mc-network-core PUBLIC include)

# CRITICAL FIX: Link Pybind11 headers to the Core Library
target_link_libraries(mc-network-core PUBLIC pybind11::pybind11)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    target_link_libraries(mc-network-core PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
endif()

# --- 3. PYTHON MODULE ---
pybind11_add_module(mc_network bindings.cpp)

# Link the core library to the python module
target_link_libraries(mc_network PRIVATE mc-network-core)

# Force output to project root
set_target_properties(mc_network PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# --- 4. C++ DEMO (Optional) ---
# Note: This might fail to link if main.cpp doesn't have Python symbols, 
# so we can comment it out or leave it if you only run the Python script.
# add_executable(mc_network_demo examples/spmv_demo/main.cpp)
# target_link_libraries(mc_network_demo PRIVATE mc-network-core)