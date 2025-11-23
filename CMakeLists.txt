cmake_minimum_required(VERSION 3.20)
project(mc-network LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library (Same as before)
add_library(mc-network
    src/gpu/metal/mc_network_metal.mm
)
target_include_directories(mc-network PUBLIC include)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(mc-network PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
    endif()
endif()

# Demo Executable
add_executable(mc_network_demo examples/spmv_demo/main.cpp)
target_link_libraries(mc_network_demo PRIVATE mc-network)

# --- NEW: Benchmark Executable ---
add_executable(mc_network_benchmark examples/spmv_demo/benchmark_main.cpp)
# Note: Benchmark doesn't strictly need the metal library if it's just testing CPU allocation, 
# but we link it for consistency.
target_link_libraries(mc_network_benchmark PRIVATE mc-network)