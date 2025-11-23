cmake_minimum_required(VERSION 3.20)
# Project name changed to mc-network
project(mc-network LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Library Definition ---
add_library(mc-network 
    # GPU Metal Host Code (Changed name)
    src/gpu/metal/mc_network_metal.mm
)

# Public includes for headers
target_include_directories(mc-network 
    PUBLIC 
        include
)

# --- Link Metal frameworks on Apple platforms ---
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(mc-network
            PUBLIC
                ${METAL_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
        )
    endif()
endif()

# --- Demo Executable ---
# The demo executable is linked against the mc-network library.
add_executable(mc_network_demo
    examples/spmv_demo/main.cpp
)

target_link_libraries(mc_network_demo 
    PRIVATE 
        mc-network
)