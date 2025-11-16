cmake_minimum_required(VERSION 3.20)
project(metalsp LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(metalsp
    src/core/csr_matrix.cpp
    src/core/spmv_cpu.cpp
    src/gpu/metal/spmv_metal.mm
)

target_include_directories(metalsp PUBLIC include)

# Link Metal frameworks on Apple platforms.
if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(metalsp
            PUBLIC
                ${METAL_FRAMEWORK}
                ${FOUNDATION_FRAMEWORK}
        )
    endif()
endif()

# Demo executable
add_executable(metalsp_demo
    examples/spmv_demo/main.cpp
)

target_link_libraries(metalsp_demo PRIVATE metalsp)
