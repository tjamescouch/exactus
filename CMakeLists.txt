cmake_minimum_required(VERSION 3.20)
project(mc-network LANGUAGES C CXX OBJC OBJCXX)

# ---- Options ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MACOSX_RPATH ON)

# ---- Dependencies: pybind11 ----
include(FetchContent)
set(PYBIND11_NEWPYTHON ON CACHE BOOL "" FORCE)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
)
FetchContent_MakeAvailable(pybind11)

# Let pybind11 find Python, but DO NOT leak Python includes into core.
# (No find_package(Python3) for core.)

# ---- Include paths ----
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- Sources ----
set(MC_NETWORK_CORE_SOURCES
  src/gpu/metal/mc_network_metal.mm
  src/gpu/metal/kernel_poly.mm
)

add_library(mc-network-core STATIC ${MC_NETWORK_CORE_SOURCES})

# Compile ObjC++ sources as such (even though this core is pure C++; keeps room for Metal later)
foreach(src ${MC_NETWORK_CORE_SOURCES})
  get_filename_component(ext ${src} EXT)
  if(ext STREQUAL ".mm")
    set_source_files_properties(${src} PROPERTIES
      COMPILE_FLAGS "-fobjc-arc"
      LANGUAGE OBJCXX
    )
  endif()
endforeach()

# Frameworks ONLY for the core if/when we actually use Metal APIs inside.
# Current CPU path doesn't require them, but keeping harmless links is OK.
if(APPLE)
  find_library(FRAMEWORK_METAL Metal)
  find_library(FRAMEWORK_FOUNDATION Foundation)
  if(FRAMEWORK_METAL AND FRAMEWORK_FOUNDATION)
    target_link_libraries(mc-network-core PUBLIC ${FRAMEWORK_METAL} ${FRAMEWORK_FOUNDATION})
  endif()
else()
  message(FATAL_ERROR "This project currently targets macOS.")
endif()

target_include_directories(mc-network-core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- pybind11 module (Python extension) ----
add_library(mc_network MODULE bindings.cpp)

set_target_properties(mc_network PROPERTIES
  PREFIX ""
  OUTPUT_NAME "mc_network"
  SUFFIX ".so"
)

# IMPORTANT: Only the binding target sees pybind11/Python.
target_link_libraries(mc_network
  PRIVATE
    pybind11::module
    mc-network-core
)

target_include_directories(mc_network
  PRIVATE
    ${pybind11_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---- Build type & warnings ----
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  target_compile_options(mc-network-core PRIVATE /W4)
  target_compile_options(mc_network PRIVATE /W4)
else()
  target_compile_options(mc-network-core PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(mc_network PRIVATE -Wall -Wextra -Wpedantic)
endif()

message(STATUS "Configured mc-network:")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  Pybind11:          OK")
