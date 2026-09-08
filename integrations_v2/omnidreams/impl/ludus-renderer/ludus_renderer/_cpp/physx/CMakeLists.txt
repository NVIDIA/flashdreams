# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.25)
project(ludus_physx_native LANGUAGES C CXX)

if(NOT DEFINED PHYSX_ROOT_DIR)
    message(FATAL_ERROR "PHYSX_ROOT_DIR must point at the pinned PhysX source tree")
endif()
if(NOT DEFINED PYBIND11_INCLUDE_DIR)
    message(FATAL_ERROR "PYBIND11_INCLUDE_DIR must point at Torch's pybind11 headers")
endif()

set(PX_BUILDSNIPPETS OFF CACHE BOOL "" FORCE)
set(PX_BUILDPVDRUNTIME OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_GPU_PROJECTS OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_GPU_PROJECTS_ONLY OFF CACHE BOOL "" FORCE)
set(PX_GENERATE_STATIC_LIBRARIES ON CACHE BOOL "" FORCE)
set(PX_CMAKE_SUPPRESS_REGENERATION ON CACHE BOOL "" FORCE)
set(NV_USE_STATIC_WINCRT OFF CACHE BOOL "" FORCE)
set(NV_USE_DEBUG_WINCRT OFF CACHE BOOL "" FORCE)

if(WIN32)
    set(TARGET_BUILD_PLATFORM windows CACHE STRING "" FORCE)
elseif(UNIX)
    set(TARGET_BUILD_PLATFORM linux CACHE STRING "" FORCE)
else()
    message(FATAL_ERROR "Ludus native PhysX supports Windows and Linux")
endif()

set(PX_OUTPUT_LIB_DIR "${CMAKE_BINARY_DIR}/physx-lib" CACHE PATH "" FORCE)
set(PX_OUTPUT_BIN_DIR "${CMAKE_BINARY_DIR}/physx-bin" CACHE PATH "" FORCE)
add_subdirectory("${PHYSX_ROOT_DIR}/compiler/public" physx-build)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
Python3_add_library(ludus_physx_native MODULE WITH_SOABI bindings.cpp)
target_compile_features(ludus_physx_native PRIVATE cxx_std_17)
target_include_directories(
    ludus_physx_native
    PRIVATE
        "${PHYSX_ROOT_DIR}/include"
        "${PYBIND11_INCLUDE_DIR}"
)
target_link_libraries(
    ludus_physx_native
    PRIVATE
        PhysX
        PhysXCommon
        PhysXExtensions
        PhysXCooking
        PhysXFoundation
)
target_compile_definitions(ludus_physx_native PRIVATE PX_PHYSX_STATIC_LIB)
set_target_properties(
    ludus_physx_native
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${LUDUS_MODULE_OUTPUT_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${LUDUS_MODULE_OUTPUT_DIR}"
)
foreach(configuration Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${configuration}" configuration_upper)
    set_target_properties(
        ludus_physx_native
        PROPERTIES
            "LIBRARY_OUTPUT_DIRECTORY_${configuration_upper}" "${LUDUS_MODULE_OUTPUT_DIR}"
            "RUNTIME_OUTPUT_DIRECTORY_${configuration_upper}" "${LUDUS_MODULE_OUTPUT_DIR}"
    )
endforeach()
