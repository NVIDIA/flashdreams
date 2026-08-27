# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.24)
project(flashdreams_webm LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "flashdreams-webm currently builds Linux wheels only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)
include(ExternalProject)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Threads REQUIRED)
find_program(MAKE_EXECUTABLE NAMES gmake make REQUIRED)
find_program(VPX_ASSEMBLER NAMES nasm yasm REQUIRED)
get_filename_component(VPX_ASSEMBLER_DIRECTORY "${VPX_ASSEMBLER}" DIRECTORY)
get_filename_component(VPX_ASSEMBLER_NAME "${VPX_ASSEMBLER}" NAME)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

## libopus 1.6.1

set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
set(OPUS_DRED OFF CACHE BOOL "" FORCE)
set(OPUS_OSCE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  opus
  URL https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz
  URL_HASH SHA256=6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(opus)

## libwebm 1.0.0.32

set(ENABLE_WEBMTS OFF CACHE BOOL "" FORCE)
set(ENABLE_WEBMINFO OFF CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_WEBM_PARSER OFF CACHE BOOL "" FORCE)
set(ENABLE_SAMPLE_PROGRAMS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  libwebm_source
  URL https://github.com/webmproject/libwebm/archive/refs/tags/libwebm-1.0.0.32.tar.gz
  URL_HASH SHA256=7fd5e085bda9f8031cf2ad2a1e52d9b7b29cba9c0b96ad2ce794ce89e4249eb8
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(libwebm_source)

## libvpx 1.17.0

set(LIBVPX_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libvpx")
file(MAKE_DIRECTORY "${LIBVPX_PREFIX}/include")
ExternalProject_Add(
  libvpx_external
  URL https://github.com/webmproject/libvpx/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH SHA256=1020f184046187baa2985dbde38e0691f49c44088bca7a1842b0236c6081dc0a
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/libvpx-build"
  CONFIGURE_COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${VPX_ASSEMBLER_DIRECTORY}:$ENV{PATH}"
    <SOURCE_DIR>/configure
    --prefix=${LIBVPX_PREFIX}
    --as=${VPX_ASSEMBLER_NAME}
    --disable-docs
    --disable-examples
    --disable-libyuv
    --disable-shared
    --disable-tools
    --disable-unit-tests
    --disable-webm-io
    --enable-pic
    --enable-static
    --enable-vp8
    --enable-vp9
    --disable-vp8-decoder
    --disable-vp9-decoder
  BUILD_COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${VPX_ASSEMBLER_DIRECTORY}:$ENV{PATH}"
    "${MAKE_EXECUTABLE}" -j2
  INSTALL_COMMAND
    "${CMAKE_COMMAND}" -E env
    "PATH=${VPX_ASSEMBLER_DIRECTORY}:$ENV{PATH}"
    "${MAKE_EXECUTABLE}" install
  BUILD_BYPRODUCTS "${LIBVPX_PREFIX}/lib/libvpx.a"
)
add_library(libvpx STATIC IMPORTED GLOBAL)
set_target_properties(
  libvpx
  PROPERTIES
    IMPORTED_LOCATION "${LIBVPX_PREFIX}/lib/libvpx.a"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBVPX_PREFIX}/include"
)
add_dependencies(libvpx libvpx_external)

Python_add_library(
  _native
  MODULE
  WITH_SOABI
  src/native/module.cpp
  src/native/webm_writer.cpp
)
target_include_directories(
  _native
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/native"
    "${libwebm_source_SOURCE_DIR}"
)
target_compile_definitions(
  _native
  PRIVATE FLASHDREAMS_WEBM_LIBWEBM_VERSION="1.0.0.32"
)
target_link_libraries(
  _native
  PRIVATE libvpx Opus::opus webm Threads::Threads ${CMAKE_DL_LIBS} m
)
install(
  TARGETS _native
  LIBRARY DESTINATION flashdreams_webm
  COMPONENT flashdreams_webm
)
