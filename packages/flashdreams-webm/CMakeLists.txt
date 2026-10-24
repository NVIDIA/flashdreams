# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24)
if(CMAKE_HOST_WIN32 AND CMAKE_GENERATOR MATCHES "^Ninja")
  find_program(WEBM_C_COMPILER NAMES cl clang-cl)
  find_program(CMAKE_RC_COMPILER NAMES rc)
  find_program(CMAKE_MT NAMES mt)
  if(NOT WEBM_C_COMPILER OR NOT CMAKE_RC_COMPILER OR NOT CMAKE_MT)
    message(FATAL_ERROR
      "flashdreams-webm requires a Windows C++ compiler and Windows SDK tools "
      "(cl/clang-cl, rc, mt) on PATH for its Ninja source build.\n"
      "Python package builds initialize Visual Studio automatically. "
      "Install Visual Studio C++ Build Tools with a Windows SDK, then retry.")
  endif()
endif()
project(flashdreams_webm LANGUAGES C CXX)
if(MSVC)
  # SVT-AV1's variadic macros require the conforming preprocessor.
  add_compile_options(/Zc:preprocessor)
endif()
include(FetchContent)
include(ExternalProject)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(MSVC_RUNTIME dll CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_APPS OFF CACHE BOOL "" FORCE)
set(LOG_QUIET ON CACHE BOOL "" FORCE)
set(ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(ENABLE_SAMPLE_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ENABLE_WEBMTS OFF CACHE BOOL "" FORCE)
set(ENABLE_WEBMINFO OFF CACHE BOOL "" FORCE)

set(DAV1D_ASM true)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64")
  find_program(YASM_EXECUTABLE yasm)
  if(NOT YASM_EXECUTABLE AND CMAKE_HOST_WIN32)
    FetchContent_Declare(yasm
      URL https://www.tortall.net/projects/yasm/releases/yasm-1.3.0-win64.exe
      URL_HASH SHA256=d160b1d97266f3f28a71b4420a0ad2cd088a7977c2dd3b25af155652d8d8d91f
      DOWNLOAD_NO_EXTRACT TRUE)
    FetchContent_MakeAvailable(yasm)
    find_program(YASM_EXECUTABLE NAMES yasm yasm-1.3.0-win64
      PATHS "${yasm_SOURCE_DIR}" NO_DEFAULT_PATH REQUIRED)
  elseif(NOT YASM_EXECUTABLE)
    message(FATAL_ERROR "Install YASM for optimized x86 AV1 encoding.")
  endif()
  # ASM_NASM is CMake's language name; the compiler is explicitly YASM.
  set(CMAKE_ASM_NASM_COMPILER "${YASM_EXECUTABLE}" CACHE FILEPATH "" FORCE)
  # ponytail: dav1d's x86 assembly requires NASM; use C decoding until a YASM-compatible decoder is selected.
  set(DAV1D_ASM false)
endif()

FetchContent_Declare(svt
  URL https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v4.2.0/SVT-AV1-v4.2.0.tar.gz
  URL_HASH SHA256=c7b13c4a84bd3751aa35fcc72be13e6875467e7c2216879251a486e5b1e4e740
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(webm
  URL https://codeload.github.com/webmproject/libwebm/tar.gz/refs/tags/libwebm-1.0.0.32
  URL_HASH SHA256=7fd5e085bda9f8031cf2ad2a1e52d9b7b29cba9c0b96ad2ce794ce89e4249eb8
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(dav1d
  URL https://downloads.videolan.org/pub/videolan/dav1d/1.5.3/dav1d-1.5.3.tar.xz
  URL_HASH SHA256=732010aa5ef461fa93355ed2c6c5fedb48ddc4b74e697eaabe8907eaeb943011
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(svt webm dav1d)

# dav1d uses Meson; build just its static library inside the isolated wheel build.
set(DAV1D_BUILD "${CMAKE_CURRENT_BINARY_DIR}/dav1d-build")
set(DAV1D_LIBRARY "${DAV1D_BUILD}/src/libdav1d.a")
ExternalProject_Add(dav1d_build
  SOURCE_DIR "${dav1d_SOURCE_DIR}"
  BINARY_DIR "${DAV1D_BUILD}"
  DOWNLOAD_COMMAND ""
  CONFIGURE_COMMAND ${Python_EXECUTABLE} -m mesonbuild.mesonmain setup --reconfigure <BINARY_DIR> <SOURCE_DIR>
    --buildtype=release --default-library=static -Db_vscrt=md
    -Denable_asm=${DAV1D_ASM}
    -Denable_tools=false -Denable_tests=false -Denable_examples=false
  BUILD_COMMAND ${Python_EXECUTABLE} -m mesonbuild.mesonmain compile -C <BINARY_DIR>
  INSTALL_COMMAND ""
  BUILD_BYPRODUCTS "${DAV1D_LIBRARY}")

pybind11_add_module(_native src/bindings.cpp)
set_target_properties(_native PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_include_directories(_native PRIVATE
  "${svt_SOURCE_DIR}/Source/API" "${webm_SOURCE_DIR}"
  "${dav1d_SOURCE_DIR}/include" "${DAV1D_BUILD}/include")
target_compile_definitions(_native PRIVATE DAV1D_API= SVT_AV1_STATIC)
target_link_libraries(_native PRIVATE SvtAv1Enc webm "${DAV1D_LIBRARY}")
if(UNIX)
  find_package(Threads REQUIRED)
  target_link_libraries(_native PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()
add_dependencies(_native dav1d_build)
install(TARGETS _native LIBRARY DESTINATION flashdreams_webm COMPONENT flashdreams-webm)
install(DIRECTORY "${svt_SOURCE_DIR}/" DESTINATION flashdreams_webm/licenses/svt-av1
  COMPONENT flashdreams-webm FILES_MATCHING PATTERN "LICENSE*" PATTERN "PATENTS*"
  PATTERN ".git*" EXCLUDE)
install(FILES "${webm_SOURCE_DIR}/LICENSE.TXT" "${webm_SOURCE_DIR}/PATENTS.TXT"
  DESTINATION flashdreams_webm/licenses/libwebm COMPONENT flashdreams-webm)
install(FILES "${dav1d_SOURCE_DIR}/COPYING"
  DESTINATION flashdreams_webm/licenses/dav1d COMPONENT flashdreams-webm)
execute_process(COMMAND ${Python_EXECUTABLE} -c
  "from importlib.metadata import distribution; d=distribution('pybind11'); print(next(d.locate_file(f).as_posix() for f in d.files if str(f).endswith('licenses/LICENSE')))"
  OUTPUT_VARIABLE PYBIND11_LICENSE OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
install(FILES "${PYBIND11_LICENSE}"
  DESTINATION flashdreams_webm/licenses/pybind11 COMPONENT flashdreams-webm)
