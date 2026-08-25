/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "streaming_dit_bindings.h"

#include "pyext/streaming_dit_bridge.h"
#include <cuda_runtime.h>

namespace py = pybind11;

namespace omnidreams_singleview {

void bind_optimized_dit(py::module_& module) {
  module.def(
      "optimized_dit_forward",
      &::optimized_dit_forward,
      py::arg("x_new"),
      py::arg("condition_mask_patched"),
      py::arg("hdmap_patched"),
      py::arg("timesteps"),
      py::arg("rope_emb"),
      py::arg("k_cross_caches"),
      py::arg("v_cross_caches"),
      py::arg("k_self_caches"),
      py::arg("v_self_caches"),
      py::arg("self_attn_write_start"),
      py::arg("weights"),
      py::arg("config"));
  module.def(
      "sage3_quantize_cross_kv_bf16",
      &sage3_quantize_cross_kv_bf16,
      py::arg("k_bmhd"),
      py::arg("v_bmhd"));
  module.def("sage3_is_built", &sage3_is_built);
  module.def("sage3_is_runtime_supported", &sage3_is_runtime_supported, py::arg("device"));
  module.def("sparge_is_built", []() {
#ifdef OMNIDREAMS_SINGLEVIEW_HAS_SPARGE
    return true;
#else
    return false;
#endif
  });
  module.def("sparge_is_runtime_supported", [](int device) {
#ifdef OMNIDREAMS_SINGLEVIEW_HAS_SPARGE
    if (device < -1) {
      return false;
    }
    if (device == -1) {
      cudaError_t err = cudaGetDevice(&device);
      if (err != cudaSuccess) {
        return false;
      }
    }
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
      return false;
    }
    return (prop.major == 8 && prop.minor == 9) ||
           (prop.major == 12 && prop.minor == 0);
#else
    (void)device;
    return false;
#endif
  }, py::arg("device") = -1);
  module.def("optimized_dit_supports_block_mod_cache", []() { return true; });
  module.def("optimized_dit_supports_hdmap_cache", []() { return true; });
}

}  // namespace omnidreams_singleview
