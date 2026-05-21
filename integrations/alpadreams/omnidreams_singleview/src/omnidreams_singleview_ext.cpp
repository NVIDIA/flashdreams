/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

#ifndef OMNIDREAMS_SINGLEVIEW_CUTLASS_SHA
#define OMNIDREAMS_SINGLEVIEW_CUTLASS_SHA "unknown"
#endif

#ifndef OMNIDREAMS_SINGLEVIEW_CUTLASS_PATCH_SHA
#define OMNIDREAMS_SINGLEVIEW_CUTLASS_PATCH_SHA "unknown"
#endif

#ifndef OMNIDREAMS_SINGLEVIEW_CUTLASS_OVERLAY_SHA
#define OMNIDREAMS_SINGLEVIEW_CUTLASS_OVERLAY_SHA "unknown"
#endif

#ifndef OMNIDREAMS_SINGLEVIEW_SOURCE_SHA
#define OMNIDREAMS_SINGLEVIEW_SOURCE_SHA "unknown"
#endif

#ifndef OMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA
#define OMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA "unknown"
#endif

#ifndef OMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA
#define OMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA "unknown"
#endif

namespace {

bool is_available() {
  return true;
}

pybind11::dict build_info() {
  pybind11::dict info;
  info["cutlass_sha"] = OMNIDREAMS_SINGLEVIEW_CUTLASS_SHA;
  info["cutlass_patch_sha256"] = OMNIDREAMS_SINGLEVIEW_CUTLASS_PATCH_SHA;
  info["cutlass_overlay_sha256"] = OMNIDREAMS_SINGLEVIEW_CUTLASS_OVERLAY_SHA;
  info["extension_source_sha256"] = OMNIDREAMS_SINGLEVIEW_SOURCE_SHA;
  info["sage_attention_sha"] = OMNIDREAMS_SINGLEVIEW_SAGE_ATTENTION_SHA;
  info["sparge_attn_sha"] = OMNIDREAMS_SINGLEVIEW_SPARGE_ATTN_SHA;
  info["with_cuda"] = false;
  return info;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("is_available", &is_available);
  module.def("build_info", &build_info);
}
