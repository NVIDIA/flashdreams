// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace omnidreams_singleview {

void bind_vae_streaming(pybind11::module_& module);

}  // namespace omnidreams_singleview
