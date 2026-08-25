// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <torch/python.h>

// Build SageAttention-3's FP4 quantization kernels into OmniDreams single-view native extension without
// registering Sage's standalone pybind module.
#undef PYBIND11_MODULE
#define PYBIND11_MODULE(name, variable) \
  static void native_sage3_fp4_quant_pybind_stub(pybind11::module_& variable)

#include "sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu"

#undef PYBIND11_MODULE
