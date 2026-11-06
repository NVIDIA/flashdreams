// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compile SageAttention-2's SM90 attention kernel into the OmniDreams native
// extension without registering SageAttention's standalone pybind module.
#include "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu"
