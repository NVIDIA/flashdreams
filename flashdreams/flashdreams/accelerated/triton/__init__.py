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

"""Public Triton attention and fused preprocessing kernels."""

# Importing the FlashAttention surface also installs the device-workspace
# allocator required by its in-kernel TMA tensor descriptors.
from flashdreams.accelerated.triton.flash_attention import (
    flash_attention_2_tma,
    is_tma_flash_attention_supported,
)
from flashdreams.accelerated.triton.rms_rope_kv_cache import (
    fused_rms_rope_kv_cache_update,
)

__all__ = [
    "flash_attention_2_tma",
    "fused_rms_rope_kv_cache_update",
    "is_tma_flash_attention_supported",
]
