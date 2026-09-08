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

"""Torch and native cuDNN scaled-dot-product attention."""

from flashdreams.accelerated.multi_head_attention.cudnn.native_fp8 import (
    native_cudnn_fp8_sdpa,
)
from flashdreams.accelerated.multi_head_attention.cudnn.torch_sdpa import (
    torch_cudnn_sdpa,
)

__all__ = ["native_cudnn_fp8_sdpa", "torch_cudnn_sdpa"]
