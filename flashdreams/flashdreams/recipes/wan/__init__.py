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

"""Public Wan recipe surface for integration plugins."""

from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoder,
    WanVAEDecoderConfig,
    WanVAEEncoder,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import (
    WanInferencePipeline,
    WanInferencePipelineCache,
    WanInferencePipelineConfig,
)
from flashdreams.recipes.wan.transformer.constants import NEGATIVE_PROMPT
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork,
    WanDiTNetwork1pt3BConfig,
    WanDiTNetwork14BConfig,
    WanDiTNetworkConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerConfig,
)
from flashdreams.recipes.wan.transformer.wan22 import (
    Wan22Transformer,
    Wan22TransformerConfig,
)

__all__ = [
    "AVAILABLE_WAN_VAE_CHECKPOINT_PATHS",
    "NEGATIVE_PROMPT",
    "Wan21Transformer",
    "Wan21TransformerConfig",
    "Wan22Transformer",
    "Wan22TransformerConfig",
    "WanDiTNetwork",
    "WanDiTNetwork1pt3BConfig",
    "WanDiTNetwork14BConfig",
    "WanDiTNetworkConfig",
    "WanI2VCtrlEncoderConfig",
    "WanInferencePipeline",
    "WanInferencePipelineCache",
    "WanInferencePipelineConfig",
    "WanVAEDecoder",
    "WanVAEDecoderConfig",
    "WanVAEEncoder",
    "WanVAEEncoderConfig",
]
