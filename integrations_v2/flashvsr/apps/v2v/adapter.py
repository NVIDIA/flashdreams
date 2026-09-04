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

"""FlashVSR binding for the reusable v2v application."""

from __future__ import annotations

from flashvsr.impl.postprocess import (
    POSTPROCESS_PRESET_FLASHVSR_V1_1_FULL_ATTN,
    POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_1_5,
    POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_2_0,
    FlashVSRPostProcessorConfig,
    _chunk_mode,
)
from v2v import V2VApplication, V2VApplicationDefaults

from flashdreams.api_v2.application import IApplication


def _create_application(
    processor: FlashVSRPostProcessorConfig,
    *,
    model_name: str,
) -> V2VApplication:
    first_chunk_size, steady_chunk_size = _chunk_mode(processor.chunk_size)
    return V2VApplication(
        defaults=V2VApplicationDefaults(
            processor=processor,
            first_chunk_size=first_chunk_size,
            steady_chunk_size=steady_chunk_size,
            model_name=model_name,
        )
    )


def create_app() -> IApplication:
    """Create the stable sparse FlashVSR V2V application."""
    return _create_application(
        POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_2_0,
        model_name="flashvsr-v1.1-sparse-ratio-2.0",
    )


def create_app_sparse_ratio_1_5() -> IApplication:
    """Create the faster sparse FlashVSR V2V application."""
    return _create_application(
        POSTPROCESS_PRESET_FLASHVSR_V1_1_SPARSE_1_5,
        model_name="flashvsr-v1.1-sparse-ratio-1.5",
    )


def create_app_full_attn() -> IApplication:
    """Create the dense-attention FlashVSR V2V application."""
    return _create_application(
        POSTPROCESS_PRESET_FLASHVSR_V1_1_FULL_ATTN,
        model_name="flashvsr-v1.1-full-attn",
    )


__all__ = ["create_app", "create_app_full_attn", "create_app_sparse_ratio_1_5"]
