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

"""CPU contract tests for the SwiftVR v2v binding."""

from typing import Any, cast

import pytest
from swiftvr.apps.v2v.adapter import create_app
from swiftvr.impl.postprocess import SwiftVRPostProcessorConfig

from flashdreams.plugins.registry import resolve_postprocess_preset

pytestmark = pytest.mark.ci_cpu


def test_entry_point_binds_swiftvr_defaults() -> None:
    application = cast(Any, create_app())

    assert application.defaults.model_name == "swiftvr-2x"
    assert application.defaults.first_chunk_size == 8
    assert application.defaults.steady_chunk_size == 8
    assert application.defaults.processor.scale == 2


def test_postprocess_preset_is_discoverable() -> None:
    preset = resolve_postprocess_preset("swiftvr-4x")
    twice = resolve_postprocess_preset("swiftvr-2x")

    assert isinstance(preset, SwiftVRPostProcessorConfig)
    assert preset.scale == 4
    assert isinstance(twice, SwiftVRPostProcessorConfig)
    assert twice.scale == 2
    assert twice.chunk_size == 8
