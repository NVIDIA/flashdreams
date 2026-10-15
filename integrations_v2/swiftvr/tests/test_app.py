# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    compiled = resolve_postprocess_preset("swiftvr-2x-compiled")

    assert isinstance(preset, SwiftVRPostProcessorConfig)
    assert preset.scale == 4
    assert isinstance(twice, SwiftVRPostProcessorConfig)
    assert twice.scale == 2
    assert twice.chunk_size == 8
    assert isinstance(compiled, SwiftVRPostProcessorConfig)
    assert compiled.compile_blocks
    assert not compiled.compile_reae_encoder
    assert compiled.compile_reae_decoder
