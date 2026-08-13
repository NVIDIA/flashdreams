# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the raw float16 token codec and its config wiring.

Uses tiny CPU tensors only; no CUDA is required.
"""

from __future__ import annotations

import pytest
import torch

from flashdreams.serving.token_stream.codec import (
    RawFloat16TokenCodec,
    RawFloat16TokenCodecConfig,
    TokenCodec,
    TokenCodecConfig,
)

pytestmark = pytest.mark.ci_cpu


def test_raw_codec_id() -> None:
    codec = RawFloat16TokenCodecConfig().setup()
    assert codec.codec_id == "raw_f16"


def test_raw_codec_static_params_empty() -> None:
    codec = RawFloat16TokenCodecConfig().setup()
    assert codec.static_params == {}


def test_raw_codec_encode_frame_matches_float16_bytes() -> None:
    codec = RawFloat16TokenCodecConfig().setup()
    latent = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)

    result = codec.encode_frame(latent)

    expected = latent.to(torch.float16).contiguous().cpu().numpy().tobytes()
    assert result.payload == expected
    assert result.frame_params == b""


def test_raw_config_setup_returns_raw_codec() -> None:
    config = RawFloat16TokenCodecConfig()
    assert config._target is RawFloat16TokenCodec
    codec = config.setup()
    assert isinstance(codec, RawFloat16TokenCodec)
    assert isinstance(codec, TokenCodec)
    assert codec.config is config


def test_base_config_target_is_base_codec() -> None:
    config = TokenCodecConfig()
    assert config._target is TokenCodec
