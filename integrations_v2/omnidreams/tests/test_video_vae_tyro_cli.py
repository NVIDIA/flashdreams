# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import tyro
from omnidreams.impl.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoder,
    PixelShuffleVAEEncoderConfig,
)

from flashdreams.infra.config import InstantiateConfig
from flashdreams.recipes.taehv import (
    TaehvVAEDecoder,
    TaehvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoder, WanVAEEncoderConfig

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("config_cls", "target_cls"),
    [
        (PixelShuffleVAEEncoderConfig, PixelShuffleVAEEncoder),
        (TaehvVAEDecoderConfig, TaehvVAEDecoder),
        (WanVAEEncoderConfig, WanVAEEncoder),
    ],
)
def test_video_vae_config_cli_defaults(
    config_cls: type[InstantiateConfig], target_cls: type
) -> None:
    config = tyro.cli(config_cls, args=[])
    assert isinstance(config, config_cls)
    # Compare by qualified name: importlib mode can create distinct class
    # objects for the same source when rootdir differs from the package root.
    actual = f"{config._target.__module__}.{config._target.__qualname__}"
    expected = f"{target_cls.__module__}.{target_cls.__qualname__}"
    assert actual == expected


def test_pixelshuffle_cli_accepts_frame_selection_override() -> None:
    config = tyro.cli(
        PixelShuffleVAEEncoderConfig,
        args=["--frame-selection-mode", "first_frame"],
    )
    assert config.frame_selection_mode == "first_frame"
