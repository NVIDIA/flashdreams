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

import pytest
import tyro
from omnidreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoder,
    PixelShuffleVAEEncoderConfig,
)

from flashdreams.infra.config import InstantiateConfig
from flashdreams.recipes.taehv import (
    TeahvVAEDecoder,
    TeahvVAEDecoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import WanVAEEncoder, WanVAEEncoderConfig

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    ("config_cls", "target_cls"),
    [
        (PixelShuffleVAEEncoderConfig, PixelShuffleVAEEncoder),
        (TeahvVAEDecoderConfig, TeahvVAEDecoder),
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
