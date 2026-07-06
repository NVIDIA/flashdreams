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

from dataclasses import field, make_dataclass
from typing import Annotated, Any

import pytest
import tyro
from omnidreams.config import RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoder,
    PixelShuffleVAEEncoderConfig,
)
from omnidreams.runner import OmnidreamsRunnerConfig

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.runner import RunnerConfig
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


def test_omnidreams_lighttae_runner_cli_defaults_parse() -> None:
    runner = RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
    subcommand_union: Any = tyro.extras.subcommand_type_from_defaults(
        defaults={runner.runner_name: runner},
        descriptions={runner.runner_name: runner.description},
        prefix_names=False,
        sort_subcommands=True,
    )
    union: Any = tyro.conf.SuppressFixed[tyro.conf.FlagConversionOff[subcommand_union]]
    args_cls = make_dataclass(
        "FlashdreamsRunArgsForTest",
        [
            ("runner", Annotated[union, tyro.conf.arg(name="")]),
            ("no_instantiate", bool, field(default=False)),
        ],
    )

    args = tyro.cli(args_cls, args=[runner.runner_name])
    config: RunnerConfig = getattr(args, "runner")

    assert isinstance(config, OmnidreamsRunnerConfig)
    assert config.runner_name == runner.runner_name
