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

"""CPU tests for LingBot engine configuration and one-run lifecycle."""

from pathlib import Path

import pytest
import torch

from lingbot_va.constants import ROBOTWIN_OBS_CAM_KEYS
from lingbot_va.engine import (
    LingbotVAEngine,
    LingbotVAEngineConfig,
    LingbotVAEngineOutput,
    LingbotVAEngineState,
    build_pipeline_config,
    expected_output_shape,
    validate_input_images,
)

pytestmark = pytest.mark.ci_cpu


class _StandInEngine(LingbotVAEngine):
    """Return fixed CPU tensors without loading external model packages."""

    def _run_impl(self) -> LingbotVAEngineOutput:
        return LingbotVAEngineOutput(
            video=torch.zeros(2, 3, 4, 5),
            actions=torch.zeros(32, 16),
            metrics={"total_s": 0.0},
        )


class _FailingEngine(LingbotVAEngine):
    """Fail after entering the RUNNING state."""

    def _run_impl(self) -> LingbotVAEngineOutput:
        raise RuntimeError("inference failed")

    def _release_denoising_state(self) -> None:
        raise ValueError("cleanup failed")


def test_pipeline_config_applies_every_model_override(tmp_path: Path) -> None:
    config = LingbotVAEngineConfig(
        seed=17,
        compile_network=False,
        guidance_scale=2.5,
        action_guidance_scale=1.5,
        video_inference_steps=7,
        action_inference_steps=9,
        video_snr_shift=4.0,
        action_snr_shift=2.0,
    )

    resolved = build_pipeline_config(config, tmp_path)

    assert resolved.checkpoint_root == str(tmp_path)
    assert resolved.enable_sync_and_profile is False
    assert resolved.diffusion_model.seed == 17
    assert resolved.diffusion_model.transformer.checkpoint_root == str(tmp_path)
    assert resolved.diffusion_model.transformer.compile_network is False
    assert resolved.diffusion_model.transformer.guidance_scale == 2.5
    assert resolved.diffusion_model.transformer.action_guidance_scale == 1.5
    assert resolved.diffusion_model.scheduler.num_inference_steps == 7
    assert resolved.diffusion_model.scheduler.shift == 4.0
    assert resolved.action_scheduler.num_inference_steps == 9
    assert resolved.action_scheduler.shift == 2.0
    assert resolved.attn_window == 72
    assert resolved.diffusion_model.transformer.attn_window == 72


def test_engine_is_one_run_and_close_is_idempotent() -> None:
    engine = _StandInEngine(LingbotVAEngineConfig())

    output = engine.run()

    assert output.video.shape == (2, 3, 4, 5)
    assert output.actions.shape == (32, 16)
    assert engine.state is LingbotVAEngineState.FINISHED
    with pytest.raises(RuntimeError, match="requires NEW state"):
        engine.run()

    engine.close()
    engine.close()
    assert engine.state is LingbotVAEngineState.CLOSED


def test_engine_failure_closes_partial_state() -> None:
    engine = _FailingEngine(LingbotVAEngineConfig())

    with pytest.raises(RuntimeError, match="inference failed"):
        engine.run()

    assert engine.state is LingbotVAEngineState.CLOSED


def test_validate_input_images_names_all_missing_cameras(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as error:
        validate_input_images(tmp_path)

    for key in ROBOTWIN_OBS_CAM_KEYS:
        assert f"{key}.png" in str(error.value)


def test_validate_input_images_returns_camera_mapping(tmp_path: Path) -> None:
    for key in ROBOTWIN_OBS_CAM_KEYS:
        (tmp_path / f"{key}.png").touch()

    image_paths = validate_input_images(tmp_path)

    assert tuple(image_paths) == ROBOTWIN_OBS_CAM_KEYS


def test_expected_shape_scales_only_with_chunk_count() -> None:
    config = LingbotVAEngineConfig(num_chunks=3)

    assert expected_output_shape(config) == (21, 3, 256, 320)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"num_chunks": 0}, "num_chunks"),
        ({"video_inference_steps": 0}, "step counts"),
        ({"video_snr_shift": 0.0}, "SNR shifts"),
        ({"guidance_scale": -1.0}, "guidance scales"),
    ],
)
def test_engine_config_rejects_invalid_values(
    changes: dict[str, int | float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LingbotVAEngineConfig(**changes)  # type: ignore[arg-type]
