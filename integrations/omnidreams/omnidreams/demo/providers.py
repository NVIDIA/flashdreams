# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams model-input providers for shared demo run modes."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from loguru import logger
from omnidreams.runner import _load_video

from flashdreams.infra.runner_io import (
    DEFAULT_RUNNER_INSTALL_HINT,
    load_first_frame_tensor,
)
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.demo import (
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.runtime.demo.session_inputs import ControlDecision
from flashdreams.runtime.inputs import InferenceInput, InferenceInputSchema, InputField
from flashdreams.runtime.types import StepRequirements

from .spec import OmnidreamsReplayScenario


class PrecomputedHDMapProvider:
    """Prepare fixed OmniDreams HDMap conditioning for replay-style runs."""

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        config: InferenceConfig,
    ) -> None:
        self._scenario = _scenario_from_prepared(scenario)
        self._device = _device_from_config(config)
        self._dtype = torch.bfloat16
        self._frame_start = 0
        self._closed = False
        self.capabilities = ProviderCapabilities(
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=precomputed_hdmap_inference_input_schema(),
        )
        self._hdmap_videos: torch.Tensor | None = self._load_hdmaps()

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        scenario = self._scenario
        first_frames = [
            load_first_frame_tensor(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self._device,
                dtype=self._dtype,
                allow_video=True,
                install_hint=DEFAULT_RUNNER_INSTALL_HINT,
            )
            for path in scenario.first_frame_paths
        ]
        return InferenceInput(
            global_conditioning={
                "scenario": scenario,
                "prompt": [list(scenario.prompts)],
                "first_frame": torch.stack(first_frames, dim=0).unsqueeze(0),
            },
            metadata={"view_names": tuple(scenario.camera_names)},
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del user_window
        self._require_open()
        hdmap_videos = self._require_hdmaps()
        frame_end = self._frame_start + request.input_frame_count
        if frame_end > hdmap_videos.shape[2]:
            return PreparedStep(
                control=ControlDecision(
                    close_session=True,
                    reason="OmniDreams precomputed HDMap input exhausted.",
                )
            )

        frame_start = self._frame_start
        self._frame_start = frame_end
        return PreparedStep(
            inference_input=InferenceInput(
                step={"hdmap": hdmap_videos[:, :, frame_start:frame_end]},
                metadata={
                    "hdmap_frame_start": frame_start,
                    "hdmap_frame_end": frame_end,
                },
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._frame_start = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hdmap_videos = None

    def _load_hdmaps(self) -> torch.Tensor:
        scenario = self._scenario
        videos = [
            _load_video(
                path,
                pixel_height=scenario.pixel_height,
                pixel_width=scenario.pixel_width,
                device=self._device,
                dtype=self._dtype,
            )
            for path in scenario.hdmap_video_paths
        ]
        hdmap_videos = torch.stack(videos, dim=0).unsqueeze(0)
        if _is_rank_zero():
            logger.info(
                "Loaded OmniDreams demo HDMaps shape={} views={}",
                tuple(hdmap_videos.shape),
                len(scenario.camera_names),
            )
        return hdmap_videos

    def _require_hdmaps(self) -> torch.Tensor:
        hdmap_videos = self._hdmap_videos
        if hdmap_videos is None:
            raise RuntimeError("OmniDreams precomputed HDMap provider is closed.")
        return hdmap_videos

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("OmniDreams precomputed HDMap provider is closed.")


def precomputed_hdmap_inference_input_schema() -> InferenceInputSchema:
    return InferenceInputSchema(
        global_conditioning_fields=(
            InputField(
                name="prompt",
                input_modality="omnidreams/prompt",
                description="OmniDreams prompt batch.",
            ),
            InputField(
                name="first_frame",
                input_modality="video/frame",
                description="Initial OmniDreams conditioning frame tensor.",
            ),
            InputField(
                name="scenario",
                required=False,
                input_modality="omnidreams/replay-scenario",
                description="Resolved OmniDreams replay scenario metadata.",
            ),
        ),
        step_fields=(
            InputField(
                name="hdmap",
                input_modality="omnidreams/hdmap-video",
                frequency_consumed="per_step",
                description="Per-step HDMap conditioning chunk.",
            ),
        ),
    )


def _scenario_from_prepared(scenario: PreparedScenario) -> OmnidreamsReplayScenario:
    value = scenario.initial_inputs.global_conditioning.get("scenario")
    if not isinstance(value, OmnidreamsReplayScenario):
        raise TypeError(
            "OmniDreams precomputed HDMap provider requires "
            "initial_inputs.global_conditioning['scenario'] to be an "
            "OmnidreamsReplayScenario."
        )
    return value


def _device_from_config(config: InferenceConfig) -> torch.device:
    if dist.is_initialized():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device(config.device or "cuda")


def _is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


__all__ = [
    "PrecomputedHDMapProvider",
    "precomputed_hdmap_inference_input_schema",
]
