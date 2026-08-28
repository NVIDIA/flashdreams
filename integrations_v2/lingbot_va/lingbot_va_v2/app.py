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

"""LingBot-VA Robotwin I2AV application for the FlashDreams V2 API."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from lingbot_va._loaders import validate_checkpoint_root
from lingbot_va.constants import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_PROMPT,
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ACTION_GUIDANCE_SCALE,
    ROBOTWIN_ACTION_INFERENCE_STEPS,
    ROBOTWIN_ACTION_SNR_SHIFT,
    ROBOTWIN_GUIDANCE_SCALE,
    ROBOTWIN_HEIGHT,
    ROBOTWIN_SNR_SHIFT,
    ROBOTWIN_USED_ACTION_CHANNEL_IDS,
    ROBOTWIN_VIDEO_INFERENCE_STEPS,
    ROBOTWIN_WIDTH,
)
from lingbot_va.engine import (
    LingbotVAEngine,
    LingbotVAEngineConfig,
    validate_device,
    validate_input_images,
)
from lingbot_va.utils import resolve_prompt

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from lingbot_va_v2.session import ACTIONS_SCHEMA, EngineFactory, LingbotVASession

_FRAMES_PER_SECOND = 10
"""Native Robotwin video playback rate."""


class LingbotVAApplication(IApplication):
    """Parse LingBot settings and create session-owned one-run engines."""

    def __init__(self, engine_factory: EngineFactory = LingbotVAEngine) -> None:
        """
        Args:
            engine_factory: Injectable model boundary for CPU lifecycle tests.
        """
        self._engine_factory = engine_factory
        self._config: LingbotVAEngineConfig | None = None

    def session_desc(self) -> SessionDesc:
        """Describe natural Robotwin outputs without parsing args or loading models."""
        return _session_desc()

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse effective overrides and validate startup state without weights."""
        args = _parse_args(commandline_args)
        prompt: str | Path = args.prompt_file or args.prompt
        config = LingbotVAEngineConfig(
            checkpoint_root=args.checkpoint_root,
            checkpoint_revision=args.checkpoint_revision,
            input_image_dir=args.input_image_dir,
            prompt=prompt,
            num_chunks=args.num_chunks,
            seed=args.seed,
            device=args.device,
            enable_offload=args.enable_offload,
            compile_network=args.compile_network,
            guidance_scale=args.guidance_scale,
            action_guidance_scale=args.action_guidance_scale,
            video_inference_steps=args.video_inference_steps,
            action_inference_steps=args.action_inference_steps,
            video_snr_shift=args.video_snr_shift,
            action_snr_shift=args.action_snr_shift,
        )
        validate_device(config.device)
        _validate_checkpoint_reference(config.checkpoint_root)
        validate_input_images(config.input_image_dir)
        resolve_prompt(config.prompt)
        self._config = config

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create an uninitialized session for the exact Robotwin contract."""
        if self._config is None:
            raise RuntimeError(
                "LingbotVAApplication.init() must run before create_session()."
            )
        canonical = _session_desc()
        _validate_requested_session(session_desc, canonical)
        resolved = replace(
            canonical,
            backpressure_mode=session_desc.backpressure_mode,
            presentation_mode=session_desc.presentation_mode,
            metadata={**session_desc.metadata, **canonical.metadata},
        )
        return LingbotVASession(
            self._config,
            resolved,
            self._engine_factory,
        )


def _parse_args(commandline_args: Sequence[str]) -> argparse.Namespace:
    """Parse only settings that alter the effective model run."""
    parser = argparse.ArgumentParser(
        prog="lingbot-va-robotwin-i2av",
        description="Generate Robotwin video and actions with LingBot-VA.",
    )
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument(
        "--input-image-dir",
        type=Path,
        required=True,
        help="Directory containing the three Robotwin camera PNGs.",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default=DEFAULT_PROMPT)
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--compile",
        dest="compile_network",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enable-offload", action="store_true")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=ROBOTWIN_GUIDANCE_SCALE,
    )
    parser.add_argument(
        "--action-guidance-scale",
        type=float,
        default=ROBOTWIN_ACTION_GUIDANCE_SCALE,
    )
    parser.add_argument(
        "--video-inference-steps",
        type=int,
        default=ROBOTWIN_VIDEO_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--action-inference-steps",
        type=int,
        default=ROBOTWIN_ACTION_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--video-snr-shift",
        type=float,
        default=ROBOTWIN_SNR_SHIFT,
    )
    parser.add_argument(
        "--action-snr-shift",
        type=float,
        default=ROBOTWIN_ACTION_SNR_SHIFT,
    )
    return parser.parse_args(list(commandline_args))


def _session_desc() -> SessionDesc:
    """Build the canonical natural Robotwin V2 session contract."""
    return SessionDesc(
        output_layout=VideoTensorLayout.tchw,
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
        frames_per_second_for_ui=_FRAMES_PER_SECOND,
        frames_per_second_for_step=_FRAMES_PER_SECOND,
        video_width=ROBOTWIN_WIDTH,
        video_height=ROBOTWIN_HEIGHT,
        tensor_artifact_schemas=(ACTIONS_SCHEMA,),
        metadata={
            "action_dim": ROBOTWIN_ACTION_DIM,
            "action_channel_ids": ROBOTWIN_USED_ACTION_CHANNEL_IDS,
        },
    )


def _validate_requested_session(
    requested: SessionDesc,
    canonical: SessionDesc,
) -> None:
    """Reject fixed-output changes while accepting runtime presentation policies."""
    fields = (
        "output_layout",
        "frames_per_second_for_ui",
        "frames_per_second_for_step",
        "video_width",
        "video_height",
        "tensor_artifact_schemas",
    )
    mismatches = [
        field_name
        for field_name in fields
        if getattr(requested, field_name) != getattr(canonical, field_name)
    ]
    if mismatches:
        raise ValueError(
            "LingBot-VA requires its natural Robotwin session contract; "
            "mismatched field(s): " + ", ".join(mismatches) + "."
        )


def _validate_checkpoint_reference(checkpoint_root: str | Path) -> None:
    """Validate existing/explicit local roots while accepting remote repo IDs."""
    value = str(checkpoint_root)
    expanded = Path(value).expanduser()
    if expanded.exists():
        validate_checkpoint_root(expanded)
        return
    if (
        isinstance(checkpoint_root, Path)
        or expanded.is_absolute()
        or value.startswith((".", "~"))
    ):
        raise FileNotFoundError(
            f"LingBot-VA checkpoint root does not exist: {expanded}"
        )


def create_app() -> IApplication:
    """Return a new uninitialized LingBot-VA V2 application."""
    return LingbotVAApplication()


__all__ = [
    "LingbotVAApplication",
    "create_app",
]
