# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slow dummy model and application for interactive Cam2V UI testing."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.api_v2.application import IApplication

from .application import Cam2VApplication
from .defaults import Cam2VApplicationDefaults, Cam2VConditioning
from .session import CameraControlInput

_DUMMY_FRAME_PATH = Path(__file__).with_name("assets") / "dummy_frame.ppm"
"""Packaged first frame used by the dummy rollout."""


@dataclass(frozen=True, slots=True)
class DummyCam2VCache:
    """Per-rollout first frame retained by the dummy pipeline."""

    first_frame: Tensor
    """Normalized ``[C, H, W]`` background on the requested device."""


class DummyCam2VDecoder:
    """Expose the spatial contract required by :class:`Cam2VApplication`."""

    spatial_compression_ratio = 1
    """Keep dummy pixels at their requested output resolution."""


class DummyCam2VPipeline:
    """Generate lightweight camera-tinted frames after a configurable wait."""

    def __init__(self, *, step_wait_seconds: float, frames_per_chunk: int) -> None:
        """Configure simulated model latency and chunk size.

        Args:
            step_wait_seconds: Time each generation call waits.
            frames_per_chunk: Frames returned by each generation call.

        Raises:
            ValueError: The wait is negative or the chunk is empty.
        """
        if step_wait_seconds < 0:
            raise ValueError("step_wait_seconds must be >= 0.")
        if frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be > 0.")
        self.step_wait_seconds = float(step_wait_seconds)
        self.frames_per_chunk = int(frames_per_chunk)
        self.decoder = DummyCam2VDecoder()
        self._device = torch.device("cpu")
        self._sleep = threading.Event()

    def to(self, device: torch.device | str) -> "DummyCam2VPipeline":
        """Select the device used for generated dummy frames."""
        self._device = torch.device(device)
        return self

    def eval(self) -> "DummyCam2VPipeline":
        """Return this stateless dummy pipeline in inference form."""
        return self

    def initialize_cache(self, *, text: list[str], image: Tensor) -> DummyCam2VCache:
        """Retain one normalized first frame for a dummy rollout.

        Args:
            text: Prompt accepted for Cam2V pipeline compatibility.
            image: First frame shaped ``[1, C, H, W]``.

        Returns:
            Per-rollout background state.

        Raises:
            ValueError: ``image`` does not contain one RGB frame.
        """
        del text
        if image.ndim != 4 or image.shape[:2] != (1, 3):
            raise ValueError("Dummy Cam2V requires one RGB first frame.")
        return DummyCam2VCache(
            first_frame=image[0].to(device=self._device, dtype=torch.float32)
        )

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        """Return the fixed dummy chunk size."""
        del autoregressive_index
        return self.frames_per_chunk

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: DummyCam2VCache,
        input: CameraControlInput,
    ) -> Tensor:
        """Wait like the real model and tint frames from camera motion.

        Args:
            autoregressive_index: Zero-based dummy chunk index.
            cache: Per-rollout first-frame state.
            input: Integrated camera intrinsics and poses.

        Returns:
            Normalized video shaped ``[T, C, H, W]``.
        """
        self._sleep.wait(self.step_wait_seconds)
        poses = input.poses.to(device=self._device, dtype=torch.float32)
        translations = poses[:, :3, 3]
        yaw = poses[:, 0, 2:3]
        tint = torch.cat(
            (
                translations[:, 0:1] + yaw,
                translations[:, 1:2],
                translations[:, 2:3] - yaw,
            ),
            dim=1,
        ).tanh()
        phase = torch.arange(
            self.frames_per_chunk,
            device=self._device,
            dtype=torch.float32,
        )
        phase = (phase + autoregressive_index * self.frames_per_chunk) % 32
        tint[:, 2] += (phase / 31.0 - 0.5) * 0.12
        background = cache.first_frame.unsqueeze(0).expand(
            self.frames_per_chunk, -1, -1, -1
        )
        return (background + tint[:, :, None, None] * 0.35).clamp(-1.0, 1.0)

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: DummyCam2VCache,
    ) -> dict[str, float]:
        """Return the configured synthetic generation latency."""
        del autoregressive_index, cache
        return {"dummy_wait_s": self.step_wait_seconds}

    def close(self) -> None:
        """Release any future dummy waits."""
        self._sleep.set()


@dataclass(frozen=True, slots=True)
class DummyCam2VPipelineConfig:
    """Configuration that constructs a slow dummy Cam2V pipeline."""

    step_wait_seconds: float = 0.9
    """Synthetic wall time for one model-generation step."""

    frames_per_chunk: int = 12
    """Video frames produced by one model-generation step."""

    def setup(self) -> DummyCam2VPipeline:
        """Construct the configured dummy pipeline."""
        return DummyCam2VPipeline(
            step_wait_seconds=self.step_wait_seconds,
            frames_per_chunk=self.frames_per_chunk,
        )


def _resolve_dummy_conditioning(values: Mapping[str, Any]) -> Cam2VConditioning:
    pixel_width = int(values["pixel_width"])
    pixel_height = int(values["pixel_height"])
    focal_length = float(max(pixel_width, pixel_height))
    return Cam2VConditioning(
        prompt="dummy camera UI test",
        first_frame_path=_DUMMY_FRAME_PATH,
        base_intrinsics=torch.tensor(
            [
                focal_length,
                focal_length,
                pixel_width / 2.0,
                pixel_height / 2.0,
            ]
        ),
        world_scale=1.0,
    )


class DummyCam2VApplication(Cam2VApplication):
    """Run the shared Cam2V UI against a sleeping synthetic model."""

    def __init__(self) -> None:
        self._step_wait_seconds = 0.9
        self._frames_per_chunk = 12
        super().__init__(
            defaults=Cam2VApplicationDefaults(
                pipeline_config=DummyCam2VPipelineConfig(),
                input_resolver=_resolve_dummy_conditioning,
                total_blocks=10_000,
                pixel_width=640,
                pixel_height=360,
                device="cuda",
                fps=16,
                ui_fps=60,
                warmup_blocks=1,
                install_hint="Install the Cam2V application with runner extras.",
            )
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse dummy latency settings without loading a model."""
        super().init(commandline_args)
        self._pipeline_config = DummyCam2VPipelineConfig(
            step_wait_seconds=self._step_wait_seconds,
            frames_per_chunk=self._frames_per_chunk,
        )

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--step-wait-seconds",
            type=float,
            default=0.9,
            help="Wall time simulated by each dummy model step.",
        )
        parser.add_argument(
            "--frames-per-chunk",
            type=int,
            default=12,
            help="Frames emitted by each dummy model step.",
        )

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        self._step_wait_seconds = args.step_wait_seconds
        self._frames_per_chunk = args.frames_per_chunk

    def _validate_arguments(self, args: argparse.Namespace) -> None:
        super()._validate_arguments(args)
        if args.step_wait_seconds < 0:
            raise ValueError("--step-wait-seconds must be >= 0.")
        if args.frames_per_chunk <= 0:
            raise ValueError("--frames-per-chunk must be > 0.")
        if args.compile is not None or args.seed is not None:
            raise ValueError("The dummy Cam2V model does not use --compile or --seed.")


def create_app() -> IApplication:
    """Return the slow dummy Cam2V application."""
    return DummyCam2VApplication()


__all__ = [
    "DummyCam2VApplication",
    "DummyCam2VCache",
    "DummyCam2VPipeline",
    "DummyCam2VPipelineConfig",
    "create_app",
]
