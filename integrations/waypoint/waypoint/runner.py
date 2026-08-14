# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line runner for repeated-control Waypoint rollouts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.config import derive_config
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import runner_artifact_path, write_runner_stats
from flashdreams.runtime.video_output import Mp4VideoOutputTarget
from waypoint.controls import WaypointControl, load_controls_from_file
from waypoint.pipeline import WaypointInferencePipeline

__all__ = [
    "EXAMPLE_DATA_BASE_URL",
    "EXAMPLE_DATA_DIR_LOCAL",
    "WaypointRunner",
    "WaypointRunnerConfig",
    "load_seed_display_frames",
    "load_seed_pixels",
]


EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Overworldai/Biome/14343a6/seeds"
)
"""HTTP directory containing the public example seed image."""

EXAMPLE_DATA_DIR_LOCAL = default_flashdreams_cache_dir() / "example_data/waypoint"
"""User-writable cache for the downloaded example seed image."""

_EXAMPLE_IMAGE_FILENAME = "crystal_desert_blade.jpg"

_EXAMPLE_CONTROL_FILE = (
    Path(__file__).parents[3]
    / "assets"
    / "example_data"
    / "waypoint"
    / "example_controls.json"
)


def load_seed_display_frames(path: Path) -> Tensor:
    """Load the four 720p RGB frames used to establish the displayed seed state."""
    import cv2
    import imageio.v3 as iio

    if not path.is_file():
        raise FileNotFoundError(f"seed image does not exist: {path}")
    pixels = iio.imread(path)
    if pixels.ndim != 3 or pixels.shape[-1] not in (3, 4):
        raise ValueError(f"seed image must be RGB or RGBA, got {tuple(pixels.shape)}")
    pixels = cv2.resize(pixels[..., :3], (1280, 720), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).repeat(4, 1, 1, 1)


def load_seed_pixels(path: Path, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Load one image into Waypoint's four-frame native codec input domain."""
    frames = load_seed_display_frames(path)
    frames = frames.unsqueeze(0).to(device=device, dtype=dtype).div_(255)
    return F.interpolate(
        frames[0], size=(512, 1024), mode="bilinear", align_corners=False
    ).unsqueeze(0)


@dataclass(kw_only=True)
class WaypointRunnerConfig(RunnerConfig):
    """User-facing inputs for a repeated-control Waypoint rollout."""

    _target: type["WaypointRunner"] = field(default_factory=lambda: WaypointRunner)

    seed_image: Path | None = None
    """RGB image used to establish the initial world state."""

    example_data: bool = False
    """Download the example seed image and use its bundled controls."""

    controls_file: Path | None = None
    """JSON file containing one keyboard/mouse action per generated step."""

    actions: int | None = None
    """Action count, or a prefix length when :attr:`controls_file` is set."""

    buttons: tuple[int, ...] = (32,)
    """Model button-vocabulary IDs held for every generated action."""

    mouse_dx: float = 0.10
    """Horizontal mouse displacement applied to every generated action."""

    mouse_dy: float = 0.0
    """Vertical mouse displacement applied to every generated action."""

    scroll: int = 0
    """Wheel direction applied to every generated action: ``-1``, ``0``, or ``1``."""

    fps: int = 60
    """Presentation frame rate for the resulting MP4."""

    output_height: int = 720
    """Presentation height for the resulting MP4."""

    output_width: int = 1280
    """Presentation width for the resulting MP4."""

    seed: int | None = None
    """Optional noise-generator seed for deterministic rollout replay."""

    postprocess_output_layout: VideoTensorLayout = "btchw"
    """Fixed decoder output layout consumed by the runner output stream."""


class WaypointRunner(Runner[WaypointRunnerConfig, WaypointInferencePipeline]):
    """Generate a repeated-control video from a seed image."""

    config: WaypointRunnerConfig

    def __init__(self, config: WaypointRunnerConfig) -> None:
        if config.seed_image is None and not config.example_data:
            raise ValueError("pass --seed-image or --example-data")
        if config.seed is None:
            seed = torch.seed()
        else:
            seed = config.seed
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        config = derive_config(config, pipeline={"diffusion_model": {"seed": seed}})
        super().__init__(config)
        if config.seed is None:
            logger.info(f"[{config.runner_name}] generated seed {seed}")

    def run(self) -> None:
        """Prime the image-established state, then write the controlled rollout."""
        self._resolve_example_data()
        config = self.config
        seed_image = cast(Path, config.seed_image)

        seed_display_frames = load_seed_display_frames(seed_image)
        seed_pixels = load_seed_pixels(
            seed_image,
            device=self.pipeline.device,
            dtype=self.pipeline.diffusion_model.dtype,
        )
        cache = self.pipeline.initialize_cache(seed_pixels=seed_pixels)
        controls = self._resolve_controls()

        output_stream = self.create_video_output_stream(fps=config.fps)
        output_target = Mp4VideoOutputTarget(
            output_path=runner_artifact_path(
                config.output_dir, config.runner_name, "mp4"
            ),
            fps=config.fps,
            output_layout=output_stream.output_layout,
            enabled=self.is_rank_zero,
        )
        output_target.open()
        started_at = time.perf_counter()
        seed_video = seed_display_frames.unsqueeze(0).float().div_(127.5).sub_(1.0)
        output_target.write(
            output_stream.process(
                seed_video,
                autoregressive_index=0,
            )
        )
        for autoregressive_index, control in enumerate(controls, start=1):
            video = self.pipeline.generate(autoregressive_index, cache, control)
            stats = self.pipeline.finalize(autoregressive_index, cache)
            output_target.write(
                output_stream.process(
                    self._resize_for_presentation(video),
                    autoregressive_index=autoregressive_index,
                    metrics=stats,
                )
            )

        tail = output_stream.finish()
        if tail is not None:
            output_target.write(tail)
        artifacts = output_target.close()
        if not artifacts:
            return
        artifact = artifacts[0]
        output_path = Path(artifact.uri)
        logger.info(
            f"[{config.runner_name}] wrote {artifact.metadata['shape']} -> "
            f"{output_path.resolve()} in {time.perf_counter() - started_at:.2f}s"
        )
        stats_history = artifact.metadata["stats_history"]
        if stats_history:
            stats_path = write_runner_stats(
                config.output_dir, config.runner_name, list(stats_history)
            )
            logger.info(
                f"[{config.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )

    def _resolve_example_data(self) -> None:
        """Fill omitted inputs from the pinned example-data pair."""
        config = self.config
        if not config.example_data:
            return
        if config.seed_image is None:
            config.seed_image = self._fetch_example_image()
        if config.controls_file is None:
            config.controls_file = _EXAMPLE_CONTROL_FILE

    def _fetch_example_image(self) -> Path:
        """Download the pinned example seed image once on rank zero."""
        if self.is_rank_zero:
            download_to_cache(
                f"{EXAMPLE_DATA_BASE_URL}/{_EXAMPLE_IMAGE_FILENAME}",
                cache_dir=EXAMPLE_DATA_DIR_LOCAL,
                filename=_EXAMPLE_IMAGE_FILENAME,
                validator=load_seed_display_frames,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return EXAMPLE_DATA_DIR_LOCAL / _EXAMPLE_IMAGE_FILENAME

    def _resolve_controls(self) -> tuple[WaypointControl, ...]:
        """Choose a file-driven sequence or repeat one direct control event."""
        config = self.config
        if config.controls_file is not None:
            controls = load_controls_from_file(config.controls_file)
            if config.actions is None:
                return controls
            if not 1 <= config.actions <= len(controls):
                raise ValueError(
                    f"--actions must be in [1, {len(controls)}] for "
                    f"{config.controls_file}, got {config.actions}"
                )
            return controls[: config.actions]

        actions = 4 if config.actions is None else config.actions
        if actions < 1:
            raise ValueError(f"--actions must be positive, got {actions}")
        if config.scroll not in (-1, 0, 1):
            raise ValueError(f"--scroll must be -1, 0, or 1, got {config.scroll}")
        control = WaypointControl(
            buttons=frozenset(config.buttons),
            mouse_dx=config.mouse_dx,
            mouse_dy=config.mouse_dy,
            scroll_wheel=config.scroll,
        )
        return (control,) * actions

    def _resize_for_presentation(self, video: Tensor) -> Tensor:
        """Map the codec's internal 2:1 image plane into the displayed 16:9 video."""
        batch_size, frames, channels, height, width = video.shape
        if (height, width) == (self.config.output_height, self.config.output_width):
            return video
        resized = F.interpolate(
            video.reshape(batch_size * frames, channels, height, width),
            size=(self.config.output_height, self.config.output_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(
            batch_size,
            frames,
            channels,
            self.config.output_height,
            self.config.output_width,
        )
