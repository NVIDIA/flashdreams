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

"""LingBot-World camera-control I2V runner classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    load_first_frame_tensor,
    runner_artifact_path,
    write_runner_stats,
    write_video_tensor,
)
from lingbot.encoder.camctrl import CamCtrlInput
from lingbot.encoder.utils import (
    get_Ks_transformed,
    preprocess_example_poses,
)
from lingbot.pipeline import (
    LingbotWorldInferencePipeline,
)

__all__ = [
    "LingbotWorldRunnerConfig",
    "LingbotWorldRunner",
]


_INTRINSICS_REFERENCE_HEIGHT = 480
"""Capture-resolution height the bundled intrinsics ``.npy`` files are
expressed in; rescaled by :func:`get_Ks_transformed` so Plücker rays
land on the right pixel centers at the runner's actual frame size."""

_INTRINSICS_REFERENCE_WIDTH = 832
"""Capture-resolution width matching :data:`_INTRINSICS_REFERENCE_HEIGHT`."""

EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples"
)
"""HTTP base URL for the canonical examples shared by all LingBot versions."""

EXAMPLE_DATA_DIR_LOCAL = default_flashdreams_cache_dir() / "example_data/lingbot_world"
"""Local cache root where downloaded example folders are stored."""

EXAMPLE_DATA_FILENAMES = (
    "image.jpg",
    "poses.npy",
    "intrinsics.npy",
    "prompt.txt",
)
"""Example assets downloaded when each file is available upstream."""

EXAMPLE_DATA_AVAILABLE_IDXS = (0, 1, 2, 3, 4, 5)
"""Supported upstream example indices currently hosted under ``examples/``."""

EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS = (0, 1, 2, 5)
"""Example indices that provide their own upstream ``prompt.txt`` file."""


def example_data_dirname(example_idx: int) -> str:
    """Format ``example_idx`` into the upstream folder naming convention."""
    assert example_idx in EXAMPLE_DATA_AVAILABLE_IDXS, (
        f"--example_idx must be one of {EXAMPLE_DATA_AVAILABLE_IDXS}."
    )
    return f"{example_idx:02d}"


def ensure_example_data_downloaded(*, is_rank_zero: bool, example_idx: int) -> Path:
    """Download bundled GitHub example files on rank 0; barrier other ranks.

    The runner calls this from :meth:`LingbotWorldRunner._fill_example_data_defaults`;
    the WebRTC server calls it from its ``main()`` so the same files
    land on disk before the server's
    ``LingbotWebRTCSessionManager._initialize_sync`` checks for them. The
    download itself is small (image + intrinsics + poses, plus a prompt
    when available), uses the public LingBot-World GitHub raw URLs, and
    is cached at :data:`EXAMPLE_DATA_DIR_LOCAL` so repeat calls are
    no-ops.
    """
    example_dirname = example_data_dirname(example_idx)
    cache_dir = EXAMPLE_DATA_DIR_LOCAL / example_dirname
    if is_rank_zero:
        for filename in EXAMPLE_DATA_FILENAMES:
            if (
                filename == "prompt.txt"
                and example_idx not in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
            ):
                continue
            download_to_cache(
                f"{EXAMPLE_DATA_BASE_URL}/{example_dirname}/{filename}",
                cache_dir=cache_dir,
                filename=filename,
            )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return cache_dir


@dataclass(kw_only=True)
class LingbotWorldRunnerConfig(RunnerConfig):
    """Runner config for every shipped LingBot-World variant."""

    _target: type["LingbotWorldRunner"] = field(
        default_factory=lambda: LingbotWorldRunner
    )

    prompt: str = ""
    """Text prompt. A non-empty value wins; otherwise the runner reads
    the first line of :attr:`prompt_path`."""

    prompt_path: Path | None = None
    """Fallback ``.txt`` whose first line is read when :attr:`prompt` is
    empty. ``--example-data True`` lazy-fills it from the bundled demo."""

    image_path: Path | None = None
    """Path to the first-frame RGB image. Required at ``run()`` time."""

    pose_path: Path | None = None
    """Path to a ``.npy`` of camera-to-world matrices, shape ``[T, 4, 4]``.
    Required at ``run()`` time."""

    intrinsic_path: Path | None = None
    """Path to a ``.npy`` of camera intrinsics, shape ``[T, 4]``.
    Required at ``run()`` time."""

    total_blocks: int = 20
    """Upper bound on the number of AR chunks to generate. The loop
    exits early once the camera stream is consumed."""

    pixel_height: int = 464
    """Output video pixel height."""

    pixel_width: int = 832
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate. Lingbot was trained at 16fps."""

    postprocess_output_layout: VideoTensorLayout | None = "tchw"
    """Pipeline output layout for streaming post-processing."""

    example_data: bool = False
    """When ``True``, lazy-download bundled GitHub example assets into
    ``$FLASHDREAMS_CACHE_DIR/example_data/lingbot_world/`` and fill ``image_path`` /
    ``pose_path`` / ``intrinsic_path`` / ``prompt_path`` from the
    bundled defaults. Use for the README demo; pass explicit paths
    for production runs."""

    example_idx: int = 0
    """Example folder index under ``.../examples/``; allowed: ``0`` through ``5``."""


class LingbotWorldRunner(
    Runner[LingbotWorldRunnerConfig, LingbotWorldInferencePipeline]
):
    """Streaming camera-control I2V driver."""

    config: LingbotWorldRunnerConfig

    def _resolve_prompt(self) -> str:
        """Pick the prompt: non-empty ``--prompt`` wins, else ``--prompt-path``."""
        cfg = self.config
        if cfg.prompt:
            return cfg.prompt
        if cfg.prompt_path is None:
            if self.is_rank_zero:
                logger.warning(
                    "LingBot prompt.txt is missing; proceeding with an empty prompt."
                )
            return ""
        text = cfg.prompt_path.read_text().splitlines()
        prompt = text[0].strip() if text else ""
        if not prompt and self.is_rank_zero:
            logger.warning(
                "LingBot prompt file {} is empty; proceeding with an empty prompt.",
                cfg.prompt_path,
            )
        return prompt

    def _fill_example_data_defaults(self) -> None:
        """Lazy-download bundled assets and fill empty path defaults in-place."""
        cfg = self.config
        example_dir = ensure_example_data_downloaded(
            is_rank_zero=self.is_rank_zero,
            example_idx=cfg.example_idx,
        )
        if cfg.image_path is None:
            cfg.image_path = example_dir / "image.jpg"
        if cfg.pose_path is None:
            cfg.pose_path = example_dir / "poses.npy"
        if cfg.intrinsic_path is None:
            cfg.intrinsic_path = example_dir / "intrinsics.npy"
        if (
            not cfg.prompt
            and cfg.prompt_path is None
            and cfg.example_idx in EXAMPLE_DATA_PROMPT_AVAILABLE_IDXS
        ):
            cfg.prompt_path = example_dir / "prompt.txt"

    def run(self) -> None:
        """Drive an AR rollout until the camera stream is exhausted."""
        cfg = self.config
        if cfg.example_data:
            self._fill_example_data_defaults()
        assert cfg.image_path is not None, (
            "LingbotWorldRunner requires --image_path (first-frame RGB image)."
        )
        assert cfg.pose_path is not None, (
            "LingbotWorldRunner requires --pose_path "
            "(.npy of [T, 4, 4] camera-to-world matrices)."
        )
        assert cfg.intrinsic_path is not None, (
            "LingbotWorldRunner requires --intrinsic_path "
            "(.npy of [T, 4] camera intrinsics)."
        )

        prompt = self._resolve_prompt()
        device = torch.device(f"cuda:{self.local_rank}")

        # Pipeline / encoder accept ``[*batch_shape, ...]`` shapes; the
        # shipped configs pin ``batch_shape=()`` so a single-rollout layout
        # is just ``[T, C, H, W]`` (image) / ``[T, 4, 4]`` (poses) /
        # ``[T, 4]`` (intrinsics).
        first_frames_t = load_first_frame_tensor(
            cfg.image_path,
            pixel_height=cfg.pixel_height,
            pixel_width=cfg.pixel_width,
            device=device,
            dtype=torch.bfloat16,
            interpolation="cubic",
            install_hint="Install the lingbot plugin: pip install flashdreams-lingbot.",
        )

        Ks = np.load(cfg.intrinsic_path)
        Ks_t = torch.from_numpy(Ks).to(device=device, dtype=torch.float32)
        # Rescale capture-resolution intrinsics to the runner's frame size.
        camera_intrinsics_t = get_Ks_transformed(
            Ks_t,
            height_org=_INTRINSICS_REFERENCE_HEIGHT,
            width_org=_INTRINSICS_REFERENCE_WIDTH,
            height_resize=cfg.pixel_height,
            width_resize=cfg.pixel_width,
            height_final=cfg.pixel_height,
            width_final=cfg.pixel_width,
        )

        c2ws = np.load(cfg.pose_path)
        c2ws, trans_normalizer = preprocess_example_poses(c2ws)
        camera_poses_t = torch.from_numpy(c2ws).to(device=device, dtype=torch.float32)
        total_camera_frames = camera_poses_t.shape[0]

        if self.is_rank_zero:
            logger.info(
                f"[{cfg.runner_name}] loaded first_frame="
                f"{tuple(first_frames_t.shape)}, camera_poses="
                f"{tuple(camera_poses_t.shape)}"
            )

        cache = self.pipeline.initialize_cache(text=[prompt], image=first_frames_t)

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        postprocess_stream = self.create_postprocess_stream(fps=cfg.fps)
        stats_history: list[dict[str, object]] = []
        start = 0
        for i in range(cfg.total_blocks):
            num_frames = self.pipeline.get_num_output_frames(i)
            end = start + num_frames
            if end > total_camera_frames:
                break
            if self.is_rank_zero:
                logger.info(
                    f"[{cfg.runner_name}] AR step {i}/{cfg.total_blocks}, "
                    f"num_frames={num_frames}, frames=[{start}, {end})"
                )
            camctrl_input = CamCtrlInput(
                intrinsics=camera_intrinsics_t[start:end],
                poses=camera_poses_t[start:end],
                world_scale=float(trans_normalizer),
            )
            video_chunk = self.pipeline.generate(
                autoregressive_index=i,
                cache=cache,
                input=camctrl_input,
            )
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            video_chunk = postprocess_stream.process(
                video_chunk, autoregressive_index=i
            )
            if postprocess_stream.collect_output and stats is not None:
                stats_history.append(
                    {
                        "autoregressive_index": i,
                        **postprocess_stream.add_process_stats(stats),
                    }
                )
            start = end

        video = postprocess_stream.finish()
        if video is None:
            return

        ensure_output_dir(cfg.output_dir)
        video_path = runner_artifact_path(cfg.output_dir, cfg.runner_name, "mp4")
        write_video_tensor(
            video,
            video_path,
            fps=cfg.fps,
            layout="tchw",
            install_hint="Install the lingbot plugin: pip install flashdreams-lingbot.",
        )
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(video.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = write_runner_stats(
                cfg.output_dir, cfg.runner_name, stats_history
            )
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )
