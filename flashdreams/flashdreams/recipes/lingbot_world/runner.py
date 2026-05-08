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

"""Lingbot-World camera-control I2V runner classes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from loguru import logger

from flashdreams.core.io.s3_sync import sync_s3_dir_to_local
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.lingbot_world.encoder.camctrl import CamCtrlInput
from flashdreams.recipes.lingbot_world.encoder.utils import (
    get_Ks_transformed,
    preprocess_example_poses,
)
from flashdreams.recipes.lingbot_world.pipeline import (
    LingbotWorldInferencePipeline,
)

_INTRINSICS_REFERENCE_HEIGHT = 480
"""Capture-resolution height the bundled intrinsics ``.npy`` files are
expressed in; rescaled by :func:`get_Ks_transformed` so Plücker rays
land on the right pixel centers at the runner's actual frame size."""

_INTRINSICS_REFERENCE_WIDTH = 832
"""Capture-resolution width matching :data:`_INTRINSICS_REFERENCE_HEIGHT`."""

_REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_DATA_DIR_S3 = "s3://flashdreams/assets/example_data/lingbot_world"
"""S3 prefix the bundled prompt + first-frame + camera arrays are pulled from."""

EXAMPLE_DATA_DIR_LOCAL = _REPO_ROOT / "assets/example_data/lingbot_world"
"""Local cache the S3 sync writes into and the runner reads from."""

S3_CREDENTIAL_PATH = _REPO_ROOT / "credentials/s3_checkpoint.secret"
"""Default S3 credentials file for the bundled example data sync."""


def _ensure_example_data_synced(*, is_rank_zero: bool) -> None:
    """Mirror the bundled S3 prefix locally on rank 0; barrier other ranks."""
    if is_rank_zero:
        assert S3_CREDENTIAL_PATH.exists(), (
            f"S3 credential file not found at {S3_CREDENTIAL_PATH}. "
            "Either populate it (see README) or unset --example-data and "
            "pass --image-path / --pose-path / --intrinsic-path explicitly."
        )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=str(S3_CREDENTIAL_PATH),
        cache_dir=str(EXAMPLE_DATA_DIR_LOCAL),
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing lingbot_world example data from S3",
    )


@dataclass(kw_only=True)
class LingbotWorldRunnerConfig(RunnerConfig):
    """Runner config for every shipped Lingbot-World variant."""

    _target: type = field(default_factory=lambda: LingbotWorldRunner)

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

    example_data: bool = False
    """When ``True``, lazy-sync the bundled S3 example assets into
    ``assets/example_data/lingbot_world/`` and fill ``image_path`` /
    ``pose_path`` / ``intrinsic_path`` / ``prompt_path`` from the
    bundled defaults. Use for the README demo; pass explicit paths
    for production runs."""


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
        assert cfg.prompt_path is not None, (
            "either --prompt or --prompt-path must be set "
            "(both empty resolved to no text input)."
        )
        text = cfg.prompt_path.read_text().splitlines()
        assert text, f"prompt file {cfg.prompt_path} is empty"
        return text[0].strip()

    def _fill_example_data_defaults(self) -> None:
        """Lazy-sync bundled assets and fill empty path defaults in-place."""
        cfg = self.config
        _ensure_example_data_synced(is_rank_zero=self.is_rank_zero)
        if cfg.image_path is None:
            cfg.image_path = EXAMPLE_DATA_DIR_LOCAL / "image.jpg"
        if cfg.pose_path is None:
            cfg.pose_path = EXAMPLE_DATA_DIR_LOCAL / "poses.npy"
        if cfg.intrinsic_path is None:
            cfg.intrinsic_path = EXAMPLE_DATA_DIR_LOCAL / "intrinsics.npy"
        if not cfg.prompt and cfg.prompt_path is None:
            cfg.prompt_path = EXAMPLE_DATA_DIR_LOCAL / "prompt.txt"

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

        first_frame = _load_first_frame(
            cfg.image_path,
            pixel_height=cfg.pixel_height,
            pixel_width=cfg.pixel_width,
            device=device,
        )
        # Lingbot's ``batch_shape`` is ``(1, 1)``: prepend ``[B=1, V=1]`` so
        # the first frame is ``[B=1, V=1, T=1, C, H, W]`` and the camera
        # stream is ``[B=1, V=1, T, ...]``. ``_preprocess_i2v_input`` expects
        # ``[*batch_shape, T, C, H, W]`` and pads along T from there.
        first_frames_t = first_frame.unsqueeze(0).unsqueeze(0)

        Ks = np.load(cfg.intrinsic_path)
        Ks_t = torch.from_numpy(Ks).to(device=device, dtype=torch.float32)
        # Rescale capture-resolution intrinsics to the runner's frame size.
        Ks_t = get_Ks_transformed(
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
        c2ws_t = torch.from_numpy(c2ws).to(device=device, dtype=torch.float32)
        camera_intrinsics_t = Ks_t.unsqueeze(0).unsqueeze(0)  # [B=1, V=1, T, 4]
        camera_poses_t = c2ws_t.unsqueeze(0).unsqueeze(0)  # [B=1, V=1, T, 4, 4]
        total_camera_frames = camera_poses_t.shape[2]

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

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
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
                intrinsics=camera_intrinsics_t[:, :, start:end],
                poses=camera_poses_t[:, :, start:end],
                world_scale=float(trans_normalizer),
            )
            video_chunk = self.pipeline.generate(
                autoregressive_index=i,
                cache=cache,
                input=camctrl_input,
            )
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())
            start = end

        video = torch.cat(chunks, dim=2)  # [B, V, T, C, H, W]
        if not self.is_rank_zero:
            return

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        # Drop B + V (both 1) and lay views out side-by-side: ``[T, H, V*W, C]``.
        canvas = rearrange(video, "1 v t c h w -> t h (v w) c")
        video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        _write_video(canvas, video_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(video.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )


__all__ = [
    "LingbotWorldRunner",
    "LingbotWorldRunnerConfig",
]


## I/O helpers (``cv2`` / ``mediapy`` lazy-imported; live under the ``runners`` extras).


def _load_first_frame(
    path: Path, *, pixel_height: int, pixel_width: int, device: torch.device
) -> torch.Tensor:
    """Load + resize a first-frame image into ``[1, C, H, W]`` ``[-1, 1]``."""
    try:
        import cv2  # noqa: PLC0415
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Loading the first-frame image needs mediapy + opencv. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = media.read_image(str(path))[..., :3]
    # Bicubic to match the upstream Lingbot World demo / generate_fast.py
    # (which uses ``F.interpolate(mode='bicubic')`` over the ``[-1, 1]``
    # tensor); bilinear here would give a different first-frame VAE latent.
    arr = cv2.resize(arr, (pixel_width, pixel_height), interpolation=cv2.INTER_CUBIC)
    tensor = (
        torch.from_numpy(arr).to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0
    )  # [H, W, 3]
    return rearrange(tensor, "h w c -> 1 c h w")  # [T=1, C, H, W]


def _write_video(canvas: torch.Tensor, path: Path, *, fps: int) -> None:
    """Save a ``[T, H, W, C]`` ``[-1, 1]`` tensor as an MP4."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Writing the output video needs mediapy. Install the runner "
            "extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = (canvas.float().numpy() + 1.0) / 2.0
    arr = (arr * 255).clip(0, 255).astype("uint8")
    media.write_video(str(path), arr, fps=fps)
