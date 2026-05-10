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

"""Non-streaming Wan 2.1 runner classes (T2V and I2V).

Pure implementation module. The per-slug ``*_RUNNER`` literals live in
:mod:`flashdreams.recipes.wan.config.wan21`, alongside the matching
pipeline configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from loguru import logger

from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

WAN_VAE_SPATIAL_COMPRESSION = 8
"""Wan VAE spatial downsample factor; pixel dims must divide cleanly."""

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_I2V_IMAGE_PATH = _REPO_ROOT / "assets/example_data/i2v/image.jpg"
"""Bundled first-frame image used when ``--image-path`` is not provided."""

DEFAULT_I2V_PROMPT_PATH = _REPO_ROOT / "assets/example_data/i2v/prompt.txt"
"""Bundled prompt that matches :data:`DEFAULT_I2V_IMAGE_PATH`. The I2V
runner config defaults ``prompt_path`` here so the out-of-the-box demo
narrates the bundled first frame instead of an unrelated T2V prompt."""


@dataclass(kw_only=True)
class _Wan21RunnerConfigBase(RunnerConfig):
    """Fields shared by both Wan 2.1 runner variants."""

    prompt: str = ""
    """Text prompt. A non-empty value wins; otherwise the runner reads
    the first line of :attr:`prompt_path`. Per-variant literals override
    the empty default with a demo prompt."""

    prompt_path: Path | None = None
    """Fallback ``.txt`` whose first line is read when :attr:`prompt`
    is empty. Per-variant literals may default this to a bundled demo
    prompt (e.g. the I2V runner points at the asset that matches
    :attr:`Wan21I2VRunnerConfig.image_path`)."""

    pixel_height: int = 480
    """Output video pixel height. Must divide
    ``WAN_VAE_SPATIAL_COMPRESSION`` cleanly. 480p landscape default
    matches Wan 2.1's training resolution."""

    pixel_width: int = 832
    """Output video pixel width. Same divisibility rule as
    :attr:`pixel_height`."""

    fps: int = 16
    """Output video frame rate. Wan 2.1's training fps."""


@dataclass(kw_only=True)
class Wan21T2VRunnerConfig(_Wan21RunnerConfigBase):
    """Runner config for ``wan21-t2v-1.3b-480p``."""

    _target: type = field(default_factory=lambda: Wan21T2VRunner)


@dataclass(kw_only=True)
class Wan21I2VRunnerConfig(_Wan21RunnerConfigBase):
    """Runner config for ``wan21-i2v-14b-480p``."""

    _target: type = field(default_factory=lambda: Wan21I2VRunner)

    image_path: Path = field(default_factory=lambda: DEFAULT_I2V_IMAGE_PATH)
    """Path to the first-frame RGB image. Defaults to the bundled
    ``assets/example_data/i2v/image.jpg`` demo frame."""

    prompt_path: Path | None = field(default_factory=lambda: DEFAULT_I2V_PROMPT_PATH)
    """Defaults to the bundled prompt that matches the bundled
    :attr:`image_path` so ``flashdreams-run wan21-i2v-14b-480p`` produces
    a coherent video out of the box. ``--prompt "..."`` overrides it."""


class _Wan21RunnerBase(Runner[_Wan21RunnerConfigBase, WanInferencePipeline]):
    """Shared single-AR-step rollout body for both Wan 2.1 variants."""

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

    def _initialize_cache(self) -> Any:
        raise NotImplementedError

    def run(self) -> None:
        """Run one AR step and dump the decoded video + stats."""
        cfg = self.config
        assert cfg.pixel_height % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_height={cfg.pixel_height} must divide "
            f"{WAN_VAE_SPATIAL_COMPRESSION}."
        )
        assert cfg.pixel_width % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_width={cfg.pixel_width} must divide {WAN_VAE_SPATIAL_COMPRESSION}."
        )

        cache = self._initialize_cache()
        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        # Call ``finalize`` even on a single-AR-step rollout so the
        # ``enable_sync_and_profile`` stats path fires.
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        # Under CP every rank holds the same gathered output -- only
        # rank 0 persists.
        if not self.is_rank_zero:
            return
        generated = generated.cpu()

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        _write_video(generated, video_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats is not None:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(
                json.dumps([{"autoregressive_index": 0, **stats}], indent=2)
            )
            logger.info(f"[{cfg.runner_name}] wrote stats -> {stats_path.resolve()}")


class Wan21T2VRunner(_Wan21RunnerBase):
    """T2V variant: text-only conditioning, latent dims from pixel dims."""

    config: Wan21T2VRunnerConfig

    def _initialize_cache(self) -> Any:
        cfg = self.config
        prompt = self._resolve_prompt()
        latent_h = cfg.pixel_height // WAN_VAE_SPATIAL_COMPRESSION
        latent_w = cfg.pixel_width // WAN_VAE_SPATIAL_COMPRESSION
        return self.pipeline.initialize_cache(
            text=[prompt], height=latent_h, width=latent_w
        )


class Wan21I2VRunner(_Wan21RunnerBase):
    """I2V variant: text + first-frame image; latent dims from image pixels."""

    config: Wan21I2VRunnerConfig

    def _initialize_cache(self) -> Any:
        cfg = self.config
        assert isinstance(cfg, Wan21I2VRunnerConfig)
        prompt = self._resolve_prompt()
        image = _load_first_frame(
            cfg.image_path,
            pixel_height=cfg.pixel_height,
            pixel_width=cfg.pixel_width,
            device=torch.device(cfg.device),
        )
        return self.pipeline.initialize_cache(text=[prompt], image=image)


__all__ = [
    "DEFAULT_I2V_IMAGE_PATH",
    "DEFAULT_I2V_PROMPT_PATH",
    "WAN_VAE_SPATIAL_COMPRESSION",
    "Wan21I2VRunner",
    "Wan21I2VRunnerConfig",
    "Wan21T2VRunner",
    "Wan21T2VRunnerConfig",
]


## I/O helpers (``cv2`` / ``mediapy`` lazy-imported; live under the ``runners`` extras).


def _load_first_frame(
    path: Path, *, pixel_height: int, pixel_width: int, device: torch.device
) -> torch.Tensor:
    """Load + resize an RGB image into Wan's ``[1, 3, H, W]`` ``[-1, 1]`` tensor."""
    try:
        import cv2  # noqa: PLC0415
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Loading the I2V first-frame image needs mediapy + opencv. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = media.read_image(str(path))[..., :3]  # drop alpha if present
    arr = cv2.resize(arr, (pixel_width, pixel_height))  # cv2 takes (W, H)
    tensor = (
        torch.from_numpy(arr).to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0
    )  # [H, W, 3] in [-1, 1]
    return rearrange(tensor, "h w c -> 1 c h w")  # [T=1, C=3, H, W]


def _write_video(video: torch.Tensor, path: Path, *, fps: int) -> None:
    """Save a ``[T, C, H, W]`` ``[-1, 1]`` tensor as an MP4."""
    try:
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Writing the output video needs mediapy. Install the runner "
            "extras: pip install 'flashdreams[runners]'."
        ) from exc

    canvas = rearrange(video, "t c h w -> t h w c")
    canvas = (canvas.float().numpy() + 1.0) / 2.0
    canvas = (canvas * 255).clip(0, 255).astype("uint8")
    media.write_video(str(path), canvas, fps=fps)
