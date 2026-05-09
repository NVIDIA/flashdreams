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

"""``flashdreams-run``-side runners for the non-streaming Wan 2.1 recipes.

Two variants, two runners (per-variant runner classes):

* :class:`Wan21T2VRunner` -- prompt-only, latent ``(height, width)`` is
  derived from the runner config's ``pixel_height`` / ``pixel_width``.
* :class:`Wan21I2VRunner` -- prompt + first-frame image. The image is
  loaded, resized to ``(pixel_height, pixel_width)``, and normalised
  to ``[-1, 1]``; latent ``(height, width)`` is then derived inside
  :meth:`WanInferencePipeline.initialize_cache` from the image's pixel
  size.

Both runners write an ``.mp4`` under ``output_dir`` and a stats JSON
when per-stage profiling is on. The body is the legacy
``examples/run_wan21.py::main`` minus the T2V/I2V branch (the per-
variant subclasses each own one half).
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
from flashdreams.recipes.wan.config.wan21 import (
    WAN21_I2V_14B_480P,
    WAN21_T2V_1PT3B_480P,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

WAN_VAE_SPATIAL_COMPRESSION = 8
"""Wan VAE downsamples 8× spatially. Pixel ↔ latent conversion is
``latent_dim = pixel_dim // WAN_VAE_SPATIAL_COMPRESSION``; both runners
require the pixel dims to divide exactly so the cross-check inside
``initialize_cache`` doesn't reject the call."""


@dataclass(kw_only=True)
class _Wan21RunnerConfigBase(RunnerConfig):
    """Fields shared by both Wan 2.1 runner variants.

    Lifted into a private base so the T2V and I2V configs stay in
    sync on the I/O knobs that apply to either rollout (prompt + pixel
    dims + fps).
    """

    prompt: str = ""
    """Text prompt. Falls back to :attr:`prompt_path` when empty; one
    of the two must resolve to a non-empty string. The default ``""``
    is overridden per-variant to a sensible demo prompt so
    ``flashdreams-run wan21-t2v-1.3b-480p`` works out of the box."""

    prompt_path: Path | None = None
    """Optional path to a ``.txt`` whose first line is the prompt.
    When set, takes precedence over :attr:`prompt`."""

    pixel_height: int = 832
    """Output video pixel height. Must divide
    ``WAN_VAE_SPATIAL_COMPRESSION`` cleanly."""

    pixel_width: int = 480
    """Output video pixel width. Same divisibility rule as
    :attr:`pixel_height`."""

    fps: int = 16
    """Output video frame rate. Wan 2.1's training fps; pinning it as
    the default keeps the saved ``.mp4`` playing at the same speed
    the model was trained on."""


@dataclass(kw_only=True)
class Wan21T2VRunnerConfig(_Wan21RunnerConfigBase):
    """Runner config for ``wan21-t2v-1.3b-480p``."""

    _target: type["Wan21T2VRunner"] = field(  # type: ignore[assignment]
        default_factory=lambda: Wan21T2VRunner
    )


@dataclass(kw_only=True)
class Wan21I2VRunnerConfig(_Wan21RunnerConfigBase):
    """Runner config for ``wan21-i2v-14b-480p``."""

    _target: type["Wan21I2VRunner"] = field(  # type: ignore[assignment]
        default_factory=lambda: Wan21I2VRunner
    )

    image_path: Path | None = None
    """Path to the first-frame RGB image. Required at ``run()`` time
    (kept ``Optional`` on the dataclass so the literal config doesn't
    have to ship a placeholder ``Path``)."""


class _Wan21RunnerBase(
    Runner[_Wan21RunnerConfigBase, WanInferencePipeline]
):
    """Shared rollout body for both Wan 2.1 variants.

    Both variants run a single AR step (``window_size_t == len_t == 21``
    in the literal configs), so the body is just ``initialize_cache``
    + one ``generate`` + one ``finalize``. The two subclasses differ
    only in how they build the ``initialize_cache`` kwargs.
    """

    def _resolve_prompt(self) -> str:
        """Pick the prompt: ``--prompt_path`` wins, else ``--prompt``."""
        cfg = self.config
        if cfg.prompt_path is not None:
            text = cfg.prompt_path.read_text().splitlines()
            assert text, f"prompt file {cfg.prompt_path} is empty"
            return text[0].strip()
        assert cfg.prompt, (
            "either --prompt or --prompt_path must be set "
            "(both empty resolved to no text input)."
        )
        return cfg.prompt

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
            f"pixel_width={cfg.pixel_width} must divide "
            f"{WAN_VAE_SPATIAL_COMPRESSION}."
        )

        cache = self._initialize_cache()
        generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
        # Single-AR-step rollouts don't strictly need ``finalize``; we
        # call it anyway so the ``enable_sync_and_profile`` stats path
        # actually fires (matches the legacy example).
        stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
        # Persist only on rank 0 -- under CP every rank holds the same
        # gathered output, so a single writer is enough.
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
        assert cfg.image_path is not None, (
            "Wan21I2VRunner requires --image_path (first-frame RGB image)."
        )
        prompt = self._resolve_prompt()
        image = _load_first_frame(
            cfg.image_path,
            pixel_height=cfg.pixel_height,
            pixel_width=cfg.pixel_width,
            device=torch.device(cfg.device),
        )
        return self.pipeline.initialize_cache(text=[prompt], image=image)


## Per-variant runner config literals.
##
## ``runner_name`` mirrors the wrapped pipeline's ``recipe_name`` so
## ``flashdreams-run <recipe_name>`` "just works" -- one runner per pipeline.

WAN21_T2V_1PT3B_480P_RUNNER = Wan21T2VRunnerConfig(
    runner_name="wan21-t2v-1.3b-480p",
    pipeline=WAN21_T2V_1PT3B_480P,
    prompt=(
        "Two anthropomorphic cats in comfy boxing gear and bright gloves "
        "fight intensely on a spotlighted stage."
    ),
)
"""Wan 2.1 1.3B T2V at 480p with the legacy demo prompt baked in."""

WAN21_I2V_14B_480P_RUNNER = Wan21I2VRunnerConfig(
    runner_name="wan21-i2v-14b-480p",
    pipeline=WAN21_I2V_14B_480P,
    prompt=(
        "A stylish woman strolls down a bustling Tokyo street, the warm "
        "glow of neon lights and animated city signs casting vibrant "
        "reflections."
    ),
)
"""Wan 2.1 14B I2V at 480p with the legacy demo prompt baked in.
``--image_path`` must be provided at the CLI."""


WAN21_RUNNERS: dict[str, _Wan21RunnerConfigBase] = {
    cfg.runner_name: cfg
    for cfg in (
        WAN21_T2V_1PT3B_480P_RUNNER,
        WAN21_I2V_14B_480P_RUNNER,
    )
}
"""All shipped non-streaming Wan 2.1 runners, keyed by ``runner_name``."""


__all__ = [
    "WAN21_I2V_14B_480P_RUNNER",
    "WAN21_RUNNERS",
    "WAN21_T2V_1PT3B_480P_RUNNER",
    "WAN_VAE_SPATIAL_COMPRESSION",
    "Wan21I2VRunner",
    "Wan21I2VRunnerConfig",
    "Wan21T2VRunner",
    "Wan21T2VRunnerConfig",
]


## I/O helpers.
##
## Lazy-imported because the underlying packages (cv2, mediapy) are
## bundled under the ``runners`` install extras, not the core install
## -- a serving deployment that only uses the bare pipeline does not
## need them.


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
