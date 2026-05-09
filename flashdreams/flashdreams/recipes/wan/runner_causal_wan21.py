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

"""Streaming causal Wan 2.1 runners (T2V and I2V) for ``flashdreams-run``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from loguru import logger

from flashdreams.infra.decoder.base import StreamingVideoDecoder
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.wan.config.causal_wan21 import (
    CAUSAL_FORCING_CHUNKWISE_I2V,
    CAUSAL_FORCING_CHUNKWISE_T2V,
    CAUSAL_FORCING_FRAMEWISE_I2V,
    CAUSAL_FORCING_FRAMEWISE_T2V,
    SELF_FORCING_I2V,
    SELF_FORCING_LIGHTTAE_I2V,
    SELF_FORCING_LIGHTTAE_T2V,
    SELF_FORCING_T2V,
    WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

DEFAULT_PIXEL_HEIGHT = 480
"""Pixel-space rollout height. Wan VAE 8x compression -> 60-latent height."""

DEFAULT_PIXEL_WIDTH = 832
"""Pixel-space rollout width. Wan VAE 8x compression -> 104-latent width."""

_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_I2V_IMAGE_PATH = _REPO_ROOT / "assets/example_data/i2v/image.jpg"
"""Bundled first-frame image used when ``--image-path`` is not provided."""

DEFAULT_I2V_PROMPT_PATH = _REPO_ROOT / "assets/example_data/i2v/prompt.txt"
"""Bundled prompt that matches :data:`DEFAULT_I2V_IMAGE_PATH`. The I2V
runner config defaults ``prompt_path`` here so the bundled demo narrates
the bundled first frame instead of the unrelated T2V default prompt."""

DEFAULT_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of "
    "neon lights and animated city signs casting vibrant reflections. She "
    "wears a sleek black leather jacket paired with a flowing red dress and "
    "black boots, her black purse slung over her shoulder. Sunglasses "
    "perched on her nose and a bold red lipstick add to her confident, "
    "casual demeanor. The street is damp and reflective, creating a "
    "mirror-like effect that enhances the colorful lights and shadows. "
    "Pedestrians move about, adding to the lively atmosphere. The scene is "
    "captured in a dynamic medium shot with the woman walking slightly to "
    "one side, highlighting her graceful strides."
)
"""Default demo prompt so ``flashdreams-run causal-wan21-...`` produces a
sensible video out of the box without ``--prompt``."""


@dataclass(kw_only=True)
class _CausalWan21RunnerConfigBase(RunnerConfig):
    """Fields shared by both causal Wan 2.1 runner variants."""

    prompt: str = DEFAULT_PROMPT
    """Text prompt. A non-empty value wins; otherwise the runner reads
    the first line of :attr:`prompt_path`. T2V variants keep the Tokyo
    demo prompt; I2V variants null out :attr:`prompt` so the bundled
    image-matching prompt-path is used by default."""

    prompt_path: Path | None = None
    """Fallback ``.txt`` whose first line is read when :attr:`prompt` is
    empty. I2V variants default this to the bundled prompt that matches
    :attr:`CausalWan21I2VRunnerConfig.image_path`."""

    total_blocks: int = 60
    """Number of AR chunks to generate before terminating the rollout."""

    pixel_height: int = DEFAULT_PIXEL_HEIGHT
    """Output video pixel height. Must divide
    ``WAN_VAE_SPATIAL_COMPRESSION`` cleanly."""

    pixel_width: int = DEFAULT_PIXEL_WIDTH
    """Output video pixel width. Same divisibility rule as
    :attr:`pixel_height`."""

    fps: int = 16
    """Output video frame rate. Wan 2.1's training fps."""


@dataclass(kw_only=True)
class CausalWan21T2VRunnerConfig(_CausalWan21RunnerConfigBase):
    """Runner config for the causal Wan 2.1 T2V variants."""

    _target: type = field(default_factory=lambda: CausalWan21T2VRunner)


@dataclass(kw_only=True)
class CausalWan21I2VRunnerConfig(_CausalWan21RunnerConfigBase):
    """Runner config for the causal Wan 2.1 I2V variants."""

    _target: type = field(default_factory=lambda: CausalWan21I2VRunner)

    prompt: str = ""
    """Empty by default so :attr:`prompt_path` (the bundled demo prompt)
    drives generation; pass ``--prompt "..."`` to override."""

    prompt_path: Path | None = field(default_factory=lambda: DEFAULT_I2V_PROMPT_PATH)
    """Defaults to the bundled prompt that matches :attr:`image_path` so
    the out-of-the-box demo narrates the bundled first frame."""

    image_path: Path = field(default_factory=lambda: DEFAULT_I2V_IMAGE_PATH)
    """Path to the first-frame RGB image. Defaults to the bundled
    ``assets/example_data/i2v/image.jpg`` demo frame."""


class _CausalWan21RunnerBase(
    Runner[_CausalWan21RunnerConfigBase, WanInferencePipeline]
):
    """Shared streaming-rollout body for both causal Wan 2.1 variants."""

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
        """Drive the AR rollout for ``total_blocks`` chunks and write outputs."""
        cfg = self.config
        assert cfg.pixel_height % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_height={cfg.pixel_height} must divide "
            f"{WAN_VAE_SPATIAL_COMPRESSION}."
        )
        assert cfg.pixel_width % WAN_VAE_SPATIAL_COMPRESSION == 0, (
            f"pixel_width={cfg.pixel_width} must divide {WAN_VAE_SPATIAL_COMPRESSION}."
        )

        cache = self._initialize_cache()

        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        chunks: list[torch.Tensor] = []
        stats_history: list[dict[str, float]] = []
        for i in range(cfg.total_blocks):
            num_frames = self.pipeline.get_num_output_frames(i)
            if self.is_rank_zero:
                logger.info(
                    f"[{cfg.runner_name}] AR step {i}/{cfg.total_blocks}, "
                    f"num_frames={num_frames}"
                )
            video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
            stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
            if stats is not None:
                stats_history.append({"autoregressive_index": i, **stats})
            chunks.append(video_chunk.cpu())

        generated = torch.cat(chunks, dim=1)  # [B=1, T, C, H, W]
        if not self.is_rank_zero:
            return

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        canvas = rearrange(generated, "1 t c h w -> t h w c")
        _write_video(canvas, video_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video {tuple(generated.shape)} "
            f"-> {video_path.resolve()}"
        )

        if stats_history:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )


class CausalWan21T2VRunner(_CausalWan21RunnerBase):
    """Streaming causal Wan 2.1 T2V driver."""

    config: CausalWan21T2VRunnerConfig

    def _initialize_cache(self) -> Any:
        cfg = self.config
        prompt = self._resolve_prompt()
        latent_h = cfg.pixel_height // WAN_VAE_SPATIAL_COMPRESSION
        latent_w = cfg.pixel_width // WAN_VAE_SPATIAL_COMPRESSION
        return self.pipeline.initialize_cache(
            text=[prompt], image=None, height=latent_h, width=latent_w
        )


class CausalWan21I2VRunner(_CausalWan21RunnerBase):
    """Streaming causal Wan 2.1 I2V driver (mask-injection first frame)."""

    config: CausalWan21I2VRunnerConfig

    def _initialize_cache(self) -> Any:
        cfg = self.config
        assert isinstance(cfg, CausalWan21I2VRunnerConfig)
        prompt = self._resolve_prompt()
        # Align the first-frame pixel dims to the decoder's spatial
        # compression so the encoded latent matches the rollout shape.
        latent_h = cfg.pixel_height // WAN_VAE_SPATIAL_COMPRESSION
        latent_w = cfg.pixel_width // WAN_VAE_SPATIAL_COMPRESSION
        assert isinstance(self.pipeline.decoder, StreamingVideoDecoder)
        decoder_sp = self.pipeline.decoder.spatial_compression_ratio
        pixel_h = latent_h * decoder_sp
        pixel_w = latent_w * decoder_sp
        image = _load_first_frame(
            cfg.image_path,
            pixel_height=pixel_h,
            pixel_width=pixel_w,
            device=torch.device(f"cuda:{self.local_rank}"),
        )
        return self.pipeline.initialize_cache(text=[prompt], image=image)


## Per-variant runner-config literals (slug == ``recipe_name``).

_CAUSAL_WAN21_DESCRIPTIONS: dict[str, str] = {
    "causal-wan21-self-forcing-t2v": (
        "Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step)."
    ),
    "causal-wan21-self-forcing-lighttae-t2v": (
        "Self-Forcing Wan 2.1 1.3B T2V with LightTAE decoder (faster)."
    ),
    "causal-wan21-causal-forcing-chunkwise-t2v": (
        "Causal-Forcing chunkwise Wan 2.1 1.3B T2V (Wan VAE decoder)."
    ),
    "causal-wan21-causal-forcing-framewise-t2v": (
        "Causal-Forcing framewise Wan 2.1 1.3B T2V (len_t=1, Wan VAE)."
    ),
    "causal-wan21-self-forcing-i2v": (
        "Self-Forcing Wan 2.1 1.3B I2V (Wan VAE, mask-injection first frame)."
    ),
    "causal-wan21-self-forcing-lighttae-i2v": (
        "Self-Forcing Wan 2.1 1.3B I2V with LightTAE decoder."
    ),
    "causal-wan21-causal-forcing-chunkwise-i2v": (
        "Causal-Forcing chunkwise Wan 2.1 1.3B I2V (Wan VAE)."
    ),
    "causal-wan21-causal-forcing-framewise-i2v": (
        "Causal-Forcing framewise Wan 2.1 1.3B I2V (len_t=1, stamp first frame)."
    ),
}
"""Per-variant CLI descriptions, keyed by ``recipe_name``."""

_T2V_PIPELINES: tuple = (
    SELF_FORCING_T2V,
    SELF_FORCING_LIGHTTAE_T2V,
    CAUSAL_FORCING_CHUNKWISE_T2V,
    CAUSAL_FORCING_FRAMEWISE_T2V,
)
_I2V_PIPELINES: tuple = (
    SELF_FORCING_I2V,
    SELF_FORCING_LIGHTTAE_I2V,
    CAUSAL_FORCING_CHUNKWISE_I2V,
    CAUSAL_FORCING_FRAMEWISE_I2V,
)


CAUSAL_WAN21_RUNNERS: dict[str, RunnerConfig] = {
    cfg.recipe_name: CausalWan21T2VRunnerConfig(
        runner_name=cfg.recipe_name,
        description=_CAUSAL_WAN21_DESCRIPTIONS[cfg.recipe_name],
        pipeline=cfg,
    )
    for cfg in _T2V_PIPELINES
} | {
    cfg.recipe_name: CausalWan21I2VRunnerConfig(
        runner_name=cfg.recipe_name,
        description=_CAUSAL_WAN21_DESCRIPTIONS[cfg.recipe_name],
        pipeline=cfg,
    )
    for cfg in _I2V_PIPELINES
}
"""All shipped streaming causal Wan 2.1 runners, keyed by ``runner_name``."""


__all__ = [
    "CAUSAL_WAN21_RUNNERS",
    "CausalWan21I2VRunner",
    "CausalWan21I2VRunnerConfig",
    "CausalWan21T2VRunner",
    "CausalWan21T2VRunnerConfig",
]


## I/O helpers (``cv2`` / ``mediapy`` lazy-imported; live under the ``runners`` extras).


def _load_first_frame(
    path: Path, *, pixel_height: int, pixel_width: int, device: torch.device
) -> torch.Tensor:
    """Load + resize a first-frame image into ``[1, 1, 3, H, W]`` ``[-1, 1]``."""
    try:
        import cv2  # noqa: PLC0415
        import mediapy as media  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "Loading the I2V first-frame image needs mediapy + opencv. "
            "Install the runner extras: pip install 'flashdreams[runners]'."
        ) from exc

    arr = media.read_image(str(path))[..., :3]
    arr = cv2.resize(arr, (pixel_width, pixel_height))
    tensor = (
        torch.from_numpy(arr).to(device=device, dtype=torch.bfloat16) / 127.5 - 1.0
    )  # [H, W, 3]
    return rearrange(tensor, "h w c -> 1 1 c h w")  # [B=1, T=1, C=3, H, W]


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
