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

"""HY-WorldPlay WAN-5B I2V runner (phase-1 vendor wrapper).

This module provides a thin shim that adapts the upstream
HY-WorldPlay :class:`wan.generate.WanRunner` (Wan 2.2 TI2V-5B backbone
with action + camera-trajectory conditioning and reconstituted-context
memory) onto a flashdreams :class:`RunnerConfig` surface so the slug
is dispatchable via ``flashdreams-run hy-worldplay-wan-i2v-5b``.

Phase-1 goal (this module): bit-for-bit reproduction of the upstream
``wan/generate.py`` invocation, driven from a flashdreams plugin
package, so the team can iterate on top of a known-good baseline
without forking the upstream tree. The wrapped pipeline construction
(``WanRunner.__init__`` -> ``_init_models``) is delegated unchanged to
upstream, so any output the wrapper produces matches what
``torchrun wan/generate.py`` produces with the same flags. Because the
upstream pipeline does not slice into flashdreams'
:class:`StreamInferencePipeline` 3-stage interface yet, the runner
sets ``pipeline=None`` on its :class:`RunnerConfig` and owns its own
``__init__`` (the base ``Runner`` skips pipeline construction in that
case).

Phase-2 (tracked in the integration ``README.md``): refactor onto
``flashdreams.recipes.wan`` infrastructure -- expose action +
trajectory + memory hooks on ``WanInferencePipeline`` so the recipe
shares CP / KV-cache / profiler with the rest of the wan family.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flashdreams.infra.runner import RunnerConfig

__all__ = [
    "HyWorldPlayWanI2VRunner",
    "HyWorldPlayWanI2VRunnerConfig",
]


DEFAULT_PROMPT = (
    "First-person view walking around ancient Athens, with Greek "
    "architecture and marble structures"
)
"""Default text prompt mirroring HY-WorldPlay's ``wan/generate.py`` example
(``--input`` argparse default). Kept *byte-for-byte identical* to upstream
-- including no trailing period -- because the UMT5 text encoder
tokenizes a trailing ``.`` as an extra token, which shifts the
conditioning embedding and produces a small-but-deterministic drift
(~mean |delta|=5/255) vs upstream's reference output. See
``tests/parity_check/README.md`` "Parity caveats" for the diagnostic."""

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "杂乱的背景,三条腿,背景人很多,倒着走"
)
"""Default negative prompt taken verbatim from upstream
``wan/generate.py`` so output matches the reference benchmark."""


def _ensure_upstream_importable(repo_root: Path) -> None:
    """Make the cloned HY-WorldPlay tree importable.

    Upstream's ``wan/`` package imports siblings (``hyvideo``,
    ``models``, ``distributed``, ``inference``) by *bare* package name,
    so both the repo root and ``<repo_root>/wan`` must be on
    ``sys.path`` -- exactly what upstream's ``run.sh`` /
    ``wan/README.md`` does via ``PYTHONPATH``.
    """
    if not repo_root.exists():
        raise FileNotFoundError(
            f"HY-WorldPlay tree not found at {repo_root}. "
            "Set ``hy_worldplay_repo_root`` to the cloned upstream repo "
            "(or run ``tests/parity_check/run.sh`` once to clone it "
            "under the parity-check directory and pass that path)."
        )
    for p in (repo_root, repo_root / "wan"):
        sp = str(p.resolve())
        if sp not in sys.path:
            sys.path.insert(0, sp)


@dataclass(kw_only=True)
class HyWorldPlayWanI2VRunnerConfig(RunnerConfig):
    """User-facing config for the HY-WorldPlay WAN-5B I2V runner.

    Mirrors the upstream ``wan/generate.py`` argparse surface (see
    ``HY-WorldPlay/wan/generate.py`` and ``HY-WorldPlay/wan/README.md``)
    so users can map directly between the two. Inherits the standard
    runner knobs (``runner_name``, ``description``, ``output_dir``,
    ``device``, ``offset_seed_by_global_rank``) from
    :class:`RunnerConfig`; leaves ``pipeline=None`` because phase-1
    wraps upstream's ``WanRunner.predict()`` directly rather than a
    flashdreams :class:`StreamInferencePipeline` (the recipe-level
    promotion is phase 2 -- see the integration README staging plan).
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanI2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Inline text prompt or a path to a ``.txt`` file whose first
    non-empty line is read as the prompt."""

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt forwarded to upstream's pipeline; defaults to the
    upstream-bundled Chinese negative-prompt string for parity."""

    image_path: Path | None = None
    """First-frame RGB image. Required for I2V (the only mode shipped
    by upstream's WAN-5B model)."""

    pose: str = "w-16"
    """Camera trajectory. Either a pose-string (e.g. ``"w-16"`` for 16
    forward latents, or ``"w-3, right-1, d-4"``) or the path to a
    JSON file produced by upstream's
    ``hyvideo/generate_custom_trajectory.py``. Total latents must equal
    ``num_chunk * 4``."""

    num_chunk: int = 4
    """Number of autoregressive chunks to roll out. Each chunk produces
    4 latents, i.e. roughly 16 decoded frames."""

    num_frames: int = 961
    """Latent budget reserved by upstream's pipeline for the longest
    rollout; passed through to ``WanRunner.predict`` unchanged."""

    num_inference_steps: int = 50
    """Diffusion denoising steps per chunk. Upstream's distilled
    ``wan_distilled_model`` checkpoint targets 4 steps -- override
    when using non-distilled weights."""

    pixel_height: int = 704
    """Output video pixel height (default matches upstream)."""

    pixel_width: int = 1280
    """Output video pixel width (default matches upstream)."""

    fps: int = 16
    """Output video frame rate."""

    use_memory: bool = True
    """Enable HY-WorldPlay's reconstituted-context memory. Set False
    only for ablation."""

    context_window_length: int = 16
    """Number of past chunks retained by the memory module."""

    seed: int = 0
    """RNG seed. Offset by ``RANK`` automatically when running under
    torchrun if :attr:`RunnerConfig.offset_seed_by_global_rank` is set,
    so each rank draws a distinct stream while preserving deterministic
    replay per rank."""

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    """HuggingFace ID for the base Wan 2.2 backbone (VAE + scheduler +
    pipeline scaffolding)."""

    ar_model_path: Path | None = None
    """Local directory containing HY-WorldPlay's
    ``wan_transformer/`` (``config.json`` + safetensors). Required."""

    ckpt_path: Path | None = None
    """Path to HY-WorldPlay's ``wan_distilled_model/model.pt`` (or any
    compatible action-conditioned ``.pt`` checkpoint). Required."""

    hy_worldplay_repo_root: Path | None = None
    """Path to the cloned upstream
    https://github.com/Tencent-Hunyuan/HY-WorldPlay tree. Required
    because the upstream ``wan/`` package imports siblings by bare
    name -- we add ``<root>`` and ``<root>/wan`` to ``sys.path`` before
    constructing the pipeline."""


class HyWorldPlayWanI2VRunner:
    """HY-WorldPlay WAN-5B I2V driver.

    Not a :class:`flashdreams.infra.runner.Runner` subclass because the
    phase-1 wrapper owns its own distributed setup (deferred to
    upstream's ``WanRunner``) and has no flashdreams
    :class:`StreamInferencePipeline` for the base ``Runner.__init__``
    to construct. The config's ``_target`` points here so
    ``HyWorldPlayWanI2VRunnerConfig.setup()`` instantiates this class
    directly.
    """

    config: HyWorldPlayWanI2VRunnerConfig

    def __init__(self, config: HyWorldPlayWanI2VRunnerConfig) -> None:
        self.config = config

        # Validate config *before* importing any heavy optional deps
        # so the smoke-tests can exercise these branches without torch
        # or the upstream HY-WorldPlay tree installed.
        if config.ar_model_path is None or config.ckpt_path is None:
            raise ValueError(
                "Both --ar-model-path and --ckpt-path are required. "
                "See the integration README for HuggingFace download "
                "instructions (``huggingface-cli download "
                "tencent/HY-WorldPlay wan_transformer wan_distilled_model``)."
            )
        if config.hy_worldplay_repo_root is None:
            raise ValueError(
                "--hy-worldplay-repo-root must point at the cloned "
                "upstream HY-WorldPlay tree (or run "
                "``tests/parity_check/run.sh`` once to provision one)."
            )

        # Make the cloned upstream tree importable *before* the
        # ``WanRunner`` import below: that module ultimately imports
        # ``inference.helper`` / ``models.utils`` /
        # ``distributed.parallel_state`` by bare name. Surfacing a
        # missing upstream tree here gives a much clearer error than
        # the ``ImportError`` we'd otherwise hit deeper in.
        _ensure_upstream_importable(config.hy_worldplay_repo_root)

        # Heavy imports deferred so the dataclass surface (and the
        # CPU-only smoke tests in ``tests/test_smoke.py``) work without
        # torch / loguru / the upstream HY-WorldPlay tree present.
        import torch

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_rank_zero = self.rank == 0

        wan_generate = importlib.import_module("wan.generate")
        upstream_runner_cls = wan_generate.WanRunner

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        self._upstream = upstream_runner_cls(
            model_id=config.model_id,
            ckpt_path=str(config.ckpt_path),
            ar_model_path=str(config.ar_model_path),
        )

    def _resolve_prompt(self) -> str:
        value = self.config.prompt
        if isinstance(value, Path):
            lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
            assert lines, f"prompt file {value} has no non-empty lines"
            return lines[0]
        assert value, "--prompt must be a non-empty string or a path to a .txt file"
        return value

    def run(self) -> None:
        """Drive a single autoregressive rollout and persist outputs.

        Mirrors :func:`wan.generate.__main__` in upstream: builds the
        ``input_dict`` argparse-style, calls ``self._upstream.predict``,
        and writes the resulting video on rank-zero only.
        """
        config = self.config
        if config.image_path is None:
            raise ValueError(
                "HY-WorldPlay WAN-5B is I2V only -- pass "
                "``--image-path <path-to-jpg>`` to provide the first frame."
            )
        if not config.image_path.exists():
            raise FileNotFoundError(f"image_path {config.image_path} does not exist")

        prompt = self._resolve_prompt()

        seed = config.seed
        if config.offset_seed_by_global_rank and self.rank != 0:
            seed = seed + self.rank

        input_dict: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": config.negative_prompt,
            "num_frames": config.num_frames,
            "num_inference_steps": config.num_inference_steps,
            "guidance_scale": 1,
            "height": config.pixel_height,
            "width": config.pixel_width,
            "image_path": str(config.image_path),
            "use_memory": config.use_memory,
            "context_window_length": config.context_window_length,
            "seed": seed,
            "pose": config.pose,
            "num_chunk": config.num_chunk,
        }

        start_time = time.time()
        result = self._upstream.predict(input_dict)
        elapsed = time.time() - start_time

        if not self.is_rank_zero:
            return

        import numpy as np
        from diffusers.utils import export_to_video
        from loguru import logger

        video = result["video"]
        config.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = config.output_dir / f"{config.runner_name}.mp4"
        # ``export_to_video`` expects a list of per-frame ndarrays; the
        # upstream pipeline returns a single ``(T, H, W, 3)`` tensor, so
        # we split along the time axis to produce the list shape diffusers
        # iterates over with ``len()`` + index access.
        frames: list[np.ndarray] = list(np.asarray(video[0]))
        export_to_video(frames, str(out_path), fps=config.fps)
        logger.info(
            f"[{config.runner_name}] wrote video "
            f"({np.asarray(video).shape}) -> {out_path.resolve()} "
            f"in {elapsed:.2f}s"
        )
