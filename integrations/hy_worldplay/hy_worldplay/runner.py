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

"""HY-WorldPlay WAN-5B I2V runner.

Two routing modes share the same user-facing
:class:`HyWorldPlayWanI2VRunnerConfig` surface:

- **Vendor wrapper (default)** -- :class:`HyWorldPlayWanI2VRunner`
  delegates inference to upstream's :class:`wan.generate.WanRunner`
  (Wan 2.2 TI2V-5B backbone with action + camera-trajectory
  conditioning and reconstituted-context memory). Output is
  bit-for-bit identical to ``torchrun wan/generate.py`` with matching
  flags. The mandatory :attr:`RunnerConfig.pipeline` slot is filled
  with the inert :class:`_NoopPipelineConfig` from
  :mod:`hy_worldplay._vendor_pipeline`.

- **Native pipeline (preview, opt-in)** -- when
  :attr:`HyWorldPlayWanI2VRunnerConfig.use_native_pipeline` is
  ``True``, the config's ``_target`` and ``pipeline`` swap to
  :class:`HyWorldPlayWanI2VNativeRunner` and a fresh copy of
  :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B` respectively.
  Sub-PRs ship the layers incrementally: 2b.1 (I2V base case + 4-step
  Euler), 2b.2 (distilled scheduler), 2b.3 (action conditioner via
  :attr:`use_action_conditioning`), and 2b.4 / 2b.5 (camera-trajectory
  PRoPE and reconstituted-context memory) per
  ``docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md``.
"""

from __future__ import annotations

import copy
import importlib
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import RunnerConfig
from hy_worldplay._vendor_pipeline import _NoopPipelineConfig

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
    so users can map directly between the two.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanI2VRunner)

    pipeline: StreamInferencePipelineConfig = field(
        default_factory=_NoopPipelineConfig,
    )
    """Inert :class:`_NoopPipelineConfig` instance. Pinned here because
    the phase-1 wrapper drives upstream's :class:`wan.generate.WanRunner`
    directly and has no flashdreams pipeline to construct."""

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
    name; ``<root>`` and ``<root>/wan`` are added to ``sys.path``
    before the pipeline is constructed. Only used in vendor-wrapper
    mode; the native pipeline does not import the upstream tree."""

    use_native_pipeline: bool = False
    """Route inference through the in-tree
    :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B` instead of
    upstream's :class:`wan.generate.WanRunner`. Phase 2b feature flag;
    defaults to ``False`` so the phase-1 vendor wrapper stays the
    bit-stable baseline. When ``True``, ``__post_init__`` swaps
    ``_target`` to :class:`HyWorldPlayWanI2VNativeRunner` and replaces
    the inert ``pipeline`` slot with a fresh copy of
    ``PIPELINE_WAN22_TI2V_5B``. The native path supports the I2V base
    case only at 2b.1; action / camera / memory conditioning land in
    2b.3 / 2b.4 / 2b.5."""

    use_action_conditioning: bool = False
    """Enable HY-WorldPlay's discrete action conditioner (phase 2b.3).
    Only honoured when :attr:`use_native_pipeline` is ``True``; in that
    case ``__post_init__`` swaps the pipeline's I2V encoder for
    :class:`HyWorldPlayWanCtrlEncoder` and its transformer for the
    :class:`HyWorldPlayWan21Transformer` + :class:`HyWorldPlayWanDiTNetwork`
    pair so the per-AR-step :attr:`pose` is parsed into 81-class action
    labels (``trans * 9 + rotate``) and summed into the AdaLN time
    embedding. Defaults to ``False`` because the action MLP's residual
    head is only meaningful once HY-WorldPlay's distilled checkpoint
    has been loaded; with zero-init weights the conditioner is a strict
    no-op so flipping this on without those weights stays parity-safe.
    Camera-trajectory PRoPE and reconstituted-context memory still land
    in 2b.4 / 2b.5 respectively."""

    def __post_init__(self) -> None:
        """Swap ``_target`` and ``pipeline`` to the native preset when
        ``use_native_pipeline`` is set.

        Only swaps defaults: a user-supplied ``pipeline=`` override
        (any non-:class:`_NoopPipelineConfig` instance) or a
        user-supplied ``_target=`` override (any class other than
        :class:`HyWorldPlayWanI2VRunner`) is respected as-is so power
        users can plug in custom pipeline configs without round-tripping
        through this flag.

        :data:`PIPELINE_WAN22_TI2V_5B` is a module-level singleton; the
        deepcopy here keeps per-run mutations (``derive_config`` /
        per-rank seed offset, and the scheduler swap below) isolated
        to this config instance.

        After the deepcopy, the base recipe's
        :class:`FlowMatchUniPCSchedulerConfig` is replaced with
        :class:`FlowMatchEulerDiscreteSchedulerConfig` pinned to the
        distilled WAN-5B 4-step schedule (matches upstream's
        ``few_step=True`` branch). The base recipe stays neutral with
        UniPC so non-HY callers of ``PIPELINE_WAN22_TI2V_5B`` keep
        their existing scheduler.

        When :attr:`use_action_conditioning` is also set, the deep-
        copied pipeline's encoder and transformer slots are further
        swapped to the action-aware variants from :mod:`hy_worldplay._action`,
        which subclass the standard I2V encoder / Wan 2.1 transformer /
        DiT network to thread the per-AR-step action slice through to
        the AdaLN modulation path.
        """
        if not self.use_native_pipeline:
            return
        if isinstance(self.pipeline, _NoopPipelineConfig):
            from flashdreams.infra.diffusion.scheduler import (
                FlowMatchEulerDiscreteSchedulerConfig,
            )
            from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B

            self.pipeline = copy.deepcopy(PIPELINE_WAN22_TI2V_5B)
            self.pipeline.diffusion_model.scheduler = (
                FlowMatchEulerDiscreteSchedulerConfig(
                    num_inference_steps=4,
                    # Distilled WAN-5B fixed-timestep schedule. Lifted
                    # verbatim from upstream
                    # ``wan/inference/pipeline_wan_w_mem_relative_rope.py``
                    # ``few_step=True`` branch; the 4-step distilled
                    # checkpoint is trained against exactly this
                    # sigma grid.
                    fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
                )
            )

        if self.use_action_conditioning:
            self._swap_in_action_conditioning_configs()

        if self._target is HyWorldPlayWanI2VRunner:
            # Lazy import: the native runner pulls in torch, the Wan
            # pipeline, and the diffusers stack. Importing eagerly would
            # break the CPU-only smoke tests in vendor-wrapper mode.
            from hy_worldplay._native_runner import HyWorldPlayWanI2VNativeRunner

            self._target = HyWorldPlayWanI2VNativeRunner

    def _swap_in_action_conditioning_configs(self) -> None:
        """Replace the pipeline's I2V encoder + transformer with the action-aware variants.

        Lazy-imported so :mod:`hy_worldplay._action` -- which pulls in
        ``torch`` and the Wan transformer stack -- is only loaded when
        the action flag is actually set. Honours user-supplied encoder /
        transformer overrides by only swapping the stock
        :class:`WanI2VCtrlEncoderConfig` / :class:`Wan21TransformerConfig`
        instances we just deep-copied from
        :data:`PIPELINE_WAN22_TI2V_5B`.
        """
        from flashdreams.recipes.wan.autoencoder.i2v import WanI2VCtrlEncoderConfig
        from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
        from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

        from hy_worldplay._action import (
            HyWorldPlayWan21TransformerConfig,
            HyWorldPlayWanCtrlEncoderConfig,
            HyWorldPlayWanDiTNetworkConfig,
        )

        assert isinstance(self.pipeline, WanInferencePipelineConfig), (
            "_swap_in_action_conditioning_configs expected a "
            f"WanInferencePipelineConfig after the deepcopy; got {type(self.pipeline).__name__}"
        )

        if (
            isinstance(self.pipeline.encoder, WanI2VCtrlEncoderConfig)
            and type(self.pipeline.encoder) is WanI2VCtrlEncoderConfig
        ):
            self.pipeline.encoder = HyWorldPlayWanCtrlEncoderConfig(
                encoder=self.pipeline.encoder.encoder,
            )
        transformer_cfg = self.pipeline.diffusion_model.transformer
        if (
            isinstance(transformer_cfg, Wan21TransformerConfig)
            and type(transformer_cfg) is Wan21TransformerConfig
        ):
            # Mirror every Wan 2.1 knob into the HY subclass; copying
            # field-by-field keeps any future Wan21TransformerConfig
            # additions from silently disappearing on the swap.
            self.pipeline.diffusion_model.transformer = (
                HyWorldPlayWan21TransformerConfig(
                    network=HyWorldPlayWanDiTNetworkConfig(
                        patch_size=transformer_cfg.network.patch_size,
                        text_len=transformer_cfg.network.text_len,
                        in_dim=transformer_cfg.network.in_dim,
                        dim=transformer_cfg.network.dim,
                        ffn_dim=transformer_cfg.network.ffn_dim,
                        freq_dim=transformer_cfg.network.freq_dim,
                        text_dim=transformer_cfg.network.text_dim,
                        out_dim=transformer_cfg.network.out_dim,
                        num_heads=transformer_cfg.network.num_heads,
                        num_layers=transformer_cfg.network.num_layers,
                        cross_attn_norm=transformer_cfg.network.cross_attn_norm,
                        cross_attn_enable_img=(
                            transformer_cfg.network.cross_attn_enable_img
                        ),
                        eps=transformer_cfg.network.eps,
                        concat_padding_mask=(
                            transformer_cfg.network.concat_padding_mask
                        ),
                        patch_embedding_type=(
                            transformer_cfg.network.patch_embedding_type
                        ),
                        apply_rope_before_kvcache=(
                            transformer_cfg.network.apply_rope_before_kvcache
                        ),
                    ),
                    dtype=transformer_cfg.dtype,
                    checkpoint_path=transformer_cfg.checkpoint_path,
                    state_dict_transform=transformer_cfg.state_dict_transform,
                    batch_shape=transformer_cfg.batch_shape,
                    len_t=transformer_cfg.len_t,
                    guidance_scale=transformer_cfg.guidance_scale,
                    window_size_t=transformer_cfg.window_size_t,
                    sink_size_t=transformer_cfg.sink_size_t,
                    h_extrapolation_ratio=transformer_cfg.h_extrapolation_ratio,
                    w_extrapolation_ratio=transformer_cfg.w_extrapolation_ratio,
                    compile_network=transformer_cfg.compile_network,
                    use_cuda_graph=transformer_cfg.use_cuda_graph,
                    cuda_graph_warmup_iters=(
                        transformer_cfg.cuda_graph_warmup_iters
                    ),
                    stamp_image_latent=transformer_cfg.stamp_image_latent,
                    concat_image_mask_to_latent=(
                        transformer_cfg.concat_image_mask_to_latent
                    ),
                    ti2v_first_frame_per_token_timestep=(
                        transformer_cfg.ti2v_first_frame_per_token_timestep
                    ),
                )
            )


class HyWorldPlayWanI2VRunner:
    """HY-WorldPlay WAN-5B I2V driver.

    Plain class -- *not* a :class:`flashdreams.infra.runner.Runner`
    subclass -- because distributed setup is deferred to upstream's
    :class:`wan.generate.WanRunner`. The matching
    :class:`HyWorldPlayWanI2VRunnerConfig` pins its ``pipeline`` slot to
    a :class:`_NoopPipelineConfig`, so the framework contract is
    satisfied without :meth:`StreamInferencePipeline.__init__` running.
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
