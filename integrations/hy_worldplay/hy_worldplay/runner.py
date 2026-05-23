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

"""HY-WorldPlay WAN-5B I2V runner config and vendor-wrapper driver."""

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
"""Upstream ``wan/generate.py`` ``--input`` default. Kept byte-for-byte
identical -- including no trailing period -- so UMT5 tokenization
matches the reference output (trailing ``.`` adds an extra token and
shifts conditioning by ~5/255)."""

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,"
    "最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,"
    "画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,"
    "杂乱的背景,三条腿,背景人很多,倒着走"
)
"""Upstream ``wan/generate.py`` negative-prompt default, verbatim."""


def _ensure_upstream_importable(repo_root: Path) -> None:
    """Add the cloned HY-WorldPlay tree to ``sys.path``.

    Upstream's ``wan/`` package imports siblings (``hyvideo``,
    ``models``, ``distributed``, ``inference``) by bare name, so both
    the repo root and ``<repo_root>/wan`` are prepended -- matching
    what upstream's ``run.sh`` does via ``PYTHONPATH``.
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

    Mirrors upstream's ``wan/generate.py`` argparse surface so users
    can map between the CLIs one-to-one.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanI2VRunner)

    pipeline: StreamInferencePipelineConfig = field(
        default_factory=_NoopPipelineConfig,
    )
    """Inert stand-in when ``use_native_pipeline=False`` -- the
    vendor wrapper drives upstream's ``WanRunner`` directly and has
    no flashdreams pipeline to construct."""

    prompt: str | Path = DEFAULT_PROMPT
    """Inline text prompt, or a path to a ``.txt`` file whose first
    non-empty line is used."""

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt forwarded to the pipeline."""

    image_path: Path | None = None
    """First-frame RGB image. Required (HY-WorldPlay WAN-5B is I2V-only)."""

    pose: str = "w-16"
    """Camera trajectory as a pose-string (e.g. ``"w-16"``,
    ``"w-3, right-1, d-4"``) or the path to a JSON file produced by
    upstream's ``hyvideo/generate_custom_trajectory.py``. Total latent
    count must equal ``num_chunk * 4``."""

    num_chunk: int = 4
    """Autoregressive chunks to roll out; each chunk emits 4 latents
    (~16 decoded frames)."""

    num_frames: int = 961
    """Latent budget upstream's pipeline reserves for the longest
    rollout."""

    num_inference_steps: int = 50
    """Diffusion denoising steps per chunk. The distilled
    ``wan_distilled_model`` checkpoint targets 4; override when using
    non-distilled weights."""

    pixel_height: int = 704
    """Output video pixel height."""

    pixel_width: int = 1280
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    use_memory: bool = True
    """Enable reconstituted-context memory. Set ``False`` for ablation."""

    context_window_length: int = 16
    """Past chunks retained by the memory module."""

    seed: int = 0
    """RNG seed. Offset by ``RANK`` under torchrun when
    :attr:`RunnerConfig.offset_seed_by_global_rank` is set, so each
    rank draws a distinct deterministic stream."""

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    """HuggingFace ID for the base Wan 2.2 backbone (VAE + scheduler +
    pipeline scaffolding)."""

    ar_model_path: Path | None = None
    """Local directory containing HY-WorldPlay's ``wan_transformer/``
    (``config.json`` + safetensors). Required."""

    ckpt_path: Path | None = None
    """Path to HY-WorldPlay's ``wan_distilled_model/model.pt`` (or any
    compatible action-conditioned ``.pt``). Required."""

    hy_worldplay_repo_root: Path | None = None
    """Cloned upstream https://github.com/Tencent-Hunyuan/HY-WorldPlay
    tree. Required in vendor-wrapper mode (the upstream ``wan/``
    package imports siblings by bare name and needs to be on
    ``sys.path``); unused on the native path."""

    use_native_pipeline: bool = True
    """Route inference through the in-tree
    :data:`flashdreams.recipes.wan.PIPELINE_WAN22_TI2V_5B`.

    When ``True`` (default), ``__post_init__`` swaps ``_target`` to
    :class:`HyWorldPlayWanI2VNativeRunner` and replaces the inert
    ``pipeline`` slot with a fresh deepcopy of
    ``PIPELINE_WAN22_TI2V_5B`` (with the scheduler swapped to the
    distilled 4-step Euler grid). Action / camera / memory conditioning
    layer on via the sibling flags below.

    Set ``False`` to fall back to the vendor wrapper, which bit-matches
    upstream's ``use_kv_cache=False`` default but pulls vendor's heavy
    deps (sageattention, cloudpickle, accelerate, transformers==4.57.6)
    at runtime."""

    use_action_conditioning: bool = False
    """Route per-AR-step :attr:`pose` through HY-WorldPlay's 81-class
    discrete action conditioner (``trans * 9 + rotate``, summed into
    the AdaLN time embedding).

    Only honoured with ``use_native_pipeline=True``. The encoder and
    transformer slots swap to the HY subclasses
    (:class:`HyWorldPlayWanCtrlEncoder`,
    :class:`HyWorldPlayWan21Transformer`, :class:`HyWorldPlayWanDiTNetwork`).
    With zero-init weights the conditioner is a strict identity, so
    flipping this on without HY-WorldPlay's distilled checkpoint is
    parity-safe."""

    use_camera_conditioning: bool = False
    """Route the camera trajectory through PRoPE dual-branch
    self-attention.

    Only honoured with ``use_native_pipeline=True``. Triggers the same
    encoder / transformer / network swap as
    :attr:`use_action_conditioning` (both conditioners share the
    :class:`HyWorldPlayCtrl` payload) and flips
    :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` on so each
    block runs :class:`hy_worldplay._camera.HyWorldPlayPRoPEBlock`. The
    PRoPE branch's ``o_prope`` projection is zero-init, so flipping
    this on without the distilled checkpoint is parity-safe."""

    use_memory_selection: bool = False
    """Emit per-AR-step ``memory_frame_indices`` on the
    :class:`HyWorldPlayCtrl` payload (FOV-overlap selection over the
    bound viewmat history).

    Requires both ``use_native_pipeline`` and
    ``use_camera_conditioning`` (the FOV scorer consumes the bound
    per-rollout viewmats). When set, the encoder is armed via
    :meth:`HyWorldPlayWanCtrlEncoder.set_memory_config` and emits a
    sorted, deduplicated index list whenever
    ``current_frame_idx >= context_window_length``. The KV-prefill
    executor consumes these indices and prepends the selected K/V
    history to each block's attention."""

    memory_frames: int = 16
    """Total memory-frame budget per AR step (temporal context +
    FOV-selected). Matches upstream's
    ``select_mem_frames_wan(..., memory_frames=16)``. Only used when
    :attr:`use_memory_selection` is set."""

    temporal_context_size: int = 12
    """Recent-frames portion of the memory budget, kept unconditionally
    each AR step."""

    memory_pred_latent_size: int = 4
    """Query-clip size for the FOV-overlap scorer (matches upstream's
    ``pred_latent_size=4``)."""

    memory_fov_h_deg: float = 60.0
    """Horizontal FOV (degrees) for the selection-time overlap."""

    memory_fov_v_deg: float = 35.0
    """Vertical FOV (degrees) for the selection-time overlap."""

    memory_points_count: int = 50_000
    """Monte-Carlo sample count in the shared point cloud consumed by
    the FOV-overlap scorer."""

    memory_points_radius: float = 8.0
    """Radius of the Monte-Carlo sphere; matches upstream's
    ``generate_points_in_sphere(50_000, 8.0)``."""

    def __post_init__(self) -> None:
        """Swap ``_target`` and ``pipeline`` for the native preset.

        No-op when ``use_native_pipeline=False``. User-supplied
        ``pipeline=`` / ``_target=`` overrides are detected by identity
        (non-:class:`_NoopPipelineConfig` / non-:class:`HyWorldPlayWanI2VRunner`)
        and left untouched.

        ``PIPELINE_WAN22_TI2V_5B`` is a module-level singleton; the
        deepcopy isolates per-run mutations (scheduler swap, action /
        camera / memory swaps below) to this config instance. The base
        recipe's UniPC scheduler is replaced with the distilled
        4-step Euler grid; the base recipe keeps UniPC so non-HY
        callers are unaffected.

        Raises ``ValueError`` when ``use_memory_selection`` is set
        without ``use_camera_conditioning`` (the FOV-overlap scorer
        needs the bound viewmat history that camera conditioning
        installs).
        """
        if not self.use_native_pipeline:
            return
        if isinstance(self.pipeline, _NoopPipelineConfig):
            from flashdreams.infra.diffusion.scheduler import (
                FlowMatchEulerDiscreteSchedulerConfig,
            )
            from flashdreams.recipes.wan import PIPELINE_WAN22_TI2V_5B

            self.pipeline = copy.deepcopy(PIPELINE_WAN22_TI2V_5B)
            # Distilled WAN-5B fixed-timestep schedule (upstream's
            # ``few_step=True`` branch in
            # ``pipeline_wan_w_mem_relative_rope.py``).
            self.pipeline.diffusion_model.scheduler = (
                FlowMatchEulerDiscreteSchedulerConfig(
                    num_inference_steps=4,
                    fixed_timesteps=(1000.0, 960.0, 888.8889, 727.2728, 0.0),
                )
            )

        # Action + camera share one subclass tree (encoder + transformer
        # + network); either flag triggers the swap. Camera additionally
        # flips on the PRoPE block path.
        if self.use_action_conditioning or self.use_camera_conditioning:
            self._swap_in_action_conditioning_configs()
            if self.ckpt_path is not None:
                self._route_distilled_checkpoint()
        if self.use_camera_conditioning:
            self._enable_prope_blocks_on_network()
        if self.use_memory_selection and not self.use_camera_conditioning:
            raise ValueError(
                "use_memory_selection=True requires "
                "use_camera_conditioning=True so the per-rollout viewmats "
                "history is parsed and bound on the encoder. Set both "
                "flags together when running the native pipeline."
            )

        if self._target is HyWorldPlayWanI2VRunner:
            # Lazy import: the native runner pulls in torch + diffusers
            # + the Wan pipeline. Eager import would break the CPU-only
            # smoke tests in vendor-wrapper mode.
            from hy_worldplay._native_runner import HyWorldPlayWanI2VNativeRunner

            self._target = HyWorldPlayWanI2VNativeRunner

    def _swap_in_action_conditioning_configs(self) -> None:
        """Swap the deep-copied pipeline's I2V encoder + transformer for the HY variants.

        Idempotent: user-supplied overrides (any non-stock subclass of
        ``WanI2VCtrlEncoderConfig`` / ``Wan21TransformerConfig``) are
        left in place. Action and camera conditioning share the same
        subclass tree because the per-AR-step ``viewmats`` / ``Ks``
        slices ride on the same :class:`HyWorldPlayCtrl` payload as
        the action labels.
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
            # Mirror every Wan 2.1 knob field-by-field so additions to
            # ``Wan21TransformerConfig`` don't silently drop on the swap.
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
                    # HY-WorldPlay autoregressive WAN-5B uses 4-latent
                    # chunks (upstream's ``pred_latent_size=4``); not
                    # the base recipe's 21. Mismatched ``len_t`` gives
                    # different total frame counts and RoPE positions.
                    len_t=4,
                    # Distilled WAN-5B bakes CFG into the checkpoint
                    # and runs a single conditional forward per step;
                    # ``guidance_scale=1.0`` skips the uncond branch +
                    # combine. (Base TI2V-5B stays at 5.0 for the
                    # non-distilled model.)
                    guidance_scale=1.0,
                    # Match the rolling KV window to a single chunk.
                    window_size_t=4,
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
                    # Upstream's HY pipeline runs the first-frame
                    # context at the stabilisation sigma
                    # ``stabilization_level - 1 = 14`` (vendor
                    # ``pipeline_wan_w_mem_relative_rope.py`` lines
                    # 680, 892); the distilled checkpoint's AdaLN
                    # table at the first frame is fitted to it.
                    first_frame_timestep_value=14.0,
                )
            )

    def _enable_prope_blocks_on_network(self) -> None:
        """Flip ``use_prope_blocks`` on the (already-swapped) HY DiT config.

        Requires :meth:`_swap_in_action_conditioning_configs` to have
        run first --
        :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` only
        exists on the HY subclass. Block construction lands later in
        :meth:`HyWorldPlayWanDiTNetwork._build_block`.
        """
        from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig

        from hy_worldplay._action import HyWorldPlayWanDiTNetworkConfig

        assert isinstance(self.pipeline, WanInferencePipelineConfig), (
            "_enable_prope_blocks_on_network expected a "
            f"WanInferencePipelineConfig after the deepcopy; got "
            f"{type(self.pipeline).__name__}"
        )
        network_cfg = self.pipeline.diffusion_model.transformer.network
        assert isinstance(network_cfg, HyWorldPlayWanDiTNetworkConfig), (
            "use_camera_conditioning=True requires the network slot to be a "
            "HyWorldPlayWanDiTNetworkConfig; ensure _swap_in_action_conditioning_configs "
            f"ran before this method (got {type(network_cfg).__name__})."
        )
        network_cfg.use_prope_blocks = True

    def _route_distilled_checkpoint(self) -> None:
        """Point the (already-swapped) transformer config at the distilled ``.pt``.

        Sets ``checkpoint_path`` to ``str(self.ckpt_path)`` and swaps
        ``state_dict_transform`` to
        :func:`hy_worldplay_distilled_state_dict_transform`, which
        unwraps upstream's ``generator`` / ``_fsdp_wrapped_module.``
        envelope and adds the ``action_embedder`` / ``to_out_prope``
        rewrites on top of the base 5B remap. The distilled file is a
        superset of the base safetensors (same 3072-dim trunk +
        HY-specific keys), so one load covers both.
        """
        from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig

        from hy_worldplay._action import HyWorldPlayWan21TransformerConfig
        from hy_worldplay._checkpoint import (
            hy_worldplay_distilled_state_dict_transform,
        )

        assert isinstance(self.pipeline, WanInferencePipelineConfig), (
            "_route_distilled_checkpoint expected a "
            f"WanInferencePipelineConfig after the deepcopy; got "
            f"{type(self.pipeline).__name__}"
        )
        transformer_cfg = self.pipeline.diffusion_model.transformer
        assert isinstance(transformer_cfg, HyWorldPlayWan21TransformerConfig), (
            "_route_distilled_checkpoint requires the transformer slot "
            "to be a HyWorldPlayWan21TransformerConfig; ensure "
            "_swap_in_action_conditioning_configs ran first (got "
            f"{type(transformer_cfg).__name__})."
        )
        assert self.ckpt_path is not None, (
            "_route_distilled_checkpoint should only be invoked when "
            "ckpt_path is supplied; the gate in __post_init__ failed."
        )

        transformer_cfg.checkpoint_path = str(self.ckpt_path)
        transformer_cfg.state_dict_transform = (
            hy_worldplay_distilled_state_dict_transform
        )


class HyWorldPlayWanI2VRunner:
    """Vendor-wrapper driver: delegates inference to upstream's :class:`wan.generate.WanRunner`.

    Plain class -- not a :class:`flashdreams.infra.runner.Runner`
    subclass -- because distributed setup is deferred to
    ``WanRunner``. :class:`HyWorldPlayWanI2VRunnerConfig` pins its
    ``pipeline`` slot to a :class:`_NoopPipelineConfig` so the
    framework contract is satisfied without
    :meth:`StreamInferencePipeline.__init__` running.
    """

    config: HyWorldPlayWanI2VRunnerConfig

    def __init__(self, config: HyWorldPlayWanI2VRunnerConfig) -> None:
        self.config = config

        # Validate *before* importing any heavy optional deps so the
        # CPU-only smoke tests can exercise these branches without
        # torch or the upstream tree installed.
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

        # Put the upstream tree on ``sys.path`` *before* importing
        # ``wan.generate``; that module imports ``inference.helper`` /
        # ``models.utils`` / ``distributed.parallel_state`` by bare name.
        _ensure_upstream_importable(config.hy_worldplay_repo_root)

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

        Mirrors upstream's ``wan/generate.py`` ``__main__``: builds
        the argparse-style ``input_dict``, calls
        ``self._upstream.predict``, and writes the video on rank-zero
        only.
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
        # ``export_to_video`` iterates with ``len()`` + index access,
        # so split the upstream ``(T, H, W, 3)`` tensor into a list of
        # per-frame ndarrays.
        frames: list[np.ndarray] = list(np.asarray(video[0]))
        export_to_video(frames, str(out_path), fps=config.fps)
        logger.info(
            f"[{config.runner_name}] wrote video "
            f"({np.asarray(video).shape}) -> {out_path.resolve()} "
            f"in {elapsed:.2f}s"
        )
