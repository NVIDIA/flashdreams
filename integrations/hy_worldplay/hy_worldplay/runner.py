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

"""HY-WorldPlay WAN-5B I2V runner config and driver."""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

__all__ = [
    "DEFAULT_PROMPT",
    "EXAMPLE_DATA_BASE_URL",
    "EXAMPLE_DATA_DIR_LOCAL",
    "HyWorldPlayWanI2VRunner",
    "HyWorldPlayWanI2VRunnerConfig",
    "preprocess_first_frame",
]


DEFAULT_PROMPT = (
    "First-person view walking around ancient Athens, with Greek "
    "architecture and marble structures"
)
"""Upstream ``wan/generate.py`` ``--input`` default. Kept byte-for-byte
identical -- including no trailing period -- so UMT5 tokenization
matches the reference output (trailing ``.`` adds an extra token and
shifts conditioning by ~5/255)."""

DEFAULT_POSE = "w-15"
"""Default camera trajectory: 15 forward steps (+ the identity input
frame) = 16 latents, matching the default ``num_chunk=4`` rollout. With
``--example-data`` this is swapped for upstream's sample pose JSON."""


# Repo root = ``<this file>.parents[3]`` (``flashdreams/integrations/hy_worldplay/hy_worldplay/runner.py``).
_REPO_ROOT = Path(__file__).resolve().parents[3]

EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/Tencent-Hunyuan/HY-WorldPlay/main/assets"
)
"""HTTP base URL where upstream's sample first-frame image / pose JSON live."""

EXAMPLE_DATA_DIR_LOCAL = _REPO_ROOT / "data_local/hy_worldplay"
"""Local cache root for the downloaded sample inputs (gitignored)."""

_EXAMPLE_IMAGE_FILENAME = "test.png"
"""Upstream's default ``--image_path`` fixture (704x1280)."""

_EXAMPLE_POSE_FILENAME = "test_forward_32_latents.json"
"""Upstream's sample camera trajectory (``assets/pose/``); 33 entries
(32 forward-motion latents + the identity input frame). Long enough to
drive any ``num_chunk <= 8`` rollout via the parser's prefix slice."""


def preprocess_first_frame(
    image_path: Path,
    pixel_height: int,
    pixel_width: int,
) -> Tensor:
    """Load and resize the first-frame image to ``WanI2VCtrlEncoder``'s input shape.

    Aspect-ratio policy is fit + centre-crop, matching upstream's
    ``hyvideo/utils/image.py`` so native and vendor see the same
    conditioning frame for matching pixel sizes.

    Returns:
        ``[1, 1, 3, H, W]`` float32 tensor in ``[-1, 1]``. The leading
        ``1`` is the pipeline's ``batch_shape``; the next ``1`` is the
        single-time-step dimension required by
        :meth:`WanInferencePipeline.initialize_cache`.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    target_h, target_w = pixel_height, pixel_width

    # Scale-to-fill (the longer side hits the target; the shorter side
    # overflows and is centre-cropped). Mirrors upstream's resize policy.
    scale = max(target_h / src_h, target_w / src_w)
    new_h = int(round(src_h * scale))
    new_w = int(round(src_w * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    img = img.crop((left, top, left + target_w, top + target_h))

    arr = torch.from_numpy(_pil_to_numpy(img)).float()  # [H, W, 3] in [0, 255]
    arr = arr.permute(2, 0, 1) / 127.5 - 1.0  # [3, H, W] in [-1, 1]
    return arr.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]


def _pil_to_numpy(img: object) -> object:
    """Convert a PIL image to a numpy array, importing numpy lazily.

    The lazy import keeps numpy off the module surface so CPU smoke
    sub-venvs without pillow / numpy can still import the module.
    """
    import numpy as np

    return np.asarray(img)


def _resolve_prompt(value: str | Path) -> str:
    """Read an inline prompt or the first non-empty line of a prompt file."""
    if isinstance(value, Path):
        lines = [ln.strip() for ln in value.read_text().splitlines() if ln.strip()]
        assert lines, f"prompt file {value} has no non-empty lines"
        return lines[0]
    assert value, "--prompt must be a non-empty string or a path to a .txt file"
    return value


def _write_mp4(video: Tensor, out_path: Path, *, fps: int) -> None:
    """Persist a decoded video tensor as mp4.

    Expects ``video`` shape ``[*batch, T, C, H, W]`` in ``[-1, 1]``.
    Drops the leading batch axis (size 1), converts to ``[T, H, W, C]``
    float32 in ``[0, 1]``, and hands the frame list to
    ``diffusers.utils.export_to_video``.

    Note:
        Frames must be float in ``[0, 1]``: ``export_to_video``
        internally multiplies ndarray frames by 255 before
        ``.astype(np.uint8)``, so passing uint8 ``[0, 255]`` overflows
        and produces visibly shifted RGB means (~40 units per channel
        for typical pixel values).
    """
    import numpy as np
    from diffusers.utils import export_to_video

    if video.dim() > 4:
        # Squeeze leading batch axes one at a time (asserting size 1)
        # so the error message is precise if a future batch > 1 config
        # sneaks through.
        while video.dim() > 4:
            assert video.shape[0] == 1, (
                f"_write_mp4 expects batch_size=1; got leading shape {video.shape[0]}."
            )
            video = video.squeeze(0)
    # video is now [T, C, H, W] in [-1, 1]; map to [0, 1] float32 for
    # diffusers' export_to_video contract on ndarray frames.
    arr = ((video.clamp(-1.0, 1.0) + 1.0) * 0.5).to(torch.float32)
    arr_thwc = arr.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]
    frames: list[np.ndarray] = list(arr_thwc)
    export_to_video(frames, str(out_path), fps=fps)


@dataclass(kw_only=True)
class HyWorldPlayWanI2VRunnerConfig(RunnerConfig):
    """User-facing config for the HY-WorldPlay WAN-5B I2V runner."""

    _target: type = field(default_factory=lambda: HyWorldPlayWanI2VRunner)

    prompt: str | Path = DEFAULT_PROMPT
    """Inline text prompt, or a path to a ``.txt`` file whose first
    non-empty line is used."""

    image_path: Path | None = None
    """First-frame RGB image. Required (HY-WorldPlay WAN-5B is I2V-only)
    unless :attr:`example_data` is ``True``, in which case the runner
    lazy-downloads upstream's ``assets/img/test.png`` fixture from the
    HY-WorldPlay GitHub repo at run time and uses that as the default."""

    example_data: bool = False
    """When ``True``, lazy-download upstream's bundled sample
    first-frame image into ``data_local/hy_worldplay/`` (rank-0 only,
    gitignored) and fill :attr:`image_path` from it when unset. Use
    for the README demo; pass ``--image-path`` explicitly for
    production runs."""

    pose: str = DEFAULT_POSE
    """Camera trajectory as a pose-string (e.g. ``"w-15"``,
    ``"w-3, right-1, d-4"``) or the path to a JSON file produced by
    upstream's ``hyvideo/generate_custom_trajectory.py``. The parser
    prepends an identity pose for the input frame, so ``w-N`` produces
    ``N + 1`` latents; the rollout consumes ``num_chunk * 4`` and a
    longer source is prefix-sliced. With ``--example-data`` left at the
    default, the runner swaps in upstream's sample pose JSON."""

    num_chunk: int = 4
    """Autoregressive chunks to roll out; each chunk emits 4 latents
    (~16 decoded frames)."""

    num_frames: int = 961
    """Latent budget reserved for the longest vendor-aligned rollout.
    Only consumed by the ``HY_VENDOR_NOISE_MODE=1`` diagnostic, which
    pre-draws noise at the same shape vendor's ``prepare_latents`` would
    use; non-diagnostic runs ignore this field."""

    pixel_height: int = 704
    """Output video pixel height."""

    pixel_width: int = 1280
    """Output video pixel width."""

    fps: int = 16
    """Output video frame rate."""

    context_window_length: int = 16
    """Frame-count threshold below which the FOV-overlap memory selector
    is bypassed (AR steps with fewer accumulated frames emit
    ``memory_frame_indices=None``)."""

    seed: int = 0
    """RNG seed. Offset by ``RANK`` under torchrun when
    :attr:`RunnerConfig.offset_seed_by_global_rank` is set."""

    ckpt_path: Path | None = None
    """Optional path to HY-WorldPlay's distilled
    ``wan_distilled_model/model.pt``. When set, the runner reroutes the
    transformer's ``checkpoint_path`` + ``state_dict_transform`` to load
    the distilled weights at construction time. When ``None``, the
    pipeline loads the base Wan 2.2 TI2V-5B safetensors and HY's
    conditioners stay zero-init (strict identity, parity-safe)."""

    memory_frames: int = 16
    """Total memory-frame budget per AR step (temporal context +
    FOV-selected). Matches upstream's
    ``select_mem_frames_wan(..., memory_frames=16)``."""

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


class HyWorldPlayWanI2VRunner(
    Runner[HyWorldPlayWanI2VRunnerConfig, WanInferencePipeline]
):
    """Drive :data:`PIPELINE_HY_WORLDPLAY_WAN_I2V_5B` end-to-end for the I2V case.

    Inherits the standard :class:`Runner` machinery (torchrun bootstrap,
    distributed init, per-rank seed offset, ``pipeline.setup()`` +
    ``.to(device).eval()``) and supplies a single :meth:`run` method
    that resolves the prompt and first frame, calls
    ``pipeline.initialize_cache``, drives the AR loop with ``generate``
    + ``finalize``, and writes an mp4 on rank 0.

    The fully-swapped HY encoder / transformer / DiT network (with PRoPE
    blocks) is wired statically in :mod:`hy_worldplay.config`; this
    runner binds the per-rollout payloads (action labels, viewmats +
    intrinsics, memory-selection knobs) on the encoder before the AR
    loop starts. When :attr:`HyWorldPlayWanI2VRunnerConfig.ckpt_path` is
    supplied, ``__init__`` routes the transformer's checkpoint slot at
    HY's distilled ``.pt`` and the matching state-dict transform before
    the base ``Runner.__init__`` builds the pipeline; ``None`` keeps the
    base diffusers checkpoint (HY conditioners stay zero-init identity).
    """

    config: HyWorldPlayWanI2VRunnerConfig

    def __init__(self, config: HyWorldPlayWanI2VRunnerConfig) -> None:
        """Route the distilled checkpoint into the pipeline, then defer to :class:`Runner`.

        When ``config.ckpt_path`` is set, derives a copy of the runner
        config with the transformer's ``checkpoint_path`` +
        ``state_dict_transform`` rewritten to load HY's distilled
        ``.pt`` (instead of the base Wan 2.2 TI2V-5B safetensors)
        before the base ``__init__`` builds the pipeline.
        """
        if config.ckpt_path is not None:
            from flashdreams.infra.config import derive_config
            from hy_worldplay._checkpoint import (
                hy_worldplay_distilled_state_dict_transform,
            )

            config = derive_config(
                config,
                pipeline=dict(
                    diffusion_model=dict(
                        transformer=dict(
                            checkpoint_path=str(config.ckpt_path),
                            state_dict_transform=(
                                hy_worldplay_distilled_state_dict_transform
                            ),
                        ),
                    ),
                ),
            )
        super().__init__(config)

    def run(self) -> None:
        """Roll one autoregressive sequence and persist the mp4 on rank 0."""
        import os

        from loguru import logger

        cfg = self.config
        # ``HY_DEBUG_DISABLE_CUDA_GRAPH=1`` disables the per-network
        # CUDAGraphWrapper so env-var-gated tensor dumps in
        # :mod:`_debug_dump` (file I/O + host-synchronous ``.item()``
        # calls) don't crash CUDA stream capture with
        # ``cudaErrorStreamCaptureInvalidated``.
        if os.environ.get("HY_DEBUG_DISABLE_CUDA_GRAPH", "") == "1":
            # CUDA-graph state lives on the inner ``Wan21Transformer``,
            # not on the :class:`DiffusionModel` wrapper (scheduler +
            # transformer) exposed via ``self.pipeline.diffusion_model``;
            # reach through one level so the disable actually takes
            # effect.
            diffusion_model = getattr(self.pipeline, "diffusion_model", None)
            wan_transformer = getattr(diffusion_model, "transformer", None)
            if wan_transformer is not None and hasattr(wan_transformer, "network"):
                wan_transformer._use_cuda_graph = False
                wan_transformer._network_call = wan_transformer.network
                wan_transformer._network_call_uncond = wan_transformer.network
                logger.info(
                    "HY_DEBUG_DISABLE_CUDA_GRAPH=1: bypassing the per-network "
                    "CUDAGraphWrapper for diagnostic dumps."
                )

        if cfg.example_data:
            if cfg.image_path is None:
                cfg.image_path = self._fetch_example_image()
            # Swap the synthetic default pose for upstream's sample
            # trajectory only when the user hasn't passed their own.
            if cfg.pose == DEFAULT_POSE:
                cfg.pose = str(self._fetch_example_pose())
        if cfg.image_path is None:
            raise ValueError(
                "HY-WorldPlay WAN-5B is I2V only -- pass "
                "``--image-path <path-to-jpg>`` to provide the first frame, "
                "or set ``--example-data`` to lazy-download upstream's "
                "``assets/img/test.png`` fixture."
            )
        if not cfg.image_path.exists():
            raise FileNotFoundError(f"image_path {cfg.image_path} does not exist")

        first_param = next(self.pipeline.parameters())
        device = first_param.device
        # The VAE encoder runs in the pipeline's parameter dtype (bf16 /
        # fp16 in production, fp32 in the CPU smoke); the float32 tensor
        # produced by ``preprocess_first_frame`` would fail the
        # ``F.conv3d`` dtype check in the residual VAE's first
        # ``CausalConv3d``. Cast here so the cast-once cost stays in
        # the runner rather than the per-AR-step encode path.
        image = preprocess_first_frame(
            cfg.image_path, cfg.pixel_height, cfg.pixel_width
        ).to(device=device, dtype=first_param.dtype)
        prompt = _resolve_prompt(cfg.prompt)

        cache = self.pipeline.initialize_cache(
            text=[prompt],
            image=image,
            height=None,  # derived from image
            width=None,
        )

        # The pipeline is statically HY-swapped (see
        # :mod:`hy_worldplay.config`), so the per-rollout payloads are
        # always bound: action labels for the AdaLN add, viewmats +
        # intrinsics for the PRoPE branch, and the memory-selection
        # knobs for the FOV-overlap scorer. With zero-init weights (no
        # ``ckpt_path``) each conditioner is a strict identity, so this
        # is still parity-safe against the base Wan 2.2 TI2V-5B output.
        self._bind_action_labels()
        self._bind_camera_data()
        self._bind_memory_config(device=device)

        vendor_noise_ctx = self._maybe_vendor_aligned_noise_ctx(
            device=device, dtype=first_param.dtype
        )

        chunks: list[Tensor] = []
        # Per-chunk encode/diffuse/decode/finalize timing comes from the
        # pipeline's own profiler (``enable_sync_and_profile=True`` on
        # the recipe config). Each :meth:`StreamInferencePipeline.finalize`
        # call returns the per-stage ms dict for that AR step; collect
        # them into ``stats_history`` and dump as JSON, mirroring the
        # ``integrations/omnidreams`` pattern.
        stats_history: list[dict[str, float]] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with vendor_noise_ctx:
            for ar_idx in range(cfg.num_chunk):
                chunk = self.pipeline.generate(ar_idx, cache)
                chunks.append(chunk)
                # ``finalize`` records the chunk's CUDA events + advances
                # the KV cache. Called on every chunk (incl. the last)
                # for consistent stats; the trailing KV advance is a
                # cheap one-block forward.
                stats = self.pipeline.finalize(ar_idx, cache)
                if stats is not None:
                    stats_history.append({"autoregressive_index": ar_idx, **stats})
        elapsed = time.time() - start_time

        if not self.is_rank_zero:
            return

        video = torch.cat(chunks, dim=-4)  # cat along T axis: [..., T, C, H, W]
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = cfg.output_dir / f"{cfg.runner_name}.mp4"
        _write_mp4(video, out_path, fps=cfg.fps)
        logger.info(
            f"[{cfg.runner_name}] wrote video "
            f"({tuple(video.shape)}) -> {out_path.resolve()} in {elapsed:.2f}s"
        )

        if stats_history:
            stats_path = cfg.output_dir / f"stats_{cfg.runner_name}.json"
            stats_path.write_text(json.dumps(stats_history, indent=2))
            logger.info(
                f"[{cfg.runner_name}] wrote per-AR-step stats -> {stats_path.resolve()}"
            )

    def _fetch_example_image(self) -> Path:
        """Lazy-download upstream's bundled ``assets/img/test.png`` on rank 0.

        Mirrors :func:`lingbot.runner._ensure_example_data_downloaded`'s
        rank-0-download + barrier-all-ranks pattern; the cached file is
        reused across rollouts.
        """
        cache_dir = EXAMPLE_DATA_DIR_LOCAL
        if self.is_rank_zero:
            download_to_cache(
                f"{EXAMPLE_DATA_BASE_URL}/img/{_EXAMPLE_IMAGE_FILENAME}",
                cache_dir=cache_dir,
                filename=_EXAMPLE_IMAGE_FILENAME,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return cache_dir / _EXAMPLE_IMAGE_FILENAME

    def _fetch_example_pose(self) -> Path:
        """Lazy-download upstream's sample ``assets/pose/`` trajectory on rank 0.

        Symmetric to :meth:`_fetch_example_image`. The parser prefix-
        slices this 33-entry file to the rollout's ``num_chunk * 4``
        latent budget, so it drives any ``num_chunk <= 8`` demo run.
        """
        cache_dir = EXAMPLE_DATA_DIR_LOCAL
        if self.is_rank_zero:
            download_to_cache(
                f"{EXAMPLE_DATA_BASE_URL}/pose/{_EXAMPLE_POSE_FILENAME}",
                cache_dir=cache_dir,
                filename=_EXAMPLE_POSE_FILENAME,
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return cache_dir / _EXAMPLE_POSE_FILENAME

    def _bind_action_labels(self) -> None:
        """Parse the pose string and bind per-rollout action labels on the encoder."""
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._pose import parse_pose_action_labels

        encoder, n_latents = self._resolve_encoder_and_n_latents()
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        labels = parse_pose_action_labels(self.config.pose, n_latents)
        encoder.set_action_labels(labels)

    def _bind_camera_data(self) -> None:
        """Parse the pose string and bind per-rollout viewmats + intrinsics on the encoder.

        :func:`parse_pose_data` returns both the per-latent W2C / K and
        the action labels; this method only consumes the camera tensors.
        """
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._pose import parse_pose_data

        encoder, n_latents = self._resolve_encoder_and_n_latents()
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        viewmats, Ks, _ = parse_pose_data(self.config.pose, n_latents)
        # Cast to the pipeline dtype so PRoPE math + cudnn attention
        # don't kick the network into fp64. ``parse_pose_data`` emits
        # ``[n_latents, 4, 4]`` / ``[n_latents, 3, 3]`` without a batch
        # axis; :func:`hy_worldplay._prope.prope_qkv`
        # requires ``[batch=1, cameras, 4, 4]``, so an ``unsqueeze(0)``
        # here lifts both the per-step slice and the per-rollout buffer
        # to the rank PRoPE expects.
        target_dtype = next(self.pipeline.parameters()).dtype
        encoder.set_camera_data(
            viewmats.to(dtype=target_dtype).unsqueeze(0),
            Ks.to(dtype=target_dtype).unsqueeze(0),
        )

    def _bind_memory_config(self, *, device: torch.device) -> None:
        """Arm reconstituted-context memory selection on the encoder.

        Builds the Monte-Carlo point cloud once (size + radius mirror
        upstream's ``generate_points_in_sphere`` call) and hands it
        plus the rest of the selection knobs to
        :meth:`HyWorldPlayWanCtrlEncoder.set_memory_config`; the
        encoder then computes per-AR-step ``memory_frame_indices`` on
        demand.
        """
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder
        from hy_worldplay._memory import generate_points_in_sphere

        cfg = self.config
        encoder, _ = self._resolve_encoder_and_n_latents()
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder)
        points_local = generate_points_in_sphere(
            cfg.memory_points_count,
            cfg.memory_points_radius,
            device=device,
        )
        encoder.set_memory_config(
            points_local=points_local,
            context_window_length=cfg.context_window_length,
            memory_frames=cfg.memory_frames,
            temporal_context_size=cfg.temporal_context_size,
            pred_latent_size=cfg.memory_pred_latent_size,
            fov_h_deg=cfg.memory_fov_h_deg,
            fov_v_deg=cfg.memory_fov_v_deg,
            device=device,
        )

    def _maybe_vendor_aligned_noise_ctx(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> contextlib.AbstractContextManager[Any]:
        """Build a context manager that overrides the diffusion noise per chunk.

        When ``HY_VENDOR_NOISE_MODE=1`` the runner pre-draws the full
        multi-chunk noise tensor with the same shape and seed as
        vendor's ``prepare_latents`` (a single
        ``randn([1, 48, T, H_lat, W_lat])`` over all chunks) and
        patchifies a per-AR-step slice. Inside the chunk loop the
        returned context manager monkey-patches ``torch.randn`` to
        return the pre-computed slice whenever the request matches the
        diffusion model's ``latent_shape``; all other randn calls fall
        through to the original implementation.

        Returns ``nullcontext()`` when the env var is unset, leaving
        the pipeline to draw noise from its private ``torch.Generator``.
        """
        import math
        import os
        from unittest.mock import patch as _mock_patch

        from einops import rearrange
        from loguru import logger

        from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig

        if os.environ.get("HY_VENDOR_NOISE_MODE", "") != "1":
            return contextlib.nullcontext()

        cfg = self.config
        diffusion_model = self.pipeline.diffusion_model
        transformer = diffusion_model.transformer
        transformer_cfg = transformer.config
        assert isinstance(transformer_cfg, Wan21TransformerConfig)

        len_t = transformer_cfg.len_t
        kt, kh, kw = transformer_cfg.network.patch_size
        # 16x spatial compression for the WAN-5B residual VAE.
        h_lat = cfg.pixel_height // 16
        w_lat = cfg.pixel_width // 16
        # Vendor's prepare_latents draws the full ``num_latent_frames``
        # of noise in one randn call, independent of how many chunks
        # the rollout actually consumes. Replicate the full tensor here
        # so the RNG stream lines up bit-for-bit; a smaller
        # ``randn(num_chunk * len_t, ...)`` would put each subsequent
        # channel slice at a different flat-memory offset and break
        # parity. ``num_latent_frames = (num_frames - 1) // 4 + 1``
        # reflects the WAN-5B residual VAE's 4x temporal compression.
        vendor_full_t = (cfg.num_frames - 1) // 4 + 1
        unpatched_shape = (
            1,
            transformer_cfg.network.in_dim,
            vendor_full_t,
            h_lat,
            w_lat,
        )
        target_shape = tuple(transformer.latent_shape)
        target_numel = math.prod(target_shape)
        # The patched per-chunk noise must reshape into the patchified
        # latent shape.
        per_chunk_numel = transformer_cfg.network.in_dim * len_t * h_lat * w_lat
        assert per_chunk_numel == target_numel, (
            f"vendor-aligned noise per-chunk numel ({per_chunk_numel}) "
            f"!= native latent_shape numel ({target_numel}); shapes are "
            f"unpatched={unpatched_shape}, target={target_shape}."
        )

        seed = cfg.seed
        if cfg.offset_seed_by_global_rank and self.global_rank != 0:
            seed = seed + self.global_rank

        # Draw the full noise tensor in fp32 to mirror vendor's
        # ``randn_tensor(..., dtype=torch.float32)`` then cast to the
        # diffusion model's dtype. ``torch.manual_seed`` matches
        # vendor's global-RNG seed at the top of ``predict``.
        torch.manual_seed(seed)
        big_noise_fp32 = torch.randn(
            unpatched_shape,
            dtype=torch.float32,
            device=device,
        )

        # Patchify per chunk to match the format
        # ``DiffusionModel.generate`` would otherwise draw directly via
        # ``randn(latent_shape)``. Vendor's patch embedding applies the
        # same ``... (t kt) c (h kh) (w kw) -> ... (t h w) (c kt kh kw)``
        # rearrange, so replicating it on the unpatched slice keeps
        # per-position bit values aligned between native and vendor.
        chunk_noise_queue: list[Tensor] = []
        for ar_idx in range(cfg.num_chunk):
            chunk_slice = big_noise_fp32[
                :, :, ar_idx * len_t : (ar_idx + 1) * len_t, :, :
            ]
            # Permute [B, C, T, H, W] -> [B, T, C, H, W] so the patchify
            # pattern's ``(t kt) c`` axes line up with the input order.
            chunk_slice = chunk_slice.permute(0, 2, 1, 3, 4).contiguous()
            patched = rearrange(
                chunk_slice,
                "b (t kt) c (h kh) (w kw) -> b (t h w) (c kt kh kw)",
                kt=kt,
                kh=kh,
                kw=kw,
            )
            # Drop the batch axis when the transformer's batch_shape is
            # empty; native's ``latent_shape = (L, D)`` in that case and
            # the reshape below would otherwise complain.
            if not transformer_cfg.batch_shape:
                patched = patched.squeeze(0)
            chunk_noise_queue.append(patched.to(dtype=dtype))

        orig_randn = torch.randn

        def patched_randn(*args: Any, **kwargs: Any) -> Tensor:
            shape_arg: tuple[int, ...] | None = None
            if args:
                first = args[0]
                if isinstance(first, (tuple, list, torch.Size)):
                    shape_arg = tuple(int(x) for x in first)
                elif isinstance(first, int):
                    shape_arg = tuple(int(x) for x in args)
            else:
                size = kwargs.get("size", None)
                if isinstance(size, (tuple, list, torch.Size)):
                    shape_arg = tuple(int(x) for x in size)

            if (
                shape_arg is not None
                and shape_arg == target_shape
                and chunk_noise_queue
            ):
                noise = chunk_noise_queue.pop(0)
                kwarg_device = kwargs.get("device", noise.device)
                kwarg_dtype = kwargs.get("dtype", noise.dtype)
                return noise.to(device=kwarg_device, dtype=kwarg_dtype)
            return orig_randn(*args, **kwargs)

        logger.info(
            f"HY_VENDOR_NOISE_MODE=1: pre-drew "
            f"randn({list(unpatched_shape)}) at seed={seed} and queued "
            f"{cfg.num_chunk} patchified per-chunk slices matching "
            f"latent_shape={target_shape}."
        )
        return _mock_patch.object(torch, "randn", patched_randn)

    def _resolve_encoder_and_n_latents(self) -> tuple[object, int]:
        """Return ``(encoder, n_latents)`` after asserting the static HY swap took.

        The HY encoder / transformer are wired into
        :data:`PIPELINE_HY_WORLDPLAY_WAN_I2V_5B` at module import, so a
        failure here means a caller built their own pipeline config
        without the HY swap.
        """
        from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
        from hy_worldplay._action import HyWorldPlayWanCtrlEncoder

        cfg = self.config
        encoder = self.pipeline.encoder
        assert isinstance(encoder, HyWorldPlayWanCtrlEncoder), (
            f"expected HyWorldPlayWanCtrlEncoder; got {type(encoder).__name__}. "
            "Build the runner config via hy_worldplay.config so the static "
            "HY pipeline swap is in place."
        )
        transformer_cfg = self.pipeline.diffusion_model.transformer.config
        assert isinstance(transformer_cfg, Wan21TransformerConfig), (
            f"expected Wan21TransformerConfig (or subclass) on the diffusion "
            f"model; got {type(transformer_cfg).__name__}."
        )
        n_latents = cfg.num_chunk * transformer_cfg.len_t
        return encoder, n_latents
