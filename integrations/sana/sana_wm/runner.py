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

"""SANA-WM bidirectional runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger

from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import StreamInferencePipelineConfig
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import (
    ensure_output_dir,
    resolve_prompt_value,
    runner_artifact_path,
)
from sana_wm.camera import (
    action_string_to_c2w,
    default_intrinsics_vec4,
    fit_camera_trajectory,
    load_intrinsics,
    resize_center_crop_geometry,
    snap_num_frames,
    transform_intrinsics_for_crop,
)
from sana_wm.conditioning import (
    SanaWMI2VConditioningRequest,
    SanaWMStreamingI2VConditioningRequest,
    streaming_chunk_boundaries,
)
from sana_wm.constants import (
    DEFAULT_ACTION,
    DEFAULT_FPS,
    DEFAULT_STREAMING_DENOISING_STEP_LIST,
    DEFAULT_STREAMING_NUM_FRAMES,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_CONFIG_PATH,
    SANA_WM_MODEL_PATH,
    SANA_WM_REFINER_GEMMA_ROOT,
    SANA_WM_REFINER_ROOT,
    SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
    SANA_WM_STREAMING_CONFIG_PATH,
    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
    SANA_WM_STREAMING_MODEL_PATH,
    SANA_WM_STREAMING_REFINER_GEMMA_ROOT,
    SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES,
    SANA_WM_STREAMING_REFINER_ROOT,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)
from sana_wm.decoder import SanaWMDecodedVideo

SamplingAlgo = Literal["auto", "flow_euler_ltx"]
"""Sampling algorithms exposed by the SANA-WM runner."""

Precision = Literal["bf16", "fp8", "fp4"]
"""SANA-WM Stage-1/refiner precision modes."""

QuantBackend = Literal["auto", "torch", "torch-fp8", "torch-fp4"]
"""Low-precision linear backend for FP8/FP4 SANA-WM paths."""

ResolvedQuantBackend = Literal["torch", "torch-fp8", "torch-fp4"]
"""Concrete low-precision linear backend selected after resolving ``auto``."""


@dataclass(kw_only=True)
class SanaWMRunnerConfig(RunnerConfig):
    """Runner config for the SANA-WM bidirectional release."""

    _target: type["SanaWMRunner"] = field(default_factory=lambda: SanaWMRunner)

    device: str = "auto"
    """Torch device for SANA-WM. ``"auto"`` picks CUDA when available."""

    image_path: Path | None = None
    """Path to the first-frame RGB image. Required at ``run()`` time."""

    prompt: str = ""
    """Inline text prompt. A non-empty value wins over ``prompt_path``."""

    prompt_path: Path | None = None
    """Fallback prompt file read when ``prompt`` is empty."""

    camera_path: Path | None = None
    """Optional ``.npy`` camera-to-world trajectory shaped ``[F, 4, 4]``."""

    intrinsics_path: Path | None = None
    """Optional ``.npy`` intrinsics shaped ``[3, 3]``, ``[F, 3, 3]``,
    ``[4]``, or ``[F, 4]``. When omitted, intrinsics are derived from the
    first-frame size using ``intrinsics_hfov_deg`` with a centered principal
    point."""

    intrinsics_hfov_deg: float = 90.0
    """Horizontal field of view in degrees used to derive intrinsics when
    ``intrinsics_path`` is not provided. The public demo intrinsics correspond
    to ~90 degrees; lower values are narrower/more zoomed-in."""

    action: str | None = DEFAULT_ACTION
    """Action DSL used to derive camera motion when ``camera_path`` is not
    provided. The action is repeated or truncated to match ``num_frames``."""

    translation_speed: float = 0.025
    """Per-frame action translation speed."""

    rotation_speed_deg: float = 0.6
    """Per-frame action rotation speed in degrees."""

    num_frames: int = 161
    """Requested output frames before LTX2-VAE stride snapping."""

    fps: int = DEFAULT_FPS
    """Output video frame rate."""

    step: int = 60
    """Stage-1 DiT sampling steps."""

    cfg_scale: float = 5.0
    """Classifier-free guidance scale for Stage 1."""

    flow_shift: float | None = None
    """Optional scheduler flow-shift override."""

    sampling_algo: SamplingAlgo = "auto"
    """Stage-1 sampler. ``"auto"`` uses ``"flow_euler_ltx"``."""

    save_stage1: bool = False
    """Also decode the unrefined Stage-1 latent when the refiner is enabled."""

    negative_prompt: str = ""
    """Negative prompt used when ``cfg_scale > 1``."""

    seed: int = 42
    """Stage-1 random seed."""

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    model_path: str = SANA_WM_MODEL_PATH
    """Stage-1 checkpoint path, ``s3://`` URI, or Hugging Face file URL."""

    stage1_precision: Precision = "bf16"
    """Stage-1 DiT compute precision. ``"bf16"`` is the default; ``"fp8"``
    requires Hopper or newer; ``"fp4"`` requires Blackwell."""

    no_refiner: bool = False
    """Skip the LTX-2 refiner and decode Stage-1 latents directly."""

    refiner_precision: Precision = "bf16"
    """LTX-2 refiner compute precision. ``"bf16"`` is the default;
    ``"fp8"`` requires Hopper or newer; ``"fp4"`` requires Blackwell.
    Ignored when ``no_refiner`` is ``True``."""

    quant_backend: QuantBackend = "auto"
    """Backend for quantized linear layers. ``"auto"`` and ``"torch"`` allow
    both FP8 and FP4 Torch replacements. ``"torch-fp8"`` and ``"torch-fp4"``
    select one ``torch._scaled_mm`` replacement explicitly."""

    refiner_root: str = SANA_WM_REFINER_ROOT
    """LTX-2 refiner root path or ``hf://`` URI."""

    refiner_gemma_root: str = SANA_WM_REFINER_GEMMA_ROOT
    """Gemma text-encoder root for the LTX-2 refiner."""

    refiner_seed: int = 42
    """Refiner random seed."""

    sink_size: int = 1
    """Number of sink latent frames used by the refiner."""

    offload_vae: bool = False
    """Move the VAE to CPU between encode/decode phases."""

    offload_stage1: bool = False
    """Tear down the Stage-1 DiT after sampling to free memory for decode/
    refine. Default keeps it resident (fastest); enable on memory-constrained
    GPUs."""

    offload_refiner: bool = False
    """Build and release the refiner only around refinement."""

    offload_text_encoder: bool = False
    """Move the Stage-1 text encoder to CPU between prompt encodes."""


@dataclass(kw_only=True)
class SanaWMStreamingRunnerConfig(SanaWMRunnerConfig):
    """Runner config for the SANA-WM streaming release."""

    _target: type["SanaWMStreamingRunner"] = field(
        default_factory=lambda: SanaWMStreamingRunner
    )

    num_frames: int = DEFAULT_STREAMING_NUM_FRAMES
    """Requested streaming output frames before chunk-stride snapping."""

    step: int = len(DEFAULT_STREAMING_DENOISING_STEP_LIST) - 1
    """Stage-1 streaming denoising steps."""

    cfg_scale: float = 1.0
    """Classifier-free guidance scale for streaming Stage 1."""

    flow_shift: float | None = 8.0
    """Streaming scheduler flow-shift override."""

    config_path: str = SANA_WM_STREAMING_CONFIG_PATH
    """SANA-WM streaming config path or built-in identifier."""

    model_path: str = SANA_WM_STREAMING_MODEL_PATH
    """Streaming Stage-1 checkpoint path, ``s3://`` URI, or Hugging Face file URL."""

    causal_vae_path: str = SANA_WM_STREAMING_CAUSAL_VAE_ROOT
    """Streaming causal LTX-2 VAE root path or ``hf://`` URI."""

    refiner_root: str = SANA_WM_STREAMING_REFINER_ROOT
    """Streaming LTX-2 refiner root path or ``hf://`` URI."""

    refiner_gemma_root: str = SANA_WM_STREAMING_REFINER_GEMMA_ROOT
    """Streaming Gemma-3 text-encoder root path or ``hf://`` URI."""

    num_frame_per_block: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    """Latent frames generated per steady-state AR block."""

    num_cached_blocks: int = 2
    """Stage-1 streaming context-window hint passed to the transformer."""

    no_sink_token: bool = False
    """Disable the Stage-1 streaming sink-token context hint."""

    denoising_step_list: tuple[int, ...] = DEFAULT_STREAMING_DENOISING_STEP_LIST
    """Explicit distilled Stage-1 timestep schedule."""

    refiner_block_size: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    """Latent frames refined per streaming AR block."""

    refiner_kv_max_frames: int = SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES
    """Refiner sliding-window size in latent frames."""


class SanaWMRunner(Runner[SanaWMRunnerConfig, Any]):
    """CLI driver for SANA-WM configs."""

    config: SanaWMRunnerConfig

    def __init__(self, config: SanaWMRunnerConfig) -> None:
        self.config = config
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.global_rank = torch.distributed.get_rank()
        else:
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self.global_rank = int(os.environ.get("RANK", "0"))
        self.is_rank_zero = self.global_rank == 0

    def _resolve_prompt(self) -> str:
        """Resolve the prompt from inline text or ``prompt_path``."""
        cfg = self.config
        if cfg.prompt:
            return resolve_prompt_value(cfg.prompt)
        if cfg.prompt_path is None:
            raise ValueError("SanaWMRunner requires --prompt or --prompt-path.")
        return resolve_prompt_value(cfg.prompt_path)

    def _resolve_device(self) -> torch.device:
        """Return the device used by SANA-WM."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.local_rank}")
            return torch.device("cpu")
        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device(self.config.device)

    def _resolve_trajectory(self, *, num_frames: int) -> np.ndarray:
        """Load, fit, or roll out the camera-to-world trajectory."""
        if self.config.camera_path is not None:
            c2w = np.load(self.config.camera_path).astype(np.float32)
            if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
                raise ValueError(
                    f"--camera-path must be a [F, 4, 4] .npy; got {c2w.shape}."
                )
            if c2w.shape[0] != num_frames and self.is_rank_zero:
                logger.info(
                    "Fitting --camera-path trajectory from {} to {} frames.",
                    c2w.shape[0],
                    num_frames,
                )
            return fit_camera_trajectory(c2w, num_frames)
        if not self.config.action:
            raise ValueError("SanaWMRunner requires --camera-path or --action.")
        if self.is_rank_zero:
            logger.info(
                "No --camera-path provided; deriving a {}-frame trajectory "
                "from --action.",
                num_frames,
            )
        return action_string_to_c2w(
            self.config.action,
            translation_speed=self.config.translation_speed,
            rotation_speed_deg=self.config.rotation_speed_deg,
            num_frames=num_frames,
        )

    def run(self) -> None:
        """Run SANA-WM bidirectional inference and write outputs."""
        cfg = self.config
        if cfg.image_path is None:
            raise ValueError("SanaWMRunner requires --image-path.")

        device = self._resolve_device()
        quant_backend = _resolve_quant_backend(
            cfg.quant_backend,
            _active_quantized_precisions(
                stage1_precision=cfg.stage1_precision,
                refiner_precision=cfg.refiner_precision,
                refiner_enabled=not cfg.no_refiner,
            ),
        )
        _validate_precision_request(
            device=device,
            stage1_precision=cfg.stage1_precision,
            refiner_precision=cfg.refiner_precision,
            refiner_enabled=not cfg.no_refiner,
            quant_backend=cfg.quant_backend,
        )
        prompt = self._resolve_prompt()
        image, c2w, intrinsics_vec4, num_frames = self._prepare_inputs()
        pipeline_cfg = _pipeline_config(
            cfg,
            quant_backend=quant_backend,
        )
        pipeline = pipeline_cfg.setup().to(device).eval()
        sampling_algo = self._sampling_algo()
        if sampling_algo != "flow_euler_ltx":
            raise ValueError(
                "SANA-WM requires flow_euler_ltx for the "
                f"bidirectional runner; got {sampling_algo!r}."
            )
        cache = pipeline.initialize_cache(
            decoder_context={
                "prompt": prompt,
                "fps": cfg.fps,
                "save_stage1": cfg.save_stage1,
                "refiner_seed": cfg.refiner_seed,
                "sink_size": cfg.sink_size,
            }
        )
        with torch.inference_mode():
            decoded = pipeline.generate(
                0,
                cache,
                input=SanaWMI2VConditioningRequest(
                    image=image,
                    prompt=prompt,
                    poses_c2w=c2w,
                    intrinsics_vec4=intrinsics_vec4,
                    num_frames=num_frames,
                    fps=cfg.fps,
                    steps=cfg.step,
                    cfg_scale=cfg.cfg_scale,
                    flow_shift=cfg.flow_shift,
                    seed=cfg.seed,
                    negative_prompt=cfg.negative_prompt,
                ),
            )
        pipeline.finalize(0, cache)
        if not isinstance(decoded, SanaWMDecodedVideo):
            raise TypeError(
                "SANA-WM pipeline decoder returned "
                f"{type(decoded).__name__}, expected SanaWMDecodedVideo."
            )
        if not self.is_rank_zero:
            return
        ensure_output_dir(cfg.output_dir)
        _write_video(
            runner_artifact_path(cfg.output_dir, cfg.runner_name, "mp4"),
            decoded.video_hwc,
            cfg.fps,
        )
        if decoded.stage1_video_hwc is not None:
            _write_video(
                runner_artifact_path(
                    cfg.output_dir, f"{cfg.runner_name}_stage1", "mp4"
                ),
                decoded.stage1_video_hwc,
                cfg.fps,
            )

    def _prepare_inputs(self) -> tuple[object, np.ndarray, np.ndarray, int]:
        """Load/crop the input image and prepare c2w/intrinsics."""
        from PIL import Image

        cfg = self.config
        assert cfg.image_path is not None
        image = Image.open(cfg.image_path).convert("RGB")
        stride = self._frame_snap_stride()
        snapped = snap_num_frames(
            cfg.num_frames,
            stride=stride,
        )
        if snapped != cfg.num_frames and self.is_rank_zero:
            logger.warning(
                "SANA-WM requires num_frames = {}k+1; requested {} snapped to {}.",
                stride,
                cfg.num_frames,
                snapped,
            )
        num_frames = snapped
        c2w = self._resolve_trajectory(num_frames=num_frames)

        resized_size, crop_offset = resize_center_crop_geometry(
            image.size,
            target_h=DEFAULT_VIDEO_HEIGHT,
            target_w=DEFAULT_VIDEO_WIDTH,
        )
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        left, top = crop_offset
        cropped = resized.crop(
            (
                left,
                top,
                left + DEFAULT_VIDEO_WIDTH,
                top + DEFAULT_VIDEO_HEIGHT,
            )
        )
        if cfg.intrinsics_path is None:
            intrinsics_src = default_intrinsics_vec4(
                image.size, num_frames, hfov_deg=cfg.intrinsics_hfov_deg
            )
            if self.is_rank_zero:
                logger.info(
                    "No --intrinsics-path provided; deriving intrinsics from "
                    "image size {} at hfov={} deg (principal point centered).",
                    image.size,
                    cfg.intrinsics_hfov_deg,
                )
        else:
            intrinsics_src = load_intrinsics(cfg.intrinsics_path, num_frames)
        intrinsics_vec4 = transform_intrinsics_for_crop(
            intrinsics_src,
            image.size,
            resized_size,
            crop_offset,
        )
        return cropped, c2w, intrinsics_vec4, num_frames

    def _sampling_algo(self) -> SamplingAlgo:
        """Resolve the sampler selection."""
        if self.config.sampling_algo == "auto":
            return "flow_euler_ltx"
        return self.config.sampling_algo

    def _frame_snap_stride(self) -> int:
        """Return the required pixel-frame stride for this runner."""
        return SANA_WM_VAE_TEMPORAL_COMPRESSION


class SanaWMStreamingRunner(SanaWMRunner):
    """CLI driver for SANA-WM streaming configs."""

    config: SanaWMStreamingRunnerConfig

    def run(self) -> None:
        """Run SANA-WM streaming inference and write outputs."""
        cfg = self.config
        if cfg.image_path is None:
            raise ValueError("SanaWMStreamingRunner requires --image-path.")

        device = self._resolve_device()
        quant_backend = _resolve_quant_backend(
            cfg.quant_backend,
            _active_quantized_precisions(
                stage1_precision=cfg.stage1_precision,
                refiner_precision=cfg.refiner_precision,
                refiner_enabled=not cfg.no_refiner,
            ),
        )
        _validate_precision_request(
            device=device,
            stage1_precision=cfg.stage1_precision,
            refiner_precision=cfg.refiner_precision,
            refiner_enabled=not cfg.no_refiner,
            quant_backend=cfg.quant_backend,
        )
        prompt = self._resolve_prompt()
        image, c2w, intrinsics_vec4, num_frames = self._prepare_inputs()
        pipeline_cfg = _streaming_pipeline_config(
            cfg,
            quant_backend=quant_backend,
        )
        pipeline = pipeline_cfg.setup().to(device).eval()
        latent_frames = (num_frames - 1) // SANA_WM_VAE_TEMPORAL_COMPRESSION + 1
        chunk_boundaries = streaming_chunk_boundaries(
            latent_frames,
            cfg.num_frame_per_block,
        )
        cache = pipeline.initialize_cache(
            decoder_context={
                "prompt": prompt,
                "fps": cfg.fps,
                "save_stage1": cfg.save_stage1,
                "refiner_seed": cfg.refiner_seed,
                "sink_size": cfg.sink_size,
                "block_size": cfg.refiner_block_size,
                "refiner_kv_max_frames": cfg.refiner_kv_max_frames,
            }
        )
        request = SanaWMStreamingI2VConditioningRequest(
            image=image,
            prompt=prompt,
            poses_c2w=c2w,
            intrinsics_vec4=intrinsics_vec4,
            num_frames=num_frames,
            fps=cfg.fps,
            steps=cfg.step,
            cfg_scale=cfg.cfg_scale,
            flow_shift=cfg.flow_shift,
            seed=cfg.seed,
            negative_prompt=cfg.negative_prompt,
            num_frame_per_block=cfg.num_frame_per_block,
        )

        decoded_chunks: list[np.ndarray] = []
        stage1_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for ar_idx in range(len(chunk_boundaries) - 1):
                decoded = pipeline.generate(ar_idx, cache, input=request)
                pipeline.finalize(ar_idx, cache)
                if not isinstance(decoded, SanaWMDecodedVideo):
                    raise TypeError(
                        "SANA-WM streaming pipeline decoder returned "
                        f"{type(decoded).__name__}, expected SanaWMDecodedVideo."
                    )
                if decoded.video_hwc.size:
                    decoded_chunks.append(decoded.video_hwc)
                if (
                    decoded.stage1_video_hwc is not None
                    and decoded.stage1_video_hwc.size
                ):
                    stage1_chunks.append(decoded.stage1_video_hwc)

        if not self.is_rank_zero:
            return
        ensure_output_dir(cfg.output_dir)
        if not decoded_chunks:
            raise RuntimeError("SANA-WM streaming produced no decoded frames.")
        video_hwc = np.concatenate(decoded_chunks, axis=0)
        _write_video(
            runner_artifact_path(cfg.output_dir, cfg.runner_name, "mp4"),
            video_hwc,
            cfg.fps,
        )
        if stage1_chunks:
            _write_video(
                runner_artifact_path(
                    cfg.output_dir, f"{cfg.runner_name}_stage1", "mp4"
                ),
                np.concatenate(stage1_chunks, axis=0),
                cfg.fps,
            )

    def _frame_snap_stride(self) -> int:
        """Return the streaming pixel-frame chunk stride."""
        return SANA_WM_VAE_TEMPORAL_COMPRESSION * self.config.num_frame_per_block


def _pipeline_config(
    cfg: SanaWMRunnerConfig,
    *,
    quant_backend: ResolvedQuantBackend,
) -> StreamInferencePipelineConfig:
    """Apply CLI runtime fields to the SANA-WM pipeline literal."""
    scheduler_updates: dict[str, object] = {"num_inference_steps": cfg.step}
    if cfg.flow_shift is not None:
        scheduler_updates["shift"] = cfg.flow_shift
    return derive_config(
        cfg.pipeline,
        diffusion_model=dict(
            seed=cfg.seed,
            scheduler=scheduler_updates,
            transformer=dict(
                config_path=cfg.config_path,
                checkpoint_path=cfg.model_path,
                stage1_precision=cfg.stage1_precision,
                quant_backend=quant_backend,
                offload_stage1=cfg.offload_stage1,
            ),
        ),
        encoder=dict(
            config_path=cfg.config_path,
            text_encoder=dict(
                config_path=cfg.config_path,
                stage1_precision=cfg.stage1_precision,
                quant_backend=quant_backend,
                offload_text_encoder=cfg.offload_text_encoder,
            ),
            first_frame_encoder=dict(
                config_path=cfg.config_path,
                offload_vae=cfg.offload_vae,
            ),
            camera_encoder=dict(
                height=DEFAULT_VIDEO_HEIGHT,
                width=DEFAULT_VIDEO_WIDTH,
            ),
            height=DEFAULT_VIDEO_HEIGHT,
            width=DEFAULT_VIDEO_WIDTH,
        ),
        decoder=dict(
            vae_decoder=dict(
                config_path=cfg.config_path,
                offload_vae=cfg.offload_vae,
            ),
            refiner=(
                None
                if cfg.no_refiner
                else dict(
                    refiner_root=cfg.refiner_root,
                    refiner_gemma_root=cfg.refiner_gemma_root,
                    refiner_precision=cfg.refiner_precision,
                    quant_backend=quant_backend,
                    offload_refiner=cfg.offload_refiner,
                )
            ),
        ),
    )


def _streaming_pipeline_config(
    cfg: SanaWMStreamingRunnerConfig,
    *,
    quant_backend: ResolvedQuantBackend,
) -> StreamInferencePipelineConfig:
    """Apply CLI runtime fields to the SANA-WM streaming pipeline literal."""
    return derive_config(
        cfg.pipeline,
        diffusion_model=dict(
            seed=cfg.seed,
            scheduler=dict(
                num_inference_steps=cfg.step,
                shift=cfg.flow_shift if cfg.flow_shift is not None else 8.0,
                denoising_step_list=cfg.denoising_step_list,
            ),
            transformer=dict(
                config_path=cfg.config_path,
                checkpoint_path=cfg.model_path,
                stage1_precision=cfg.stage1_precision,
                quant_backend=quant_backend,
                offload_stage1=cfg.offload_stage1,
                num_frame_per_block=cfg.num_frame_per_block,
                num_cached_blocks=cfg.num_cached_blocks,
                sink_token=not cfg.no_sink_token,
            ),
        ),
        encoder=dict(
            config_path=cfg.config_path,
            text_encoder=dict(
                config_path=cfg.config_path,
                stage1_precision=cfg.stage1_precision,
                quant_backend=quant_backend,
                offload_text_encoder=cfg.offload_text_encoder,
            ),
            first_frame_encoder=dict(
                config_path=cfg.config_path,
                offload_vae=cfg.offload_vae,
            ),
            camera_encoder=dict(
                height=DEFAULT_VIDEO_HEIGHT,
                width=DEFAULT_VIDEO_WIDTH,
            ),
            height=DEFAULT_VIDEO_HEIGHT,
            width=DEFAULT_VIDEO_WIDTH,
        ),
        decoder=dict(
            vae_decoder=dict(
                config_path=cfg.config_path,
                vae_path=cfg.causal_vae_path,
                offload_vae=cfg.offload_vae,
            ),
            refiner=(
                None
                if cfg.no_refiner
                else dict(
                    refiner_root=cfg.refiner_root,
                    refiner_gemma_root=cfg.refiner_gemma_root,
                    refiner_precision=cfg.refiner_precision,
                    quant_backend=quant_backend,
                    offload_refiner=cfg.offload_refiner,
                    kv_max_frames=cfg.refiner_kv_max_frames,
                    block_size=cfg.refiner_block_size,
                )
            ),
        ),
    )


def _write_video(path: Path, video_hwc: np.ndarray, fps: int) -> Path:
    """Write an HWC uint8 video to ``path``."""
    import imageio.v3 as iio

    iio.imwrite(path, video_hwc, fps=fps)
    logger.info("Saved {}", path)
    return path


def _validate_precision_request(
    *,
    device: torch.device,
    stage1_precision: Precision,
    refiner_precision: Precision,
    refiner_enabled: bool,
    quant_backend: QuantBackend,
) -> None:
    """Fail early when requested quantized precision cannot run."""
    quantized = _active_quantized_precisions(
        stage1_precision=stage1_precision,
        refiner_precision=refiner_precision,
        refiner_enabled=refiner_enabled,
    )
    if not quantized:
        return

    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(
            "SANA-WM fp8/fp4 precision requires a CUDA device; "
            f"resolved device is {device}."
        )

    capability = torch.cuda.get_device_capability(device)
    major, minor = capability
    sm_name = f"sm_{major}{minor}"
    if "fp8" in quantized and major < 9:
        raise ValueError(
            "SANA-WM fp8 precision requires a Hopper or newer GPU "
            f"(sm_90+); detected {sm_name}."
        )
    if "fp4" in quantized and major < 10:
        raise ValueError(
            "SANA-WM fp4/NVFP4 precision requires a Blackwell GPU "
            f"(sm_100+); detected {sm_name}. Use bf16 or fp8 on this GPU."
        )

    resolved_backend = _resolve_quant_backend(quant_backend, quantized)
    if resolved_backend == "torch-fp8":
        _validate_torch_fp8_backend(quantized)
    elif resolved_backend == "torch-fp4":
        _validate_torch_fp4_backend(quantized)
    elif resolved_backend == "torch":
        _validate_quant_backend(quantized)
    else:
        raise ValueError(f"Unsupported SANA-WM quant backend: {quant_backend!r}.")


def _active_quantized_precisions(
    *,
    stage1_precision: Precision,
    refiner_precision: Precision,
    refiner_enabled: bool,
) -> list[Precision]:
    """Return active non-BF16 precision requests in execution order."""
    active_precisions = [stage1_precision]
    if refiner_enabled:
        active_precisions.append(refiner_precision)
    return [precision for precision in active_precisions if precision != "bf16"]


def _resolve_quant_backend(
    quant_backend: QuantBackend,
    quantized_precisions: list[Precision],
) -> ResolvedQuantBackend:
    """Resolve ``auto`` to the Torch low-precision backend."""
    del quantized_precisions
    if quant_backend == "auto":
        return "torch"
    if quant_backend in {"torch", "torch-fp8", "torch-fp4"}:
        return quant_backend
    raise ValueError(f"Unsupported SANA-WM quant backend: {quant_backend!r}.")


def _validate_torch_fp8_backend(precisions: list[Precision]) -> None:
    """Validate the PyTorch scaled-MM backend request."""
    unsupported = sorted({precision for precision in precisions if precision != "fp8"})
    if unsupported:
        raise ValueError(
            "SANA-WM --quant-backend torch-fp8 accepts fp8 only; "
            f"unsupported requested precision(s): {', '.join(unsupported)}. "
            "Use --quant-backend torch-fp4 or --quant-backend torch for fp4."
        )
    _validate_torch_fp8_primitives()


def _validate_torch_fp4_backend(precisions: list[Precision]) -> None:
    """Validate the PyTorch NVFP4 backend request."""
    unsupported = sorted({precision for precision in precisions if precision != "fp4"})
    if unsupported:
        raise ValueError(
            "SANA-WM --quant-backend torch-fp4 accepts fp4 only; "
            f"unsupported requested precision(s): {', '.join(unsupported)}. "
            "Use --quant-backend torch-fp8 or --quant-backend torch for fp8."
        )
    _validate_torch_fp4_primitives()


def _validate_quant_backend(precisions: list[Precision]) -> None:
    """Validate PyTorch low-precision primitives."""
    if "fp8" in precisions:
        _validate_torch_fp8_primitives()
    if "fp4" in precisions:
        _validate_torch_fp4_primitives()


def _validate_torch_fp8_primitives() -> None:
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "SANA-WM --quant-backend torch-fp8 requires PyTorch with "
            "torch._scaled_mm and torch.float8_e4m3fn support."
        )


def _validate_torch_fp4_primitives() -> None:
    missing = [
        name
        for name in ("_scaled_mm", "float4_e2m1fn_x2", "float8_e4m3fn")
        if not hasattr(torch, name)
    ]
    if missing:
        raise RuntimeError(
            "SANA-WM fp4 requires PyTorch with "
            f"{', '.join(f'torch.{name}' for name in missing)} support."
        )


__all__ = [
    "Precision",
    "QuantBackend",
    "SamplingAlgo",
    "SanaWMRunner",
    "SanaWMRunnerConfig",
    "SanaWMStreamingRunner",
    "SanaWMStreamingRunnerConfig",
]
