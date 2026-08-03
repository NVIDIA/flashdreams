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

"""SANA-WM latent refiner and VAE decoder components."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.decoder import (
    DecoderConfig,
    StreamingDecoderCache,
    StreamingVideoDecoder,
)
from sana_wm._tools import resolve_hf_path
from sana_wm.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_CONFIG_PATH,
    SANA_WM_REFINER_GEMMA_ROOT,
    SANA_WM_REFINER_ROOT,
    SANA_WM_STREAMING_CAUSAL_VAE_ROOT,
    SANA_WM_STREAMING_CONFIG_PATH,
    SANA_WM_STREAMING_REFINER_GEMMA_ROOT,
    SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES,
    SANA_WM_STREAMING_REFINER_ROOT,
    SANA_WM_VAE_SPATIAL_COMPRESSION,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)
from sana_wm.transformer import (
    Precision,
    QuantBackend,
    _avoid_degenerate_tile_tail,
    _get_vae,
    _get_weight_dtype,
    _load_inference_config,
    _vae_decode_ltx2,
)


@dataclass(kw_only=True)
class SanaWMDecodedVideo:
    """Decoded SANA-WM video outputs."""

    video_hwc: np.ndarray
    stage1_video_hwc: np.ndarray | None = None


@dataclass(kw_only=True)
class SanaWMVideoDecoderCache(StreamingDecoderCache):
    """Per-rollout settings for the SANA-WM decoder/refiner stage."""

    prompt: str = ""
    fps: int = 16
    save_stage1: bool = False
    refiner_seed: int = 42
    sink_size: int = 1


@dataclass(kw_only=True)
class SanaWMLTX2VAEDecoderConfig(InstantiateConfig):
    """Config for the LTX-2 VAE latent-to-video decoder."""

    _target: type["SanaWMLTX2VAEDecoder"] = field(
        default_factory=lambda: SanaWMLTX2VAEDecoder
    )

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    vae_path: str | None = None
    """Optional VAE root override. ``None`` uses the configured YAML path."""

    height: int = DEFAULT_VIDEO_HEIGHT
    width: int = DEFAULT_VIDEO_WIDTH

    offload_vae: bool = False
    """Move the VAE to CPU between decode calls."""

    vae_tile_sample_min_width: int = 512
    vae_tile_sample_stride_width: int = 448
    vae_tile_sample_min_height: int = 512
    vae_tile_sample_stride_height: int = 448
    vae_tile_sample_min_num_frames: int = 96
    vae_tile_sample_stride_num_frames: int = 64

    vae_oom_retry_tile_sample_min_width: int = 128
    vae_oom_retry_tile_sample_stride_width: int = 64
    vae_oom_retry_tile_sample_min_height: int = 128
    vae_oom_retry_tile_sample_stride_height: int = 64
    vae_oom_retry_tile_sample_min_num_frames: int = 16
    vae_oom_retry_tile_sample_stride_num_frames: int = 8


class SanaWMLTX2VAEDecoder(nn.Module):
    """Decode SANA-WM LTX-2 VAE latents into HWC uint8 video."""

    config: SanaWMLTX2VAEDecoderConfig
    vae: Any
    vae_dtype: torch.dtype

    def __init__(self, config: SanaWMLTX2VAEDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self._dummy = nn.Parameter(torch.empty(0))
        self._runtime_config: Any | None = None
        self._vae_built = False

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    @torch.inference_mode()
    def decode_latents(self, latents: Tensor) -> np.ndarray:
        """Decode VAE latents to ``uint8`` HWC video."""
        self._ensure_vae()
        if self.config.offload_vae:
            self.vae.to(self.device)
        samples = latents.to(device=self.device, dtype=self.vae_dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        retry_decode = False
        try:
            decoded = self._vae_decode(samples)
        except torch.OutOfMemoryError:
            retry_decode = True

        if retry_decode:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            logger.warning(
                "[sana-vae] decode OOM; retrying with smaller tiles "
                "width={} stride_width={} height={} stride_height={} "
                "frames={} stride_frames={}",
                self.config.vae_oom_retry_tile_sample_min_width,
                self.config.vae_oom_retry_tile_sample_stride_width,
                self.config.vae_oom_retry_tile_sample_min_height,
                self.config.vae_oom_retry_tile_sample_stride_height,
                self.config.vae_oom_retry_tile_sample_min_num_frames,
                self.config.vae_oom_retry_tile_sample_stride_num_frames,
            )
            self._configure_vae_tiling(
                tile_sample_min_width=self.config.vae_oom_retry_tile_sample_min_width,
                tile_sample_stride_width=(
                    self.config.vae_oom_retry_tile_sample_stride_width
                ),
                tile_sample_min_height=self.config.vae_oom_retry_tile_sample_min_height,
                tile_sample_stride_height=(
                    self.config.vae_oom_retry_tile_sample_stride_height
                ),
                tile_sample_min_num_frames=(
                    self.config.vae_oom_retry_tile_sample_min_num_frames
                ),
                tile_sample_stride_num_frames=(
                    self.config.vae_oom_retry_tile_sample_stride_num_frames
                ),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            decoded = self._vae_decode(samples)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            "[timing] vae decode: {:.3f}s (latent T={} -> pixels {})",
            time.perf_counter() - t0,
            latents.shape[2],
            tuple(decoded.shape) if isinstance(decoded, Tensor) else "list",
        )
        video = _decoded_video_to_hwc_uint8(decoded)
        if self.config.offload_vae:
            self.vae.to("cpu")
        del samples, decoded
        if self.config.offload_vae and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return video

    @torch.inference_mode()
    def decode_streaming_chunk(
        self,
        latents: Tensor,
        *,
        reset_cache: bool,
    ) -> np.ndarray | None:
        """Decode one causal VAE chunk when the runtime exposes chunk caching."""
        if not bool(getattr(self.config, "use_streaming_decode_cache", False)):
            return None
        self._ensure_vae()
        if not _supports_causal_vae_chunk_decode(self.vae):
            return None
        if self.config.offload_vae:
            self.vae.to(self.device)

        samples = latents.to(device=self.device, dtype=self.vae_dtype)
        samples = _prepare_ltx2_vae_decode_samples(self.vae, samples)
        decode_per_frame = getattr(self.vae, "decode_per_frame_with_cache")
        if reset_cache:
            self.vae.clear_decoder_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            decoded = decode_per_frame(
                samples,
                temb=None,
                causal=True,
                reset_cache=reset_cache,
            )
        except TypeError as exc:
            logger.debug("[sana-vae] streaming chunk decode unavailable: {}", exc)
            if self.config.offload_vae:
                self.vae.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            "[timing] vae streaming decode: {:.3f}s (latent T={} reset={})",
            time.perf_counter() - t0,
            latents.shape[2],
            reset_cache,
        )
        video = _decoded_video_to_hwc_uint8(decoded)
        if self.config.offload_vae:
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del samples, decoded
        return video

    def _ensure_runtime_config(self) -> Any:
        if self._runtime_config is None:
            self._runtime_config = _load_inference_config(self.config.config_path)
        return self._runtime_config

    def _ensure_vae(self) -> None:
        if self._vae_built:
            return
        cfg = self._ensure_runtime_config()
        self.vae_dtype = _get_weight_dtype(cfg.vae.weight_dtype)
        vae_pretrained = self.config.vae_path or cfg.vae.vae_pretrained
        cfg.vae.vae_pretrained = resolve_hf_path(vae_pretrained)
        self.vae = _get_vae(
            cfg.vae.vae_type,
            cfg.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=cfg.vae,
        )
        if hasattr(self.vae, "enable_tiling"):
            self._configure_vae_tiling()
        self._vae_built = True

    def _configure_vae_tiling(
        self,
        *,
        tile_sample_min_width: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_min_height: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
    ) -> None:
        vae: Any = self.vae
        min_width = int(tile_sample_min_width or self.config.vae_tile_sample_min_width)
        stride_width = int(
            tile_sample_stride_width or self.config.vae_tile_sample_stride_width
        )
        min_height = int(
            tile_sample_min_height or self.config.vae_tile_sample_min_height
        )
        stride_height = int(
            tile_sample_stride_height or self.config.vae_tile_sample_stride_height
        )
        spatial_ratio = int(getattr(vae, "spatial_compression_ratio", 1))
        stride_width = _avoid_degenerate_tile_tail(
            sample_extent=self.config.width,
            sample_tile_min=min_width,
            sample_stride=stride_width,
            compression_ratio=spatial_ratio,
        )
        stride_height = _avoid_degenerate_tile_tail(
            sample_extent=self.config.height,
            sample_tile_min=min_height,
            sample_stride=stride_height,
            compression_ratio=spatial_ratio,
        )
        min_frames = int(
            tile_sample_min_num_frames or self.config.vae_tile_sample_min_num_frames
        )
        stride_frames = int(
            tile_sample_stride_num_frames
            or self.config.vae_tile_sample_stride_num_frames
        )
        temporal_ratio = int(getattr(vae, "temporal_compression_ratio", 1))
        if temporal_ratio > 1:
            min_frames = max(min_frames, temporal_ratio)
            stride_frames = max(stride_frames, temporal_ratio)
        kwargs = {
            "tile_sample_min_height": min_height,
            "tile_sample_stride_height": stride_height,
            "tile_sample_min_width": min_width,
            "tile_sample_stride_width": stride_width,
            "tile_sample_min_num_frames": min_frames,
            "tile_sample_stride_num_frames": stride_frames,
        }
        if hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling(**kwargs)
            except TypeError:
                vae.enable_tiling()
        for name, value in kwargs.items():
            if hasattr(vae, name):
                setattr(vae, name, value)
        if hasattr(vae, "use_framewise_encoding"):
            vae.use_framewise_encoding = True
        if hasattr(vae, "use_framewise_decoding"):
            vae.use_framewise_decoding = True
        logger.info(
            "[sana-vae] tiling width={} stride_width={} height={} "
            "stride_height={} frames={} stride_frames={}",
            min_width,
            stride_width,
            min_height,
            stride_height,
            min_frames,
            stride_frames,
        )

    def _vae_decode(self, latents: Tensor) -> Tensor:
        return _vae_decode_ltx2(
            self._ensure_runtime_config().vae.vae_type,
            self.vae,
            latents,
        )


def _prepare_ltx2_vae_decode_samples(vae: Any, latents: Tensor) -> Tensor:
    """Convert normalized SANA-WM latents to the VAE decoder latent domain."""
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(
        latents.device,
        latents.dtype,
    )
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(
        latents.device,
        latents.dtype,
    )
    scale = float(vae.config.scaling_factor)
    return latents * latents_std / scale + latents_mean


def _supports_causal_vae_chunk_decode(vae: Any) -> bool:
    """Return whether ``vae`` exposes the causal chunk decode contract."""
    decoder = getattr(vae, "decoder", None)
    if (
        decoder is not None
        and hasattr(decoder, "is_causal")
        and not bool(getattr(decoder, "is_causal"))
    ):
        return False
    return callable(getattr(vae, "clear_decoder_cache", None)) and callable(
        getattr(vae, "decode_per_frame_with_cache", None)
    )


def _decoded_video_to_hwc_uint8(decoded: Tensor | list[Tensor]) -> np.ndarray:
    """Convert VAE decode output to a single ``uint8`` HWC video array."""
    decoded_tensor: Tensor
    if isinstance(decoded, list):
        if not decoded:
            raise ValueError("VAE decode returned no frames.")
        frames = cast(list[Tensor], decoded)
        first = frames[0]
        if first.ndim == 5:
            decoded_tensor = torch.cat(frames, dim=2)
        elif first.ndim == 4:
            decoded_tensor = torch.stack(frames, dim=2)
        else:
            decoded_tensor = torch.stack(frames, dim=0)
    else:
        decoded_tensor = decoded
    if decoded_tensor.ndim != 5:
        raise ValueError(
            "SANA-WM VAE decode output must have shape [B, C, T, H, W]; "
            f"got {tuple(decoded_tensor.shape)}."
        )
    return (
        torch.clamp(127.5 * decoded_tensor + 127.5, 0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 4, 1)
        .contiguous()
        .cpu()
        .numpy()[0]
    )


@dataclass(kw_only=True)
class SanaWMLTX2LatentRefinerConfig(InstantiateConfig):
    """Config for the optional LTX-2 latent refiner component."""

    _target: type["SanaWMLTX2LatentRefiner"] = field(
        default_factory=lambda: SanaWMLTX2LatentRefiner
    )

    refiner_root: str = SANA_WM_REFINER_ROOT
    refiner_gemma_root: str = SANA_WM_REFINER_GEMMA_ROOT
    refiner_precision: Precision = "bf16"
    quant_backend: QuantBackend = "torch"
    offload_refiner: bool = False
    cache_text_encoder: bool = False
    """Keep Gemma cached on CPU after prompt encoding for repeated pipeline use."""


class SanaWMLTX2LatentRefiner(nn.Module):
    """Run the optional LTX-2 refinement stage over Stage-1 latents."""

    config: SanaWMLTX2LatentRefinerConfig

    def __init__(self, config: SanaWMLTX2LatentRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self._dummy = nn.Parameter(torch.empty(0))
        self._refiner_built = False

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    @torch.inference_mode()
    def refine_latents(
        self,
        *,
        latents: Tensor,
        prompt: str,
        fps: int,
        sink_size: int,
        seed: int,
    ) -> Tensor:
        """Run the LTX-2 refiner."""
        self._ensure_refiner()
        refined = self.refiner.refine_latents(
            latents,
            prompt,
            fps=float(fps),
            sink_size=int(sink_size),
            seed=int(seed),
            progress=True,
        )
        if self.config.offload_refiner:
            self.release_runtime()
        return refined

    def release_runtime(self) -> None:
        """Release refiner tensors."""
        if not self._refiner_built:
            return
        del self.refiner
        self._refiner_built = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _ensure_refiner(self) -> None:
        if self._refiner_built:
            return
        from sana_wm.refiner import SanaWMLTX2Refiner

        t0 = time.perf_counter()
        compute_dtype = (
            torch.bfloat16
            if self.config.refiner_precision in {"fp8", "fp4"}
            else _get_weight_dtype(self.config.refiner_precision)
        )
        self.refiner = SanaWMLTX2Refiner(
            refiner_root=resolve_hf_path(self.config.refiner_root),
            gemma_root=resolve_hf_path(self.config.refiner_gemma_root),
            dtype=compute_dtype,
            device=self.device,
            precision=self.config.refiner_precision,
            quant_backend=self.config.quant_backend,
            cache_text_encoder=self.config.cache_text_encoder,
        )
        self._refiner_built = True
        logger.info(
            "[timing] refiner build: {:.3f}s (precision={})",
            time.perf_counter() - t0,
            self.config.refiner_precision,
        )


@dataclass(kw_only=True)
class SanaWMVideoDecoderConfig(DecoderConfig):
    """Config for the SANA-WM latent refiner plus VAE decode boundary."""

    _target: type["SanaWMVideoDecoder"] = field(
        default_factory=lambda: SanaWMVideoDecoder
    )

    vae_decoder: SanaWMLTX2VAEDecoderConfig = field(
        default_factory=SanaWMLTX2VAEDecoderConfig
    )
    refiner: SanaWMLTX2LatentRefinerConfig | None = field(
        default_factory=SanaWMLTX2LatentRefinerConfig
    )


class SanaWMVideoDecoder(StreamingVideoDecoder[SanaWMVideoDecoderCache]):
    """Decode Stage-1 latents, optionally through the LTX-2 refiner."""

    config: SanaWMVideoDecoderConfig

    def __init__(self, config: SanaWMVideoDecoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.vae_decoder = config.vae_decoder.setup()
        self.refiner = config.refiner.setup() if config.refiner is not None else None

    def initialize_autoregressive_cache(
        self,
        **context: Any,
    ) -> SanaWMVideoDecoderCache:
        """Build per-rollout decode/refiner settings."""
        return SanaWMVideoDecoderCache(**context)

    @torch.inference_mode()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: SanaWMVideoDecoderCache | None = None,
    ) -> SanaWMDecodedVideo:
        """Refine/decode one SANA-WM latent rollout."""
        if autoregressive_index != 0:
            raise ValueError("SANA-WM bidirectional inference has one AR step.")
        cache = cache or SanaWMVideoDecoderCache()
        stage1_latent = input
        output_latent = input
        if self.refiner is not None:
            output_latent = self.refiner.refine_latents(
                latents=stage1_latent,
                prompt=cache.prompt,
                fps=cache.fps,
                sink_size=cache.sink_size,
                seed=cache.refiner_seed,
            )
            # refine_latents() already releases the refiner when offload_refiner
            # is set; keep it resident otherwise so a reused pipeline avoids a
            # full refiner rebuild + re-quantization on the next call.
        elif cache.save_stage1:
            logger.info(
                "SANA-WM is already running without the refiner; "
                "--save-stage1 does not create an extra output."
            )

        video_hwc = self.vae_decoder.decode_latents(output_latent)
        if self.refiner is not None:
            video_hwc = video_hwc[1:]
        stage1_video_hwc = None
        if cache.save_stage1 and self.refiner is not None:
            stage1_video_hwc = self.vae_decoder.decode_latents(stage1_latent)
        return SanaWMDecodedVideo(
            video_hwc=video_hwc,
            stage1_video_hwc=stage1_video_hwc,
        )

    @property
    def spatial_compression_ratio(self) -> int:
        """Pixel side divided by latent side."""
        return SANA_WM_VAE_SPATIAL_COMPRESSION

    @property
    def temporal_compression_ratio(self) -> int:
        """Pixel frame compression ratio after the first latent."""
        return SANA_WM_VAE_TEMPORAL_COMPRESSION

    def get_output_temporal_size(
        self,
        autoregressive_index: int,
        input_temporal_size: int,
    ) -> int:
        """Return decoded pixel frames for a SANA-WM latent sequence."""
        if autoregressive_index != 0:
            raise ValueError("SANA-WM bidirectional inference has one AR step.")
        if input_temporal_size <= 0:
            raise ValueError(
                f"input_temporal_size must be positive, got {input_temporal_size}."
            )
        return 1 + (input_temporal_size - 1) * self.temporal_compression_ratio

    def get_input_temporal_size(
        self,
        autoregressive_index: int,
        output_temporal_size: int,
    ) -> int:
        """Return latent frames required to decode ``output_temporal_size`` pixels."""
        if autoregressive_index != 0:
            raise ValueError("SANA-WM bidirectional inference has one AR step.")
        ratio = self.temporal_compression_ratio
        remainder = (output_temporal_size - 1) % ratio
        if output_temporal_size <= 0 or remainder != 0:
            raise ValueError(
                "SANA-WM output frame count must be positive and equal to 8k+1; "
                f"got {output_temporal_size}."
            )
        return ((output_temporal_size - 1) // ratio) + 1


@dataclass(kw_only=True)
class SanaWMStreamingVideoDecoderCache(SanaWMVideoDecoderCache):
    """Per-rollout cache for streaming SANA-WM decode/refine."""

    block_size: int = 3
    refiner_kv_max_frames: int = SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES
    vae_streaming_cache_ready: bool = False
    stage1_chunks: list[Tensor] = field(default_factory=list)
    refined_chunks: list[Tensor] = field(default_factory=list)
    stage1_sink: Tensor | None = None
    refiner_prompt_embeds: Tensor | None = None
    refiner_prompt_attention_mask: Tensor | None = None
    refiner_generator: torch.Generator | None = None


@dataclass(kw_only=True)
class SanaWMStreamingLTX2VAEDecoderConfig(SanaWMLTX2VAEDecoderConfig):
    """Config for the streaming causal LTX-2 VAE decode path."""

    config_path: str = SANA_WM_STREAMING_CONFIG_PATH
    vae_path: str | None = SANA_WM_STREAMING_CAUSAL_VAE_ROOT
    use_streaming_decode_cache: bool = True
    """Use the causal VAE's per-chunk decoder cache when available."""


@dataclass(kw_only=True)
class SanaWMStreamingLTX2LatentRefinerConfig(SanaWMLTX2LatentRefinerConfig):
    """Config for the streaming chunk-causal LTX-2 refiner path."""

    _target: type["SanaWMStreamingLTX2LatentRefiner"] = field(
        default_factory=lambda: SanaWMStreamingLTX2LatentRefiner
    )

    refiner_root: str = SANA_WM_STREAMING_REFINER_ROOT
    refiner_gemma_root: str = SANA_WM_STREAMING_REFINER_GEMMA_ROOT
    kv_max_frames: int = SANA_WM_STREAMING_REFINER_KV_MAX_FRAMES
    block_size: int = 3


class SanaWMStreamingLTX2LatentRefiner(SanaWMLTX2LatentRefiner):
    """Streaming refiner adapter using FlashDreams-owned chunk state."""

    config: SanaWMStreamingLTX2LatentRefinerConfig

    @torch.inference_mode()
    def refine_chunk(
        self,
        *,
        context_latents: Tensor,
        active_latents: Tensor,
        prompt: str,
        prompt_embeds: Tensor | None,
        prompt_attention_mask: Tensor | None,
        fps: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Refine one active chunk against clean context latents."""
        self._ensure_refiner()
        if prompt_embeds is None or prompt_attention_mask is None:
            prompt_embeds, prompt_attention_mask = self.refiner.encode_prompt(prompt)
        refined = self.refiner.refine_active_latents(
            context_latents=context_latents,
            active_latents=active_latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            fps=fps,
            generator=generator,
        )
        return refined, prompt_embeds, prompt_attention_mask


@dataclass(kw_only=True)
class SanaWMStreamingVideoDecoderConfig(DecoderConfig):
    """Config for streaming SANA-WM latent refinement and chunk decode."""

    _target: type["SanaWMStreamingVideoDecoder"] = field(
        default_factory=lambda: SanaWMStreamingVideoDecoder
    )

    vae_decoder: SanaWMStreamingLTX2VAEDecoderConfig = field(
        default_factory=SanaWMStreamingLTX2VAEDecoderConfig
    )
    refiner: SanaWMStreamingLTX2LatentRefinerConfig | None = field(
        default_factory=SanaWMStreamingLTX2LatentRefinerConfig
    )


class SanaWMStreamingVideoDecoder(
    StreamingVideoDecoder[SanaWMStreamingVideoDecoderCache]
):
    """Decode SANA-WM streaming latent chunks into newly available frames."""

    config: SanaWMStreamingVideoDecoderConfig

    def __init__(self, config: SanaWMStreamingVideoDecoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.vae_decoder = config.vae_decoder.setup()
        self.refiner = config.refiner.setup() if config.refiner is not None else None

    def initialize_autoregressive_cache(
        self,
        **context: Any,
    ) -> SanaWMStreamingVideoDecoderCache:
        """Build per-rollout streaming decode/refiner settings."""
        return SanaWMStreamingVideoDecoderCache(**context)

    @torch.inference_mode()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: SanaWMStreamingVideoDecoderCache | None = None,
    ) -> SanaWMDecodedVideo:
        """Refine/decode one SANA-WM streaming latent chunk."""
        cache = cache or SanaWMStreamingVideoDecoderCache()
        stage1_active = _split_streaming_stage1_chunk(
            input,
            autoregressive_index=autoregressive_index,
            cache=cache,
        )
        refiner_context = _latent_context(
            sink=cache.stage1_sink,
            chunks=cache.refined_chunks,
            max_frames=_max_streaming_context_frames(
                cache,
                active_frames=int(stage1_active.shape[2]),
            ),
        )
        output_active = stage1_active
        if self.refiner is not None:
            if cache.refiner_generator is None:
                cache.refiner_generator = torch.Generator(
                    device=stage1_active.device
                ).manual_seed(int(cache.refiner_seed))
            (
                output_active,
                cache.refiner_prompt_embeds,
                cache.refiner_prompt_attention_mask,
            ) = self.refiner.refine_chunk(
                context_latents=refiner_context,
                active_latents=stage1_active,
                prompt=cache.prompt,
                prompt_embeds=cache.refiner_prompt_embeds,
                prompt_attention_mask=cache.refiner_prompt_attention_mask,
                fps=cache.fps,
                generator=cache.refiner_generator,
            )
        elif cache.save_stage1:
            logger.info(
                "SANA-WM streaming is already running without the refiner; "
                "--save-stage1 does not create an extra output."
            )

        video_hwc = _decode_streaming_active_frames(
            self.vae_decoder,
            context_latents=refiner_context,
            active_latents=output_active,
            temporal_compression_ratio=self.temporal_compression_ratio,
            cache=None if cache.save_stage1 else cache,
        )
        cache.refined_chunks.append(output_active.detach().contiguous())
        stage1_video_hwc = None
        if cache.save_stage1 and self.refiner is not None:
            stage1_context = _latent_context(
                sink=cache.stage1_sink,
                chunks=cache.stage1_chunks,
                max_frames=_max_streaming_context_frames(
                    cache,
                    active_frames=int(stage1_active.shape[2]),
                ),
            )
            stage1_video_hwc = _decode_streaming_active_frames(
                self.vae_decoder,
                context_latents=stage1_context,
                active_latents=stage1_active,
                temporal_compression_ratio=self.temporal_compression_ratio,
                cache=None,
            )
        cache.stage1_chunks.append(stage1_active.detach().contiguous())
        return SanaWMDecodedVideo(
            video_hwc=video_hwc,
            stage1_video_hwc=stage1_video_hwc,
        )

    @property
    def spatial_compression_ratio(self) -> int:
        """Pixel side divided by latent side."""
        return SANA_WM_VAE_SPATIAL_COMPRESSION

    @property
    def temporal_compression_ratio(self) -> int:
        """Pixel frames emitted per steady-state latent frame."""
        return SANA_WM_VAE_TEMPORAL_COMPRESSION

    def get_output_temporal_size(
        self,
        autoregressive_index: int,
        input_temporal_size: int,
    ) -> int:
        """Return newly emitted pixel frames for one streaming latent chunk."""
        if input_temporal_size <= 0:
            raise ValueError(
                f"input_temporal_size must be positive, got {input_temporal_size}."
            )
        ratio = self.temporal_compression_ratio
        if autoregressive_index == 0:
            if input_temporal_size <= 1:
                raise ValueError(
                    "Streaming AR step 0 must include sink + active frames."
                )
            return (input_temporal_size - 1) * ratio
        return input_temporal_size * ratio

    def get_input_temporal_size(
        self,
        autoregressive_index: int,
        output_temporal_size: int,
    ) -> int:
        """Return latent frames needed to emit ``output_temporal_size`` pixels."""
        ratio = self.temporal_compression_ratio
        if output_temporal_size <= 0 or output_temporal_size % ratio != 0:
            raise ValueError(
                "SANA-WM streaming output frame count must be a positive "
                f"multiple of {ratio}; got {output_temporal_size}."
            )
        latent_frames = output_temporal_size // ratio
        return latent_frames + 1 if autoregressive_index == 0 else latent_frames


def _split_streaming_stage1_chunk(
    chunk: Tensor,
    *,
    autoregressive_index: int,
    cache: SanaWMStreamingVideoDecoderCache,
) -> Tensor:
    """Return active latent frames and initialize the sink on the first chunk."""
    if autoregressive_index < 0:
        raise ValueError(
            f"autoregressive_index must be >= 0, got {autoregressive_index}."
        )
    if chunk.shape[2] <= 0:
        raise ValueError("SANA-WM streaming latent chunk must contain frames.")
    if autoregressive_index == 0:
        if cache.stage1_sink is not None:
            raise RuntimeError("SANA-WM streaming sink was already initialized.")
        if chunk.shape[2] <= cache.sink_size:
            raise ValueError("Streaming AR step 0 must include sink + active frames.")
        cache.stage1_sink = chunk[:, :, : cache.sink_size].detach().contiguous()
        return chunk[:, :, cache.sink_size :].contiguous()
    if cache.stage1_sink is None:
        raise RuntimeError("SANA-WM streaming AR step 0 must run before later chunks.")
    return chunk.contiguous()


def _max_streaming_context_frames(
    cache: SanaWMStreamingVideoDecoderCache,
    *,
    active_frames: int,
) -> int:
    """Return sink plus rolling-history context length for one active block."""
    if active_frames <= 0:
        raise ValueError(f"active_frames must be positive, got {active_frames}.")
    return max(int(cache.sink_size), int(cache.refiner_kv_max_frames) - active_frames)


def _latent_context(
    *,
    sink: Tensor | None,
    chunks: list[Tensor],
    max_frames: int,
) -> Tensor:
    """Build clean sink-plus-history context capped to ``max_frames`` latents."""
    if sink is None:
        raise RuntimeError("SANA-WM streaming sink has not been initialized.")
    if max_frames <= sink.shape[2]:
        return sink
    history_budget = max_frames - int(sink.shape[2])
    history = _tail_latent_frames(chunks, history_budget)
    if history is None:
        return sink
    return torch.cat([sink, history], dim=2)


def _tail_latent_frames(chunks: list[Tensor], max_frames: int) -> Tensor | None:
    """Return the last ``max_frames`` frames from a latent chunk list."""
    if max_frames <= 0 or not chunks:
        return None
    selected: list[Tensor] = []
    remaining = int(max_frames)
    for chunk in reversed(chunks):
        if remaining <= 0:
            break
        take = min(int(chunk.shape[2]), remaining)
        selected.append(chunk[:, :, -take:])
        remaining -= take
    if not selected:
        return None
    return torch.cat(list(reversed(selected)), dim=2).contiguous()


def _decode_streaming_active_frames(
    vae_decoder: SanaWMLTX2VAEDecoder,
    *,
    context_latents: Tensor,
    active_latents: Tensor,
    temporal_compression_ratio: int,
    cache: SanaWMStreamingVideoDecoderCache | None = None,
) -> np.ndarray:
    """Decode active frames with latent context and drop context pixels."""
    if cache is not None:
        reset_cache = not cache.vae_streaming_cache_ready
        streaming_latents = (
            torch.cat([context_latents, active_latents], dim=2)
            if reset_cache
            else active_latents
        )
        video = vae_decoder.decode_streaming_chunk(
            streaming_latents,
            reset_cache=reset_cache,
        )
        if video is not None:
            cache.vae_streaming_cache_ready = True
            if not reset_cache:
                return video
            start = _pixel_frames_for_latents(
                int(context_latents.shape[2]),
                temporal_compression_ratio=temporal_compression_ratio,
            )
            return video[start:]
        cache.vae_streaming_cache_ready = False

    decode_latents = torch.cat([context_latents, active_latents], dim=2)
    video = vae_decoder.decode_latents(decode_latents)
    start = _pixel_frames_for_latents(
        int(context_latents.shape[2]),
        temporal_compression_ratio=temporal_compression_ratio,
    )
    return video[start:]


def _pixel_frames_for_latents(
    latent_frames: int,
    *,
    temporal_compression_ratio: int,
) -> int:
    """Return decoded pixel frames for a non-empty latent prefix."""
    if latent_frames <= 0:
        return 0
    return 1 + (latent_frames - 1) * int(temporal_compression_ratio)


__all__ = [
    "SanaWMDecodedVideo",
    "SanaWMLTX2LatentRefiner",
    "SanaWMLTX2LatentRefinerConfig",
    "SanaWMLTX2VAEDecoder",
    "SanaWMLTX2VAEDecoderConfig",
    "SanaWMStreamingLTX2LatentRefiner",
    "SanaWMStreamingLTX2LatentRefinerConfig",
    "SanaWMStreamingLTX2VAEDecoderConfig",
    "SanaWMStreamingVideoDecoder",
    "SanaWMStreamingVideoDecoderCache",
    "SanaWMStreamingVideoDecoderConfig",
    "SanaWMVideoDecoder",
    "SanaWMVideoDecoderCache",
    "SanaWMVideoDecoderConfig",
]
