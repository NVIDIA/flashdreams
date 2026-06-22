# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams streaming integration for Helios (HeliosPyramidPipeline)."""

from __future__ import annotations

import gc
import time
from typing import Any, Generator

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from flashdreams.infra.pipeline import StreamInferencePipeline
from helios.cache import HeliosPipelineCache
from helios.compiler import compile_transformer, enable_flash_attention
from helios.encoder import HeliosEncoder
from helios.helios_loader import load_helios_pipeline

# Helios native pixel chunk size (33 = (9-1)*4+1 latent frames × temporal scale).
HELIOS_CHUNK_FRAMES = 33
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)


class HeliosStreamingPipeline(StreamInferencePipeline):
    """Expose Helios' native 33-frame chunks as a FlashDreams streaming interface."""

    diffusion_model: Any = None
    encoder: Any = None
    decoder: Any = None

    def __init__(self, config: Any) -> None:
        nn.Module.__init__(self)
        self.config = config
        self._device = torch.device(getattr(config, "device", None) or "cuda")

        checkpoint = getattr(config, "checkpoint", "BestWishYsh/Helios-Distilled")
        self.checkpoint = checkpoint
        self.pyramid_steps: list[int] = list(
            getattr(config, "pyramid_steps", [2, 2, 2])
        )
        self.guidance_scale: float = float(getattr(config, "guidance_scale", 1.0))
        self.use_compile: bool = bool(getattr(config, "compile", False))
        self.warmup_discard_chunks: int = int(
            getattr(config, "warmup_discard_chunks", 0)
        )
        self.amplify_first: bool = bool(getattr(config, "amplify_first_chunk", True))
        self.use_flash_attention: bool = bool(getattr(config, "flash_attention", True))
        self.history_len: int = int(getattr(config, "history_len", 8))
        self.enable_parallelism: bool = bool(
            getattr(config, "enable_parallelism", False)
        )
        self.cp_backend: str = str(getattr(config, "cp_backend", "ulysses"))
        self.group_offload: bool = bool(getattr(config, "group_offload", False))

        if self.use_flash_attention:
            enable_flash_attention()

        self.pipe = load_helios_pipeline(
            checkpoint,
            self._device,
            enable_parallelism=self.enable_parallelism,
            cp_backend=self.cp_backend,
        )

        if self.group_offload:
            offload_fn = getattr(self.pipe, "enable_group_offload", None)
            if offload_fn is not None:
                offload_fn(offload_type="leaf_level")
                print("[Helios pipeline] Group offloading enabled (leaf_level)")

        self._transformer_orig = self.pipe.transformer
        self._transformer_compiled: nn.Module | None = None
        if self.use_compile:
            self._transformer_compiled = compile_transformer(self.pipe.transformer)
            self.pipe.transformer = self._transformer_compiled
            self._ensure_transformer_on_device()

        self.encoder_wrapper = HeliosEncoder(self.pipe)

        self._optimization_warmup_done = False
        self._warmup_prompt: str | None = None
        self._compile_disabled_reason: str | None = None

    def _disable_compile(self, reason: str) -> None:
        """Permanently fall back to eager DiT after a compile failure."""
        if not self.use_compile:
            return
        self.use_compile = False
        self._compile_disabled_reason = reason
        self.pipe.transformer = self._transformer_orig
        print(f"[Helios pipeline] torch.compile disabled: {reason}", flush=True)

    @property
    def optimization_warmup_done(self) -> bool:
        return self._optimization_warmup_done

    @torch.no_grad()
    def warmup_optimizations(
        self,
        prompt: str,
        width: int = 640,
        height: int = 384,
        *,
        discard_ar_steps: int | None = None,
    ) -> dict[str, float | int | bool]:
        """Run discarded chunk(s) to compile kernels or prime cuDNN before timed streaming."""
        if discard_ar_steps is None:
            discard_ar_steps = (
                1 if self.use_compile else max(0, self.warmup_discard_chunks)
            )
        if discard_ar_steps <= 0:
            self._optimization_warmup_done = True
            self._warmup_prompt = prompt
            return {"seconds": 0.0, "discarded_chunks": 0, "already_warm": True}

        self._ensure_transformer_on_device()
        t0 = time.perf_counter()
        cache = self.initialize_cache(text=[prompt])
        for ar_step in range(discard_ar_steps):
            self.generate(ar_step, cache, width=width, height=height)
            self.finalize(ar_step, cache)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._optimization_warmup_done = True
        self._warmup_prompt = prompt
        del cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
        label = "compile" if self.use_compile else "kernel"
        print(
            f"[Helios pipeline] Optimization warmup complete "
            f"({discard_ar_steps} discarded chunk(s), {elapsed:.1f}s, {label})"
        )
        return {
            "seconds": elapsed,
            "discarded_chunks": discard_ar_steps,
            "already_warm": False,
            "compile_active": self.use_compile,
        }

    @property
    def device(self) -> torch.device:
        return self._device

    def _ensure_transformer_on_device(self) -> None:
        """Keep DiT weights on the inference device (compile wraps the same module)."""
        dev = self._device
        base = self._transformer_orig
        if any(p.device != dev for p in base.parameters()):
            print(f"[Helios pipeline] Moving DiT weights to {dev}", flush=True)
            base.to(dev)
        if self.use_compile and self._transformer_compiled is not None:
            self.pipe.transformer = self._transformer_compiled
        else:
            self.pipe.transformer = base

    @torch.no_grad()
    def initialize_cache(
        self,
        transformer_context: dict[str, Any] | None = None,
        encoder_context: dict[str, Any] | None = None,
        decoder_context: dict[str, Any] | None = None,
        *,
        text: list[str] | None = None,
        prompt: list[str] | None = None,
        negative_prompt: list[str] | None = None,
        image: Image.Image | None = None,
        **kwargs: Any,
    ) -> HeliosPipelineCache:
        prompts = text or prompt
        if prompts is None:
            raise ValueError("initialize_cache requires text= or prompt=")
        if negative_prompt is None:
            negative_prompt = [DEFAULT_NEGATIVE_PROMPT] * len(prompts)

        cond = self.encoder_wrapper.encode(
            prompt=prompts,
            negative_prompt=negative_prompt,
            device=self.device,
            guidance_scale=self.guidance_scale,
            image=image,
        )
        return HeliosPipelineCache(cond=cond)

    def _video_conditioning(self, cache: HeliosPipelineCache) -> Tensor | None:
        """Convert prior decoded frames to [B, T, C, H, W] for Helios V2V conditioning.

        Helios requires at least 33 pixel frames for ``video=`` conditioning.
        """
        if not cache.decoded_chunks:
            return None
        # decoded_chunks entries are [T, C, H, W] in [-1, 1]
        prior = torch.cat(cache.decoded_chunks, dim=0)
        if prior.shape[0] < HELIOS_CHUNK_FRAMES:
            return None
        # Helios needs ≥33 frames; use the most recent window for continuity.
        prior = prior[-HELIOS_CHUNK_FRAMES:]
        prior = (prior.float() + 1.0) / 2.0
        # VideoProcessor.preprocess_video expects [B, T, C, H, W].
        video = prior.unsqueeze(0)
        return video.to(device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: HeliosPipelineCache,
        input: Any = None,
        *,
        width: int = 640,
        height: int = 384,
        **kwargs: Any,
    ) -> Tensor:
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous={prev}, expected={expected}, "
            f"got={autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        cond = cache.cond
        assert cond is not None

        pipe_kwargs: dict[str, Any] = dict(
            prompt_embeds=cond.prompt_embeds,
            negative_prompt_embeds=cond.negative_prompt_embeds,
            num_frames=HELIOS_CHUNK_FRAMES,
            height=height,
            width=width,
            pyramid_num_inference_steps_list=self.pyramid_steps,
            guidance_scale=self.guidance_scale,
            is_amplify_first_chunk=(autoregressive_index == 0 and self.amplify_first),
            output_type="pt",
            generator=torch.Generator(device="cuda").manual_seed(
                42 + autoregressive_index
            ),
        )

        if cond.image is not None and autoregressive_index == 0:
            pipe_kwargs["image"] = cond.image

        video_cond = self._video_conditioning(cache)
        if video_cond is not None and autoregressive_index > 0:
            pipe_kwargs["video"] = video_cond

        if self.use_compile:
            self._ensure_transformer_on_device()
            torch.compiler.cudagraph_mark_step_begin()

        try:
            result = self.pipe(**pipe_kwargs)
        except Exception as exc:
            if self.use_compile:
                print(
                    f"[Helios pipeline] chunk {autoregressive_index}: compiled forward "
                    f"failed ({exc!r}), retrying with uncompiled DiT",
                    flush=True,
                )
                self.pipe.transformer = self._transformer_orig
                try:
                    result = self.pipe(**pipe_kwargs)
                except Exception as exc2:
                    if video_cond is not None:
                        print(
                            f"[Helios pipeline] chunk {autoregressive_index}: video conditioning "
                            f"failed ({exc2!r}), retrying without history",
                            flush=True,
                        )
                        pipe_kwargs.pop("video", None)
                        result = self.pipe(**pipe_kwargs)
                    else:
                        raise
                self._disable_compile(f"chunk {autoregressive_index} forward failed")
            elif video_cond is not None:
                print(
                    f"[Helios pipeline] chunk {autoregressive_index}: video conditioning "
                    f"failed ({exc!r}), retrying without history",
                    flush=True,
                )
                pipe_kwargs.pop("video", None)
                result = self.pipe(**pipe_kwargs)
            else:
                raise
        frames = result.frames
        if isinstance(frames, list):
            frames = frames[0]
        if frames.ndim == 5 and frames.shape[0] == 1:
            frames = frames[0]

        normalized = self._normalize_frames_to_flashdreams(frames)
        cache.decoded_chunks.append(normalized.detach().cpu())
        cache.pending_history = normalized[-self.history_len :].unsqueeze(0)

        return normalized

    @staticmethod
    def _normalize_frames_to_flashdreams(frames: Tensor) -> Tensor:
        """Return [T, C, H, W] float in [-1, 1] for FlashDreams runners."""
        if frames.ndim == 5:
            frames = frames[0]
        if frames.shape[0] <= 4 and frames.shape[1] > 4:
            frames = frames.permute(1, 0, 2, 3)
        if frames.max() <= 1.0 and frames.min() >= 0.0:
            frames = frames * 2.0 - 1.0
        return frames.float()

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: HeliosPipelineCache,
    ) -> dict[str, float] | None:
        assert cache.autoregressive_index == autoregressive_index
        if cache.pending_history is not None:
            cache.history_frames = cache.pending_history
            cache.pending_history = None
        return {"history_len": float(self.history_len)}

    def stream(
        self,
        prompt: str,
        total_blocks: int = 8,
        width: int = 640,
        height: int = 384,
        image: Image.Image | None = None,
    ) -> Generator[Tensor, None, None]:
        cache = self.initialize_cache(text=[prompt], image=image)
        for ar_step in range(total_blocks):
            frames = self.generate(ar_step, cache, width=width, height=height)
            self.finalize(ar_step, cache)
            yield frames.unsqueeze(0)

    @property
    def active_optimizations(self) -> dict[str, bool]:
        return {
            "flash_attention": self.use_flash_attention,
            "compile": self.use_compile,
            "kernel_warmup": self.warmup_discard_chunks > 0,
            "parallelism": self.enable_parallelism,
            "group_offload": self.group_offload,
        }
