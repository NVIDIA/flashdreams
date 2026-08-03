# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full FlashDreams LTX-Video streaming pipeline."""

from __future__ import annotations

import gc
import time
from typing import Any, Generator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor

from flashdreams.infra.pipeline import StreamInferencePipeline

from ltx_video.attention import install_kv_attention_processors
from ltx_video.cache import LTXPipelineCache
from ltx_video.compiler import CUDAGraphRunner, compile_transformer, enable_flash_attention
from ltx_video.decoder import LTXDecoder
from ltx_video.encoder import LTXEncoder
from ltx_video.kv_cache import LTXKVCache
from ltx_video.kv_context import configure_kv_context, get_kv_context, reset_kv_context
from ltx_video.ltx_loader import load_ltx_pipeline


class LTXVideoStreamingPipeline(StreamInferencePipeline):
    """FlashDreams integration for Lightricks/LTX-Video."""

    diffusion_model: Any = None
    encoder: Any = None
    decoder: Any = None

    def __init__(self, config: Any) -> None:
        nn.Module.__init__(self)
        self.config = config
        self._device = torch.device(getattr(config, "device", None) or "cuda")

        checkpoint = getattr(config, "checkpoint", "Lightricks/LTX-Video")
        self.pipe = load_ltx_pipeline(checkpoint, self._device, dtype=torch.bfloat16)

        self.chunk_frames: int = getattr(config, "chunk_frames", 25)
        self.chunk_overlap: int = getattr(config, "chunk_overlap", 1)
        self.num_inference_steps: int = getattr(config, "num_inference_steps", 50)
        self.guidance_scale: float = getattr(config, "guidance_scale", 3.0)
        self.frame_rate: float = getattr(config, "frame_rate", 24.0)

        self.use_kv_cache: bool = getattr(config, "kv_cache", False)
        self.use_compile: bool = getattr(config, "compile", False)
        self.use_cuda_graphs: bool = getattr(config, "cuda_graphs", False)
        self.use_taehv: bool = getattr(config, "use_taehv", False)
        self.use_flash_attention: bool = getattr(config, "flash_attention", True)
        self.use_manual_denoise: bool = getattr(config, "manual_denoise", False)
        self.kv_window_size: int | None = getattr(config, "kv_window_size", None)

        if self.use_flash_attention:
            enable_flash_attention()

        self._num_kv_layers = 0
        if self.use_kv_cache:
            self._num_kv_layers = install_kv_attention_processors(self.pipe.transformer)

        self._transformer_orig = self.pipe.transformer
        self._transformer_compiled: nn.Module | None = None
        if self.use_compile:
            self._transformer_compiled = compile_transformer(
                self.pipe.transformer, use_kv_cache=self.use_kv_cache
            )
            self.pipe.transformer = self._transformer_compiled
            # Do not offload _transformer_orig to CPU — it is the same module object
            # torch.compile wraps; moving it to CPU leaves proj_in weights on CPU while
            # activations stay on GPU (Dynamo fake-tensor device mismatch).
            self._ensure_transformer_on_device(recompile_if_moved=False)

        self._optimization_warmup_done = False
        self._warmup_prompt: str | None = None

        self._cuda_graph_runner: CUDAGraphRunner | None = (
            CUDAGraphRunner() if self.use_cuda_graphs else None
        )
        self._cuda_graphs_enabled = self.use_cuda_graphs
        if self.use_cuda_graphs and self.use_compile:
            print(
                "[LTX compiler] CUDA graphs disabled with torch.compile "
                "(incompatible graph capture on compiled DiT; graphs still apply when compile=False)"
            )
            self._cuda_graphs_enabled = False

        self.encoder_wrapper = LTXEncoder(self.pipe)
        self.decoder_wrapper = LTXDecoder(self.pipe.vae, use_taehv=self.use_taehv)
        self.scheduler = self.pipe.scheduler

        # Manual denoise + compile/KV/graphs must not fall back to pipe().
        self._strict_manual = self.use_manual_denoise and (
            self.use_compile or self.use_cuda_graphs or self.use_kv_cache
        )

    @staticmethod
    def _vae_spatial_scale(pipe: Any) -> int:
        return int(
            getattr(pipe, "vae_spatial_compression_ratio", None)
            or getattr(pipe, "vae_scale_factor", None)
            or getattr(pipe.vae, "spatial_compression_ratio", 8)
        )

    @staticmethod
    def _vae_temporal_scale(pipe: Any) -> int:
        return int(
            getattr(pipe, "vae_temporal_compression_ratio", None)
            or getattr(pipe.vae, "temporal_compression_ratio", 8)
        )

    @property
    def device(self) -> torch.device:
        return self._device

    def _ensure_transformer_on_device(self, *, recompile_if_moved: bool = True) -> None:
        """Ensure DiT weights live on the inference device before compile/warmup."""
        base = self._transformer_orig
        dev = self._device
        bad = any(p.device != dev for p in base.parameters())
        if bad:
            print(f"[LTX pipeline] Moving DiT weights to {dev} (mixed cpu/gpu detected)", flush=True)
            base.to(dev)
            if self.use_compile and recompile_if_moved:
                self._transformer_compiled = compile_transformer(
                    base, use_kv_cache=self.use_kv_cache
                )
                self.pipe.transformer = self._transformer_compiled
        elif self.use_compile and self._transformer_compiled is not None:
            self.pipe.transformer = self._transformer_compiled
        else:
            self.pipe.transformer = base

        sample = base.proj_in.weight
        if sample.device != dev:
            raise RuntimeError(
                f"DiT proj_in still on {sample.device}, expected {dev}. "
                "Restart the comparison UI (./scripts/start_comparison_ui.sh) "
                "and click Free GPU memory before warmup."
            )
        print(f"[LTX pipeline] DiT proj_in on {sample.device}", flush=True)

    def _latent_dims(self, width: int, height: int) -> tuple[int, int, int]:
        temporal = self._vae_temporal_scale(self.pipe)
        spatial = self._vae_spatial_scale(self.pipe)
        latent_t = (self.chunk_frames - 1) // temporal + 1
        latent_h = height // spatial
        latent_w = width // spatial
        return latent_t, latent_h, latent_w

    def _prepare_timesteps(
        self, width: int, height: int
    ) -> tuple[torch.Tensor, int, int, int, tuple]:
        from diffusers.pipelines.ltx.pipeline_ltx import calculate_shift, retrieve_timesteps

        latent_t, latent_h, latent_w = self._latent_dims(width, height)
        seq_len = latent_t * latent_h * latent_w
        sigmas = np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
        mu = calculate_shift(
            seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )
        rope_scale = (
            self._vae_temporal_scale(self.pipe) / self.frame_rate,
            self._vae_spatial_scale(self.pipe),
            self._vae_spatial_scale(self.pipe),
        )
        return timesteps, latent_t, latent_h, latent_w, rope_scale

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
    ) -> LTXPipelineCache:
        prompts = text or prompt
        if prompts is None:
            raise ValueError("initialize_cache requires text= or prompt=")
        if negative_prompt is None:
            negative_prompt = [
                "worst quality, inconsistent motion, blurry, jittery"
            ] * len(prompts)

        cond = self.encoder_wrapper.encode(
            prompt=prompts,
            negative_prompt=negative_prompt,
            device=self.device,
            image=image,
        )
        return LTXPipelineCache(
            cond=cond,
            kv=LTXKVCache(window_size=self.kv_window_size),
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: LTXPipelineCache,
        input: Any = None,
        *,
        width: int = 768,
        height: int = 512,
        **kwargs: Any,
    ) -> Tensor:
        prev = cache.autoregressive_index
        expected = (prev + 1) if prev is not None else 0
        assert autoregressive_index == expected, (
            f"AR step out of order: previous={prev}, expected={expected}, "
            f"got={autoregressive_index}"
        )
        cache.autoregressive_index = autoregressive_index

        if self.use_manual_denoise:
            try:
                frames = self._generate_manual_denoise(
                    autoregressive_index, cache, width, height
                )
            except Exception as exc:
                if self._strict_manual:
                    raise RuntimeError(
                        f"Optimized manual denoise failed with compile/kv/graphs enabled: {exc}"
                    ) from exc
                print(f"[LTX pipeline] manual denoise failed ({exc}), using pipe()")
                frames = self._generate_streaming_decode(
                    autoregressive_index, cache, width, height
                )
        else:
            frames = self._generate_streaming_decode(
                autoregressive_index, cache, width, height
            )

        cache.decoded_chunks.append(frames.detach().cpu())
        return frames

    def _transformer_denoise_step(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
        timestep: Tensor,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        rope_scale: tuple,
    ) -> Tensor:
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()

        if self.use_kv_cache and self._num_kv_layers > 0:
            ctx = get_kv_context()
            ctx.begin_forward()

        with self.pipe.transformer.cache_context("cond_uncond"):
            return self.pipe.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                num_frames=latent_t,
                height=latent_h,
                width=latent_w,
                rope_interpolation_scale=rope_scale,
                return_dict=False,
            )[0]

    @property
    def optimization_warmup_done(self) -> bool:
        return self._optimization_warmup_done

    def is_ready_for_timed_stream(self, prompt: str) -> bool:
        return self._optimization_warmup_done and self._warmup_prompt == prompt

    @torch.no_grad()
    def warmup_optimizations(
        self,
        prompt: str,
        width: int = 768,
        height: int = 512,
        *,
        discard_ar_steps: int = 1,
    ) -> dict[str, float | int | bool]:
        """Run full discarded chunk(s) to compile kernels before timed streaming."""
        if not self.use_manual_denoise:
            self._optimization_warmup_done = True
            self._warmup_prompt = prompt
            return {"seconds": 0.0, "discarded_chunks": 0, "already_warm": True}

        self._ensure_transformer_on_device()
        t0 = time.perf_counter()
        cache = self.initialize_cache(
            text=[prompt],
            negative_prompt=["worst quality, inconsistent motion, blurry, jittery"],
        )
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
        print(
            f"[LTX compiler] Optimization warmup complete "
            f"({discard_ar_steps} discarded chunk(s), {elapsed:.1f}s)"
        )
        return {
            "seconds": elapsed,
            "discarded_chunks": discard_ar_steps,
            "already_warm": False,
        }

    def _generate_manual_denoise(
        self,
        step: int,
        cache: LTXPipelineCache,
        width: int,
        height: int,
    ) -> Tensor:
        """Manual denoising loop with optional KV-cache, compile, and CUDA graphs."""
        self._ensure_transformer_on_device()
        cond = cache.cond
        assert cond is not None

        batch_size = cond.prompt_embeds.shape[0]
        timesteps, latent_t, latent_h, latent_w, rope_scale = self._prepare_timesteps(
            width, height
        )

        generator = torch.Generator(self.device).manual_seed(42 + step)
        latents = self.pipe.prepare_latents(
            batch_size,
            self.pipe.transformer.config.in_channels,
            height,
            width,
            self.chunk_frames,
            torch.float32,
            self.device,
            generator,
        )

        enc_hs = torch.cat([cond.negative_prompt_embeds, cond.prompt_embeds])
        enc_mask = torch.cat(
            [cond.negative_prompt_attention_mask, cond.prompt_attention_mask]
        )

        past_kv = cache.kv.get() if self.use_kv_cache else None
        configure_kv_context(
            past_kv=past_kv,
            collect=self.use_kv_cache and self._num_kv_layers > 0,
            num_layers=self._num_kv_layers,
        )

        last_present_kv = None

        for t in timesteps:
            latent_input = torch.cat([latents] * 2).to(dtype=enc_hs.dtype)
            t_batch = t.expand(latent_input.shape[0])

            step_kwargs = dict(
                hidden_states=latent_input,
                encoder_hidden_states=enc_hs,
                encoder_attention_mask=enc_mask,
                timestep=t_batch,
            )

            if self._cuda_graph_runner is not None and self._cuda_graphs_enabled:
                try:

                    def _fwd(
                        hidden_states: Tensor,
                        encoder_hidden_states: Tensor,
                        encoder_attention_mask: Tensor,
                        timestep: Tensor,
                    ) -> Tensor:
                        return self._transformer_denoise_step(
                            hidden_states,
                            encoder_hidden_states,
                            encoder_attention_mask,
                            timestep,
                            latent_t,
                            latent_h,
                            latent_w,
                            rope_scale,
                        )

                    noise_pred = self._cuda_graph_runner(_fwd, **step_kwargs)
                except Exception as graph_exc:
                    print(
                        f"[LTX compiler] CUDA graph failed ({graph_exc}), disabling graphs"
                    )
                    self._cuda_graph_runner.reset()
                    self._cuda_graphs_enabled = False
                    noise_pred = self._transformer_denoise_step(
                        latent_input, enc_hs, enc_mask, t_batch,
                        latent_t, latent_h, latent_w, rope_scale,
                    )
            else:
                noise_pred = self._transformer_denoise_step(
                    latent_input, enc_hs, enc_mask, t_batch,
                    latent_t, latent_h, latent_w, rope_scale,
                )

            if self.use_kv_cache:
                last_present_kv = get_kv_context().collected_kv()

            noise_pred = noise_pred.float()
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred_guided = noise_uncond + self.guidance_scale * (
                noise_cond - noise_uncond
            )
            latents = self.scheduler.step(
                noise_pred_guided, t, latents, return_dict=False
            )[0]

        reset_kv_context()
        cache.pending_kv = last_present_kv

        latents = self.pipe._unpack_latents(
            latents,
            latent_t,
            latent_h,
            latent_w,
            self.pipe.transformer_spatial_patch_size,
            self.pipe.transformer_temporal_patch_size,
        )
        latents = self.pipe._denormalize_latents(
            latents,
            self.pipe.vae.latents_mean,
            self.pipe.vae.latents_std,
            self.pipe.vae.config.scaling_factor,
        )
        decoded = self.decoder_wrapper.decode_from_denoised(latents)
        return self._to_flashdreams_video_tensor(decoded)

    def _generate_streaming_decode(
        self,
        step: int,
        cache: LTXPipelineCache,
        width: int,
        height: int,
    ) -> Tensor:
        """Standard diffusers pipe() per chunk — always uses uncompiled transformer."""
        cond = cache.cond
        assert cond is not None

        if self.use_compile:
            self.pipe.transformer = self._transformer_orig

        result = self.pipe(
            prompt_embeds=cond.prompt_embeds,
            negative_prompt_embeds=cond.negative_prompt_embeds,
            prompt_attention_mask=cond.prompt_attention_mask,
            negative_prompt_attention_mask=cond.negative_prompt_attention_mask,
            num_frames=self.chunk_frames,
            num_inference_steps=self.num_inference_steps,
            width=width,
            height=height,
            guidance_scale=self.guidance_scale,
            output_type="pt",
            generator=torch.Generator(self.device).manual_seed(42 + step),
        )

        if self.use_compile and self._transformer_compiled is not None:
            self.pipe.transformer = self._transformer_compiled

        frames = result.frames
        if isinstance(frames, list):
            frames = frames[0]
        if frames.ndim == 5:
            frames = frames[0]
        return self._normalize_frames_to_flashdreams(frames)

    @staticmethod
    def _to_flashdreams_video_tensor(decoded: Tensor) -> Tensor:
        # Match streaming path: [T, C, H, W] in [-1, 1]
        frames = decoded[0]
        return (frames * 2.0 - 1.0).float()

    @staticmethod
    def _normalize_frames_to_flashdreams(frames: Tensor) -> Tensor:
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
        cache: LTXPipelineCache,
    ) -> dict[str, float] | None:
        assert cache.autoregressive_index == autoregressive_index
        if self.use_kv_cache and cache.pending_kv is not None:
            cache.kv.update(cache.pending_kv)
            cache.pending_kv = None
        stats: dict[str, float] = {}
        if self.use_kv_cache:
            stats["kv_seq_len"] = float(cache.kv.seq_len)
        return stats or None

    def stream(
        self,
        prompt: str,
        total_blocks: int = 7,
        width: int = 768,
        height: int = 512,
        **kwargs: Any,
    ) -> Generator[Tensor, None, None]:
        cache = self.initialize_cache(
            text=[prompt],
            negative_prompt=["worst quality, inconsistent motion, blurry, jittery"],
        )
        for ar_step in range(total_blocks):
            frames = self.generate(ar_step, cache, width=width, height=height, **kwargs)
            self.finalize(ar_step, cache)
            yield frames.unsqueeze(0)

    @property
    def active_optimizations(self) -> dict[str, bool]:
        return {
            "flash_attention": self.use_flash_attention,
            "manual_denoise": self.use_manual_denoise,
            "kv_cache": self.use_kv_cache and self._num_kv_layers > 0,
            "compile": self.use_compile,
            "cuda_graphs": self._cuda_graphs_enabled,
        }
