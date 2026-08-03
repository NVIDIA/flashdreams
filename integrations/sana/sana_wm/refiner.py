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

"""LTX-2 refiner used by the SANA-WM integration."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from sana_wm.quant import (
    TorchScaledMMFP4Recipe,
    TorchScaledMMFP8Recipe,
    replace_linear_with_quant,
)

Precision = Literal["bf16", "fp8", "fp4"]
QuantBackend = Literal["auto", "torch", "torch-fp8", "torch-fp4"]

_REFINER_QUANT_SKIP_DEFAULTS = (
    r"^proj_in$",
    r"^proj_out$",
    r"(^|\.)audio_",
    r"audio_to_video",
    r"video_to_audio",
    r"av_cross_attn",
    r"caption_projection",
    r"time_embed",
)


@dataclass(frozen=True)
class _RefinerBlockContext:
    """Cached context K/V for one refiner transformer block."""

    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class _RefinerContextCache:
    """Per-chunk refiner context state reused across denoise steps."""

    blocks: tuple[_RefinerBlockContext, ...]
    encoder_hidden_states: Tensor
    encoder_attention_mask: Tensor | None
    video_rotary_emb: tuple[Tensor, Tensor]
    n_context_tokens: int


def _move_tensor_attr(module: nn.Module, name: str, device: torch.device | str) -> None:
    tensor = getattr(module, name, None)
    if isinstance(tensor, nn.Parameter):
        setattr(
            module,
            name,
            nn.Parameter(tensor.to(device), requires_grad=tensor.requires_grad),
        )
    elif isinstance(tensor, Tensor):
        setattr(module, name, tensor.to(device))


def _offload_video_unused_audio_modules(
    transformer: nn.Module,
    device: torch.device | str = "cpu",
) -> None:
    for name in (
        "audio_proj_in",
        "audio_caption_projection",
        "audio_time_embed",
        "av_cross_attn_video_scale_shift",
        "av_cross_attn_audio_scale_shift",
        "av_cross_attn_video_a2v_gate",
        "av_cross_attn_audio_v2a_gate",
        "audio_rope",
        "cross_attn_rope",
        "cross_attn_audio_rope",
        "audio_norm_out",
        "audio_proj_out",
    ):
        child = getattr(transformer, name, None)
        if isinstance(child, nn.Module):
            child.to(device)
    for block in getattr(transformer, "transformer_blocks", ()):
        for name in (
            "audio_norm1",
            "audio_attn1",
            "audio_norm2",
            "audio_attn2",
            "audio_to_video_norm",
            "audio_to_video_attn",
            "video_to_audio_norm",
            "video_to_audio_attn",
            "audio_norm3",
            "audio_ff",
        ):
            child = getattr(block, name, None)
            if isinstance(child, nn.Module):
                child.to(device)


def _move_ltx2_video_modules_to(
    transformer: nn.Module,
    device: torch.device | str,
) -> None:
    for name in (
        "proj_in",
        "caption_projection",
        "time_embed",
        "rope",
        "norm_out",
        "proj_out",
    ):
        child = getattr(transformer, name, None)
        if isinstance(child, nn.Module):
            child.to(device)
    _move_tensor_attr(transformer, "scale_shift_table", device)
    for block in getattr(transformer, "transformer_blocks", ()):
        _move_tensor_attr(block, "scale_shift_table", device)
        for name in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
            child = getattr(block, name, None)
            if isinstance(child, nn.Module):
                child.to(device)


class SanaWMLTX2Refiner(nn.Module):
    """Run the SANA-WM LTX-2 latent refiner without importing a Sana checkout."""

    transformer: Any
    connectors: Any
    tokenizer: Any
    text_encoder: Any

    def __init__(
        self,
        *,
        refiner_root: str | Path,
        gemma_root: str | Path,
        dtype: torch.dtype,
        device: torch.device | str,
        precision: Precision = "bf16",
        quant_backend: QuantBackend = "torch",
        text_max_sequence_length: int = 1024,
        cache_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.refiner_root = Path(refiner_root)
        self.gemma_root = Path(gemma_root)
        self.dtype = dtype
        self.device = torch.device(device)
        self.precision = precision
        self.quant_backend = quant_backend
        self.text_max_sequence_length = int(text_max_sequence_length)
        self.cache_text_encoder = bool(cache_text_encoder)
        self._quantized = False
        self._text_encoder_built = False
        self.transformer, self.connectors = self._load_diffusers_components()

    @torch.inference_mode()
    def refine_latents(
        self,
        sana_latent: Tensor,
        prompt: str,
        *,
        fps: float,
        sink_size: int = 1,
        seed: int = 42,
        progress: bool = True,
        sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0),
    ) -> Tensor:
        """Refine Stage-1 VAE latents with the sink-bidirectional LTX-2 path."""
        if sana_latent.shape[2] <= sink_size:
            raise ValueError(
                f"Stage-1 latent has {sana_latent.shape[2]} frames but "
                f"sink_size={sink_size}."
            )

        prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt)
        self._prepare_video_runtime()

        z = sana_latent.to(device=self.device, dtype=self.dtype)
        sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas_t[0])
        sink = z[:, :, :sink_size].contiguous()
        current = z[:, :, sink_size:].contiguous()
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        eps = torch.randn(
            current.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        noisy = (1.0 - start_sigma) * current + start_sigma * eps
        packed_context = self._pack_refiner_context(sink)
        context_cache = self._build_refiner_context_cache(
            context_tokens=packed_context,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=int(z.shape[2]),
            height=int(z.shape[3]),
            width=int(z.shape[4]),
            fps=fps,
        )

        iterator = range(len(sigmas_t) - 1)
        if progress:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc="refiner", unit="step")

        for step_index in iterator:
            sigma = sigmas_t[step_index]
            denoised = self._predict_current_x0(
                sink=sink,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
                packed_context_tokens=packed_context,
                context_cache=context_cache,
            )
            noisy_tokens = _pack_latents(
                noisy,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = (
                noisy_tokens.float()
                + velocity * (sigmas_t[step_index + 1] - sigma).float()
            )
            noisy = _unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )

        return torch.cat([sink, noisy], dim=2)

    @torch.inference_mode()
    def encode_prompt(self, prompt: str) -> tuple[Tensor, Tensor]:
        """Encode a prompt for one or more LTX-2 refinement calls."""
        return self._encode_prompt(prompt)

    @torch.inference_mode()
    def refine_active_latents(
        self,
        *,
        context_latents: Tensor,
        active_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        fps: float,
        generator: torch.Generator,
        sigmas: tuple[float, ...] = (0.909375, 0.725, 0.421875, 0.0),
    ) -> Tensor:
        """Refine one active latent block against frozen context latents.

        This is the FlashDreams-owned streaming refinement primitive. The
        context is treated as clean, fixed latent history; only ``active_latents``
        are noised and updated by the deterministic Euler steps.
        """
        if context_latents.shape[2] <= 0:
            raise ValueError("context_latents must contain at least one frame.")
        if active_latents.shape[2] <= 0:
            raise ValueError("active_latents must contain at least one frame.")

        self._prepare_video_runtime()
        context = context_latents.to(device=self.device, dtype=self.dtype).contiguous()
        active = active_latents.to(device=self.device, dtype=self.dtype).contiguous()
        sigmas_t = torch.tensor(sigmas, dtype=torch.float32, device=self.device)
        start_sigma = float(sigmas_t[0])
        eps = torch.randn(
            active.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        noisy = (1.0 - start_sigma) * active + start_sigma * eps
        packed_context = self._pack_refiner_context(context)
        context_cache = self._build_refiner_context_cache(
            context_tokens=packed_context,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=int(context.shape[2] + active.shape[2]),
            height=int(context.shape[3]),
            width=int(context.shape[4]),
            fps=fps,
        )

        for step_index in range(len(sigmas_t) - 1):
            sigma = sigmas_t[step_index]
            denoised = self._predict_current_x0(
                sink=context,
                noisy_current=noisy,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                sigma=sigma,
                fps=fps,
                packed_context_tokens=packed_context,
                context_cache=context_cache,
            )
            noisy_tokens = _pack_latents(
                noisy,
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            velocity = (noisy_tokens.float() - denoised.float()) / sigma.float()
            next_tokens = (
                noisy_tokens.float()
                + velocity * (sigmas_t[step_index + 1] - sigma).float()
            )
            noisy = _unpack_latents(
                next_tokens.to(self.dtype),
                num_frames=noisy.shape[2],
                height=noisy.shape[3],
                width=noisy.shape[4],
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )

        return noisy

    def _prepare_video_runtime(self) -> None:
        """Prepare video-only refiner modules for latent refinement."""
        _move_ltx2_video_modules_to(self.transformer, self.device)
        _offload_video_unused_audio_modules(self.transformer, "cpu")
        self.transformer.eval()
        self._prepare_quantization()

    def _pack_refiner_context(self, context: Tensor) -> Tensor | None:
        """Pack fixed context once when temporal patches do not cross the split."""
        if int(self.transformer.config.patch_size_t) != 1:
            return None
        return _pack_latents(
            context,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )

    def _build_refiner_context_cache(
        self,
        *,
        context_tokens: Tensor | None,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
    ) -> _RefinerContextCache | None:
        """Precompute clean context block state for active-only refinement."""
        if context_tokens is None or not _refiner_context_cache_enabled():
            return None
        transformer = self.transformer
        for name in ("rope", "proj_in", "time_embed", "caption_projection"):
            if not hasattr(transformer, name):
                return None
        if not hasattr(transformer, "transformer_blocks"):
            return None
        batch_size = context_tokens.size(0)
        encoder_attention_mask = _prepare_encoder_attention_mask(
            prompt_attention_mask,
            context_tokens.dtype,
        )
        video_coords = transformer.rope.prepare_video_coords(
            batch_size,
            num_frames,
            height,
            width,
            context_tokens.device,
            fps=fps,
        )
        video_rotary_emb = transformer.rope(video_coords, device=context_tokens.device)
        context_rotary_emb = _slice_refiner_rotary_emb(
            video_rotary_emb,
            0,
            context_tokens.shape[1],
        )

        hidden_states = transformer.proj_in(context_tokens)
        context_timestep = torch.zeros(
            batch_size,
            context_tokens.shape[1],
            dtype=torch.float32,
            device=context_tokens.device,
        )
        temb, _embedded_timestep = transformer.time_embed(
            context_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        encoder_hidden_states = transformer.caption_projection(prompt_embeds)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size,
            -1,
            hidden_states.size(-1),
        )

        blocks: list[_RefinerBlockContext] = []
        for block in transformer.transformer_blocks:
            hidden_states, block_context = _forward_refiner_context_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=context_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
            blocks.append(block_context)

        return _RefinerContextCache(
            blocks=tuple(blocks),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            video_rotary_emb=video_rotary_emb,
            n_context_tokens=int(context_tokens.shape[1]),
        )

    def _load_diffusers_components(self) -> tuple[nn.Module, nn.Module]:
        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2 import LTX2TextConnectors

        transformer = LTX2VideoTransformer3DModel.from_pretrained(
            self.refiner_root,
            subfolder="transformer",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).eval()
        connectors = LTX2TextConnectors.from_pretrained(
            self.refiner_root,
            subfolder="connectors",
            torch_dtype=self.dtype,
            local_files_only=True,
        ).eval()
        return transformer, connectors

    def _prepare_quantization(self) -> None:
        if self._quantized or self.precision == "bf16":
            return
        recipe = (
            TorchScaledMMFP8Recipe()
            if self.precision == "fp8"
            else TorchScaledMMFP4Recipe()
        )
        converted, skipped = replace_linear_with_quant(
            self.transformer,
            recipe=recipe,
            params_dtype=self.dtype,
            skip_patterns=_REFINER_QUANT_SKIP_DEFAULTS,
        )
        if converted <= 0:
            raise RuntimeError(
                f"SANA-WM refiner {self.precision} converted no Linear "
                f"layers; skipped={skipped}."
            )
        self._quantized = True
        recipe_detail = ""
        if isinstance(recipe, TorchScaledMMFP4Recipe):
            recipe_detail = (
                f" rht={recipe.use_rht}"
                f" global_scale={recipe.use_global_scale}"
                f" weight_scale_2d={recipe.weight_scale_2d}"
            )
        logger.info(
            "[refiner-quant] precision={}{} converted {} Linear layers (skipped {})",
            self.precision,
            recipe_detail,
            converted,
            skipped,
        )

    def _ensure_text_encoder(self) -> None:
        """Load the Gemma tokenizer + encoder once and cache them in CPU RAM.

        The encoder is ~20 GB. By default it is released after the one-shot
        prompt encode to match upstream's single-generation path and avoid a
        large GPU-to-CPU copy. Set ``cache_text_encoder=True`` for repeated
        pipeline reuse, where paying the copy once can beat reloading Gemma on
        every call.
        """
        if self._text_encoder_built:
            return
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

        t0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.gemma_root, local_files_only=True
        )
        tokenizer = cast(Any, tokenizer)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        # Built on CPU; moved to the GPU on demand for each encode call only.
        self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
            self.gemma_root,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        self._text_encoder_built = True
        logger.info(
            "[timing] refiner text-encoder build: {:.3f}s",
            time.perf_counter() - t0,
        )

    @torch.inference_mode()
    def _encode_prompt(self, prompt: str) -> tuple[Tensor, Tensor]:
        self._ensure_text_encoder()
        tokenizer = self.tokenizer

        text_inputs = tokenizer(
            [prompt.strip()],
            padding="max_length",
            max_length=self.text_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        # Pull the cached encoder onto the GPU only for the forward, then park
        # it back on the CPU so it is absent during denoise/decode.
        self.text_encoder.to(self.device)
        text_backbone: nn.Module | None = None
        try:
            text_backbone = getattr(self.text_encoder, "model", self.text_encoder)
            outputs = text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = torch.stack(outputs.hidden_states, dim=-1)
            sequence_lengths = attention_mask.sum(dim=-1)
            del outputs
        finally:
            release_text_encoder = not self.cache_text_encoder
            if release_text_encoder:
                del text_backbone
                del self.text_encoder
                self._text_encoder_built = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                self.text_encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        prompt_embeds = _pack_text_embeds(
            hidden_states,
            sequence_lengths,
            device=self.device,
            padding_side=tokenizer.padding_side,
        ).to(dtype=self.dtype)
        del hidden_states

        self.connectors.to(self.device)
        connector_prompt_embeds, _, connector_attention_mask = self.connectors(
            prompt_embeds,
            attention_mask,
        )
        self.connectors.to("cpu")
        del prompt_embeds, attention_mask
        return (
            connector_prompt_embeds.to(device=self.device, dtype=self.dtype),
            connector_attention_mask.to(device=self.device),
        )

    def _predict_current_x0(
        self,
        *,
        sink: Tensor,
        noisy_current: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        sigma: Tensor,
        fps: float,
        packed_context_tokens: Tensor | None = None,
        context_cache: _RefinerContextCache | None = None,
    ) -> Tensor:
        batch_size, _, context_frames, height, width = sink.shape
        current_tokens = _pack_latents(
            noisy_current,
            patch_size=self.transformer.config.patch_size,
            patch_size_t=self.transformer.config.patch_size_t,
        )
        if context_cache is not None and packed_context_tokens is not None:
            raw_timestep = torch.full(
                (batch_size, current_tokens.shape[1], 1),
                float(sigma),
                dtype=torch.float32,
                device=self.device,
            )
            model_timestep = raw_timestep.squeeze(-1) * float(
                self.transformer.config.timestep_scale_multiplier
            )
            velocity = self._forward_video_current_only(
                hidden_states=current_tokens,
                timestep=model_timestep,
                context_cache=context_cache,
            )
            denoised = current_tokens.float() - velocity.float() * raw_timestep
            return denoised.to(self.dtype)

        if packed_context_tokens is None:
            latent_tokens = _pack_latents(
                torch.cat([sink, noisy_current], dim=2),
                patch_size=self.transformer.config.patch_size,
                patch_size_t=self.transformer.config.patch_size_t,
            )
            n_context_tokens = latent_tokens.shape[1] - current_tokens.shape[1]
        else:
            latent_tokens = torch.cat([packed_context_tokens, current_tokens], dim=1)
            n_context_tokens = packed_context_tokens.shape[1]
        num_frames = context_frames + int(noisy_current.shape[2])

        raw_timestep = torch.zeros(
            batch_size,
            latent_tokens.shape[1],
            1,
            dtype=torch.float32,
            device=self.device,
        )
        raw_timestep[:, n_context_tokens:, 0] = sigma.float()
        model_timestep = raw_timestep.squeeze(-1) * float(
            self.transformer.config.timestep_scale_multiplier
        )

        velocity = self._forward_video_only(
            hidden_states=latent_tokens,
            encoder_hidden_states=prompt_embeds,
            timestep=model_timestep,
            encoder_attention_mask=prompt_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            n_context_tokens=n_context_tokens,
        )
        denoised = latent_tokens.float() - velocity.float() * raw_timestep
        return denoised[:, n_context_tokens:, :].to(self.dtype)

    def _forward_video_only(
        self,
        *,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Tensor | None,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        n_context_tokens: int,
    ) -> Tensor:
        transformer = self.transformer
        batch_size = hidden_states.size(0)
        encoder_attention_mask = _prepare_encoder_attention_mask(
            encoder_attention_mask,
            hidden_states.dtype,
        )
        video_coords = transformer.rope.prepare_video_coords(
            batch_size,
            num_frames,
            height,
            width,
            hidden_states.device,
            fps=fps,
        )
        video_rotary_emb = transformer.rope(video_coords, device=hidden_states.device)

        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size,
            -1,
            embedded_timestep.size(-1),
        )

        encoder_hidden_states = transformer.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size,
            -1,
            hidden_states.size(-1),
        )

        for block in transformer.transformer_blocks:
            hidden_states = _forward_video_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                video_rotary_emb=video_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                n_context_tokens=n_context_tokens,
            )

        scale_shift_values = (
            transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)

    def _forward_video_current_only(
        self,
        *,
        hidden_states: Tensor,
        timestep: Tensor,
        context_cache: _RefinerContextCache,
    ) -> Tensor:
        """Run the refiner over active tokens with cached clean context K/V."""
        transformer = self.transformer
        batch_size = hidden_states.size(0)
        if batch_size != context_cache.encoder_hidden_states.size(0):
            raise ValueError(
                "Refiner context cache batch size does not match active tokens: "
                f"{context_cache.encoder_hidden_states.size(0)} != {batch_size}."
            )
        current_start = int(context_cache.n_context_tokens)
        current_end = current_start + int(hidden_states.shape[1])
        current_rotary_emb = _slice_refiner_rotary_emb(
            context_cache.video_rotary_emb,
            current_start,
            current_end,
        )

        hidden_states = transformer.proj_in(hidden_states)
        temb, embedded_timestep = transformer.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size,
            -1,
            embedded_timestep.size(-1),
        )

        if len(context_cache.blocks) != len(transformer.transformer_blocks):
            raise ValueError("Refiner context cache block count is stale.")
        for block, block_context in zip(
            transformer.transformer_blocks,
            context_cache.blocks,
        ):
            hidden_states = _forward_refiner_current_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=context_cache.encoder_hidden_states,
                temb=temb,
                video_rotary_emb=current_rotary_emb,
                encoder_attention_mask=context_cache.encoder_attention_mask,
                context=block_context,
            )

        scale_shift_values = (
            transformer.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = transformer.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        return transformer.proj_out(hidden_states)


def _forward_video_block(
    *,
    block: Any,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    temb: Tensor,
    video_rotary_emb: tuple[Tensor, Tensor],
    encoder_attention_mask: Tensor | None,
    n_context_tokens: int,
) -> Tensor:
    batch_size = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size,
        temb.size(1),
        num_ada_params,
        -1,
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(
        dim=2
    )
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    attn_hidden_states = _streaming_self_attention(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
        n_context_tokens=n_context_tokens,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    return hidden_states + block.ff(norm_hidden_states) * gate_mlp


def _forward_refiner_context_block(
    *,
    block: Any,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    temb: Tensor,
    video_rotary_emb: tuple[Tensor, Tensor],
    encoder_attention_mask: Tensor | None,
) -> tuple[Tensor, _RefinerBlockContext]:
    """Run one block for clean context and cache its self-attention K/V."""
    batch_size = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _refiner_block_ada_values(block, temb, batch_size)
    )
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    query, key, value, gate_logits = _refiner_self_attention_qkv(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
    )
    attn_hidden_states = _finish_refiner_self_attention(
        attn=block.attn1,
        hidden_states=_refiner_attention(query, key, value),
        gate_logits=gate_logits,
        dtype=query.dtype,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    hidden_states = hidden_states + block.ff(norm_hidden_states) * gate_mlp
    return hidden_states, _RefinerBlockContext(key=key, value=value)


def _forward_refiner_current_block(
    *,
    block: Any,
    hidden_states: Tensor,
    encoder_hidden_states: Tensor,
    temb: Tensor,
    video_rotary_emb: tuple[Tensor, Tensor],
    encoder_attention_mask: Tensor | None,
    context: _RefinerBlockContext,
) -> Tensor:
    """Run one active-token block using cached clean context K/V."""
    batch_size = hidden_states.size(0)
    norm_hidden_states = block.norm1(hidden_states)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        _refiner_block_ada_values(block, temb, batch_size)
    )
    norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

    query, key, value, gate_logits = _refiner_self_attention_qkv(
        attn=block.attn1,
        hidden_states=norm_hidden_states,
        query_rotary_emb=video_rotary_emb,
    )
    key = torch.cat([context.key, key], dim=1)
    value = torch.cat([context.value, value], dim=1)
    attn_hidden_states = _finish_refiner_self_attention(
        attn=block.attn1,
        hidden_states=_refiner_attention(query, key, value),
        gate_logits=gate_logits,
        dtype=query.dtype,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa

    norm_hidden_states = block.norm2(hidden_states)
    attn_hidden_states = block.attn2(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        query_rotary_emb=None,
        attention_mask=encoder_attention_mask,
    )
    hidden_states = hidden_states + attn_hidden_states

    norm_hidden_states = block.norm3(hidden_states) * (1 + scale_mlp) + shift_mlp
    return hidden_states + block.ff(norm_hidden_states) * gate_mlp


def _refiner_block_ada_values(
    block: Any,
    temb: Tensor,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    num_ada_params = block.scale_shift_table.shape[0]
    ada_values = block.scale_shift_table[None, None].to(temb.device) + temb.reshape(
        batch_size,
        temb.size(1),
        num_ada_params,
        -1,
    )
    return ada_values[:, :, :6].unbind(dim=2)


def _streaming_self_attention(
    *,
    attn: Any,
    hidden_states: Tensor,
    query_rotary_emb: tuple[Tensor, Tensor],
    n_context_tokens: int,
) -> Tensor:
    query, key, value, gate_logits = _refiner_self_attention_qkv(
        attn=attn,
        hidden_states=hidden_states,
        query_rotary_emb=query_rotary_emb,
    )

    if n_context_tokens <= 0 or n_context_tokens >= query.shape[1]:
        hidden_states = _refiner_attention(query, key, value)
    else:
        context_hidden_states = _refiner_attention(
            query[:, :n_context_tokens],
            key[:, :n_context_tokens],
            value[:, :n_context_tokens],
        )
        current_hidden_states = _refiner_attention(
            query[:, n_context_tokens:],
            key,
            value,
        )
        hidden_states = torch.cat(
            [context_hidden_states, current_hidden_states],
            dim=1,
        )

    return _finish_refiner_self_attention(
        attn=attn,
        hidden_states=hidden_states,
        gate_logits=gate_logits,
        dtype=query.dtype,
    )


def _refiner_self_attention_qkv(
    *,
    attn: Any,
    hidden_states: Tensor,
    query_rotary_emb: tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    from diffusers.models.transformers.transformer_ltx2 import (
        apply_interleaved_rotary_emb,
        apply_split_rotary_emb,
    )

    gate_logits = (
        attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
    )
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(key, query_rotary_emb)
    elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(key, query_rotary_emb)
    else:
        raise ValueError(f"Unsupported LTX-2 RoPE type: {attn.rope_type}")

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    return query, key, value, gate_logits


def _finish_refiner_self_attention(
    *,
    attn: Any,
    hidden_states: Tensor,
    gate_logits: Tensor | None,
    dtype: torch.dtype,
) -> Tensor:
    hidden_states = hidden_states.flatten(2, 3).to(dtype)
    if gate_logits is not None:
        hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
        gates = (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        hidden_states = hidden_states * gates
        hidden_states = hidden_states.flatten(2, 3)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)


def _slice_refiner_rotary_emb(
    rotary_emb: tuple[Tensor, Tensor],
    start: int,
    end: int,
) -> tuple[Tensor, Tensor]:
    cos, sin = rotary_emb
    if cos.ndim == 4:
        return cos[:, :, start:end], sin[:, :, start:end]
    if cos.ndim >= 3:
        return cos[:, start:end], sin[:, start:end]
    return cos[start:end], sin[start:end]


def _refiner_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    hidden_states = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
    )
    return hidden_states.transpose(1, 2)


def _pack_text_embeds(
    text_hidden_states: Tensor,
    sequence_lengths: Tensor,
    *,
    device: torch.device | str,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> Tensor:
    batch_size, seq_len, hidden_dim, _ = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype
    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(
            f"padding_side must be 'left' or 'right', got {padding_side!r}."
        )
    mask = mask[:, :, None, None]

    masked = text_hidden_states.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)
    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(
        dim=(1, 2),
        keepdim=True,
    )
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2),
        keepdim=True,
    )
    normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
    normalized = normalized * scale_factor
    normalized = normalized.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, normalized.shape[-1])
    normalized = normalized.masked_fill(~mask_flat, 0.0)
    return normalized.to(dtype=original_dtype)


def _pack_latents(
    latents: Tensor, patch_size: int = 1, patch_size_t: int = 1
) -> Tensor:
    batch_size, _, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.flatten(4, 7).flatten(1, 3)


def _unpack_latents(
    latents: Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    patch_size: int = 1,
    patch_size_t: int = 1,
) -> Tensor:
    batch_size = latents.size(0)
    latents = latents.reshape(
        batch_size,
        num_frames,
        height,
        width,
        -1,
        patch_size_t,
        patch_size,
        patch_size,
    )
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return latents.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def _prepare_encoder_attention_mask(
    mask: Tensor | None,
    dtype: torch.dtype,
) -> Tensor | None:
    if mask is None:
        return None
    if mask.ndim != 2:
        return mask
    if not torch.compiler.is_compiling() and bool(torch.all(mask)):
        return None
    return ((1 - mask.to(dtype)) * -10000.0).unsqueeze(1)


def _refiner_context_cache_enabled() -> bool:
    value = os.environ.get("SANA_WM_REFINER_CONTEXT_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}
