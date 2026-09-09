# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged MiniMax H3 inference composed from FlashDreams sampling and codecs."""

from dataclasses import dataclass, field, fields, replace
from functools import partial
from pathlib import Path
import time
from typing import Any, Literal

import torch
from torch import Tensor, nn

from flashdreams.infra.acceleration import (
    collect_and_release_cuda_memory,
    run_one_shot_stage,
)
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.diffusion.scheduler import (
    DataFlowEulerSchedulerConfig,
    sample_synchronized,
)
from minimax_h3.impl.constants import (
    MODEL_ID,
    MODEL_REVISION,
    align_num_frames,
    validate_canvas,
)
from minimax_h3.impl.transformer import MiniMaxH3TransformerConfig
from minimax_h3.impl.weights import MiniMaxH3Weights


@dataclass(kw_only=True)
class MiniMaxH3PipelineConfig(InstantiateConfig):
    """Native joint model with stage-scoped weight residency."""

    _target: type["MiniMaxH3Pipeline"] = field(
        default_factory=lambda: MiniMaxH3Pipeline
    )
    name: str = "minimax-h3-t2va"
    workflow: Literal["t2va", "fl2va", "ref2va"] = "t2va"
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    transformer: MiniMaxH3TransformerConfig = field(
        default_factory=MiniMaxH3TransformerConfig
    )
    scheduler: DataFlowEulerSchedulerConfig = field(
        default_factory=DataFlowEulerSchedulerConfig
    )
    audio_scheduler: DataFlowEulerSchedulerConfig = field(
        default_factory=lambda: DataFlowEulerSchedulerConfig(shift=3.0)
    )
    seed: int = 42
    compile_network: bool = False
    """Compile only the joint transformer forward, never stage loading or I/O."""


@dataclass(kw_only=True)
class MiniMaxH3Request:
    """Validated request and in-memory state for one video-only rollout."""

    prompt: str
    width: int
    height: int
    num_frames: int
    image_path: Path | None = None
    last_image_path: Path | None = None
    references: tuple = ()
    lora: str | None = None
    lora_weight_name: str | None = None
    lora_scale: float = 1.0
    generated: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class MiniMaxH3Decoder(nn.Module):
    """Lazy native video decoder with fixed spatial layout metadata."""

    spatial_compression_ratio = 16

    def __init__(self, weights: MiniMaxH3Weights) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, latents: Tensor) -> Tensor:
        """Decode normalized latents to TCHW frames in ``[-1, 1]``."""
        latents = latents.float()
        decoder = self.weights.video_decoder(latents.device)
        try:
            mean = latents.new_tensor(decoder.config.latents_mean).view(1, -1, 1, 1, 1)
            std = latents.new_tensor(decoder.config.latents_std).view(1, -1, 1, 1, 1)
            pixels = decoder.decode(latents * std + mean).float()
            pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
            pixel_std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
            return (
                (pixels * pixel_std + pixel_mean)
                .clamp(0, 1)[0]
                .permute(1, 0, 2, 3)
                .mul(2)
                .sub(1)
                .contiguous()
            )
        finally:
            del decoder
            collect_and_release_cuda_memory(device=latents.device)


class MiniMaxH3Pipeline(nn.Module):
    """One-block v2 pipeline; no legacy runner or third-party orchestration."""

    def __init__(self, config: MiniMaxH3PipelineConfig) -> None:
        super().__init__()
        self.config = config
        self.weights = MiniMaxH3Weights(config.model_id, config.revision)
        self.decoder = MiniMaxH3Decoder(self.weights)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def initialize_cache(
        self,
        *,
        text: list[str],
        height: int,
        width: int,
        image: Any = None,
        duration: float = 5.0,
        image_path: Path | None = None,
        last_image_path: Path | None = None,
        references: tuple = (),
        lora: str | None = None,
        lora_weight_name: str | None = None,
        lora_scale: float = 1.0,
    ) -> MiniMaxH3Request:
        """Validate inputs without loading checkpoints or encoding on the UI thread."""
        if len(text) != 1 or not isinstance(text[0], str) or not text[0].strip():
            raise ValueError("H3 requires exactly one nonempty prompt")
        if image is not None:
            raise ValueError("Use image_path/last_image_path for H3 keyframes")
        height, width = height * 16, width * 16
        validate_canvas(width, height)
        workflow = self.config.workflow
        if workflow not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError(f"Unsupported workflow: {workflow}")
        if workflow == "t2va" and (image_path or last_image_path or references):
            raise ValueError("t2va does not accept keyframes or references")
        if workflow == "fl2va" and not (image_path or last_image_path):
            raise ValueError("fl2va requires a first and/or last image")
        if workflow == "ref2va" and not references:
            raise ValueError("ref2va requires ordered references")
        if workflow != "ref2va" and references:
            raise ValueError(f"{workflow} does not accept references")
        if workflow == "ref2va" and (image_path or last_image_path):
            raise ValueError("ref2va accepts references rather than keyframes")
        for path in (image_path, last_image_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        if not 0 <= lora_scale <= 4:
            raise ValueError("LoRA scale must be between 0 and 4")
        return MiniMaxH3Request(
            prompt=text[0],
            width=width,
            height=height,
            num_frames=align_num_frames(duration),
            image_path=image_path,
            last_image_path=last_image_path,
            references=references,
            lora=lora,
            lora_weight_name=lora_weight_name,
            lora_scale=lora_scale,
        )

    def _condition(self, request: MiniMaxH3Request) -> dict:
        from flashdreams.infra.encoder.text.qwen3_vl import Qwen3VLEncoderConfig
        from minimax_h3.impl.conditioning import condition_request

        return condition_request(
            prompt=request.prompt,
            workflow=self.config.workflow,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            image_path=request.image_path,
            last_image_path=request.last_image_path,
            references=request.references,
            device=self.device,
            video_encoder_factory=lambda: self.weights.video_encoder(self.device),
            audio_encoder_factory=lambda: self.weights.audio_encoder(self.device),
            qwen_encoder_factory=lambda: Qwen3VLEncoderConfig(
                model_name=self.config.model_id,
                revision=self.config.revision,
                loading_device=str(self.device),
            )
            .setup()
            .eval(),
        )

    def _denoise(self, request: MiniMaxH3Request, conditioned: dict) -> Tensor:
        from minimax_h3.impl.layout import prepare_denoise_state
        from minimax_h3.impl.lora import load_lora

        state = prepare_denoise_state(
            conditioned, self.config.seed, self.config.workflow
        )
        state = replace(
            state,
            **{
                item.name: getattr(state, item.name).to(self.device)
                for item in fields(state)
                if isinstance(getattr(state, item.name), Tensor)
            },
        )
        component = (
            "transformer_ref" if self.config.workflow == "ref2va" else "transformer"
        )
        network = replace(
            self.config.transformer,
            device=str(self.device),
            checkpoint_path=self.weights.checkpoint(component),
        ).setup()
        try:
            if request.lora is not None:
                load_lora(
                    network, request.lora, request.lora_scale, request.lora_weight_name
                )
                network.refresh_derived_weights()
            predictor = (
                compile_module(network) if self.config.compile_network else network
            )
            schedulers = (
                self.config.scheduler.setup().to(self.device),
                self.config.audio_scheduler.setup().to(self.device),
            )
            nv, na = state.num_condition_video_rows, state.num_condition_audio_rows
            video_condition, audio_condition = (
                state.latents[:nv],
                state.audio_latents[:na],
            )

            def predict(samples, times):
                video, audio = samples
                video_time, audio_time = times
                row_times = video_time.expand(state.position_ids.shape[0]).clone()
                row_times[state.video_indices[:nv]] = video_time.clamp_min(0.999)
                row_times[state.audio_indices[:na]] = 1.0
                row_times[state.audio_indices[na:]] = audio_time
                timestep, timestep_indices = torch.unique(
                    row_times, sorted=True, return_inverse=True
                )
                video_flow, audio_flow = predictor(
                    hidden_states=torch.cat((video_condition, video))[None],
                    audio_hidden_states=torch.cat((audio_condition, audio))[None],
                    encoder_hidden_states=state.prompt_embeds,
                    timestep=timestep,
                    timestep_indices=timestep_indices,
                    token_tags=state.token_tags,
                    position_ids=state.position_ids,
                    video_indices=state.video_indices,
                    audio_indices=state.audio_indices,
                    text_indices=state.text_indices,
                )
                return video_flow[0, nv:].float(), audio_flow[0, na:].float()

            video, _ = sample_synchronized(
                (state.latents[nv:], state.audio_latents[na:]), schedulers, predict
            )
            pt, ph, pw = self.config.transformer.patch_size
            channels = self.config.transformer.in_channels
            rows = video.reshape(
                1,
                state.num_latent_frames // pt,
                state.latent_height // ph,
                state.latent_width // pw,
                channels,
                pt,
                ph,
                pw,
            )
            return (
                rows.permute(0, 4, 1, 5, 2, 6, 3, 7)
                .reshape(
                    1,
                    channels,
                    state.num_latent_frames,
                    state.latent_height,
                    state.latent_width,
                )
                .contiguous()
            )
        finally:
            # Closures/compiled wrappers must release their network before collection.
            if "predictor" in locals():
                del predictor
            del network
            collect_and_release_cuda_memory(device=self.device)

    @torch.no_grad()
    def generate(self, autoregressive_index: int, cache: MiniMaxH3Request) -> Tensor:
        """Condition, jointly sample, and decode one full video on the model thread."""
        if autoregressive_index != 0 or cache.generated:
            raise ValueError("H3 supports exactly one generation block per request")
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        conditioning = self._condition(cache)
        self._record_stage(cache, "conditioning_seconds", started)
        denoise_started = time.perf_counter()
        latents = run_one_shot_stage(
            partial(self._denoise, cache, conditioning), cpu_result=False
        )
        del conditioning
        self._record_stage(cache, "denoise_seconds", denoise_started)
        decode_started = time.perf_counter()
        frames = run_one_shot_stage(lambda: self.decoder(latents), cpu_result=False)
        self._record_stage(cache, "decode_seconds", decode_started)
        cache.metrics["total_seconds"] = time.perf_counter() - started
        if self.device.type == "cuda":
            cache.metrics["peak_gpu_memory_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
            )
        cache.generated = True
        return frames

    def _record_stage(
        self, request: MiniMaxH3Request, name: str, started: float
    ) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        request.metrics[name] = time.perf_counter() - started

    def finalize(
        self, autoregressive_index: int, cache: MiniMaxH3Request
    ) -> dict[str, float]:
        """Report completed-stage timings without a redundant model evaluation."""
        if autoregressive_index != 0 or not cache.generated:
            raise ValueError("finalize requires a completed H3 request")
        return dict(cache.metrics)

    def close(self) -> None:
        """Release reclaimed stage memory when the application closes."""
        collect_and_release_cuda_memory(device=self.device)
