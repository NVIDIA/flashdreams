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

"""Crash-safe MiniMax H3 FL2VA pipeline for the FlashDreams runtime."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from loguru import logger
from torch import nn

from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from minimax_h3.constants import MODEL_ID, align_num_frames, validate_canvas
from minimax_h3.lora import load_lora
from minimax_h3.model import (
    MiniMaxH3DenoiseState,
    MiniMaxH3DiffusionModelConfig,
)
from minimax_h3.references import MiniMaxH3ReferenceSpec, load_references
from minimax_h3.transformer import MiniMaxH3TransformerConfig

MiniMaxH3Workflow = Literal["t2va", "fl2va", "ref2va"]


@dataclass(kw_only=True)
class MiniMaxH3PipelineConfig(StreamInferencePipelineConfig):
    """Config for the FlashDreams-native H3 denoising pipeline."""

    _target: type[MiniMaxH3Pipeline] = field(default_factory=lambda: MiniMaxH3Pipeline)

    diffusion_model: MiniMaxH3DiffusionModelConfig = field(
        default_factory=MiniMaxH3DiffusionModelConfig
    )
    """Native joint transformer and paired scheduler configuration."""

    model_id: str = MODEL_ID
    """Hugging Face model repository or local snapshot path."""

    cache_dir: Path | None = None
    """Optional Hugging Face model cache root."""

    workflow: MiniMaxH3Workflow = "fl2va"
    """Released checkpoint workflow selected by this registered pipeline."""


@dataclass(frozen=True)
class _ReferenceLayout:
    """Reference properties needed after its encoded media is checkpointed."""

    kind: str
    has_audio: bool


@dataclass(kw_only=True)
class MiniMaxH3PipelineCache:
    """Per-rollout H3 request, checkpoints, and runtime metrics."""

    prompt: str
    workflow: MiniMaxH3Workflow
    image_path: Path | None
    last_image_path: Path | None
    references: tuple[MiniMaxH3ReferenceSpec, ...]
    output_path: Path
    width: int
    height: int
    duration: float
    steps: int
    seed: int
    low_ram: bool
    restart: bool
    attention: str
    lora: str | None
    lora_weight_name: str | None
    lora_scale: float
    latent_checkpoint: Path
    conditioning_checkpoint: Path
    generated: bool = False
    elapsed_seconds: float = 0.0
    conditioning_seconds: float = 0.0
    denoise_seconds: float = 0.0
    denoise_prepare_seconds: float = 0.0
    transformer_load_seconds: float = 0.0
    denoise_compute_seconds: float = 0.0
    denoise_cleanup_seconds: float = 0.0
    latent_checkpoint_seconds: float = 0.0
    latent_checkpoint_future: Future[float] | None = field(default=None, repr=False)
    decode_seconds: float = 0.0
    peak_gpu_memory_gib: float = 0.0
    attention_backend: str = "default"
    resumed_stage: str | None = None


def _replace_blocks(pipe: Any, names: tuple[str, ...]) -> None:
    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks

    selected: dict[str, Any] = {}
    for requested in names:
        if requested in pipe._blocks.sub_blocks:
            selected[requested] = pipe._blocks.sub_blocks[requested]
            continue
        prefix = requested + "."
        for actual, block in pipe._blocks.sub_blocks.items():
            if actual.startswith(prefix):
                selected[actual.removeprefix(prefix)] = block
    if not selected:
        raise KeyError(f"none of the requested pipeline stages exist: {names}")
    pipe._blocks = SequentialPipelineBlocks.from_blocks_dict(selected)


def _release_pipeline(pipe: Any) -> None:
    for component in pipe.components.values():
        if isinstance(component, nn.Module):
            try:
                component.to_empty(device="cpu")
            except (AttributeError, RuntimeError):
                pass
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _write_status(checkpoint: Path, stage: str, **details: Any) -> None:
    _atomic_json(
        checkpoint.with_suffix(checkpoint.suffix + ".status.json"),
        {
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **details,
        },
    )


def _file_identity(path: Path | None) -> dict[str, str | int] | None:
    if path is None:
        return None
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _conditioning_manifest(
    cache: MiniMaxH3PipelineCache, model_id: str
) -> dict[str, Any]:
    return {
        "workflow": cache.workflow,
        "prompt": cache.prompt,
        "image": _file_identity(cache.image_path),
        "last_image": _file_identity(cache.last_image_path),
        "references": [reference.manifest() for reference in cache.references],
        "width": cache.width,
        "height": cache.height,
        "duration": cache.duration,
        "model_id": model_id,
    }


def _generation_manifest(
    cache: MiniMaxH3PipelineCache, model_id: str
) -> dict[str, Any]:
    return {
        **_conditioning_manifest(cache, model_id),
        "steps": cache.steps,
        "seed": cache.seed,
        "attention": cache.attention,
        "lora": cache.lora,
        "lora_weight_name": cache.lora_weight_name,
        "lora_scale": cache.lora_scale,
    }


def _signature(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _save_conditioning(
    cache: MiniMaxH3PipelineCache,
    model_id: str,
    values: dict[str, Any],
) -> None:
    from safetensors.torch import save_file

    manifest = _conditioning_manifest(cache, model_id)
    path = cache.conditioning_checkpoint
    temporary = path.with_name(path.name + ".tmp")
    condition_latents = values["condition_latents"]
    audio_condition_latents = values["audio_condition_latents"]
    tensors = {
        "prompt_embeds": values["prompt_embeds"].detach().cpu().contiguous(),
        "text_token_tags": values["text_token_tags"].detach().cpu().contiguous(),
        **{
            f"condition_latents.{index}": latent.detach().cpu().contiguous()
            for index, latent in enumerate(condition_latents)
        },
        **{
            f"audio_condition_latents.{index}": latent.detach().cpu().contiguous()
            for index, latent in enumerate(audio_condition_latents)
        },
    }
    save_file(
        tensors,
        str(temporary),
        metadata={
            "stage": "conditioned",
            "manifest": json.dumps(manifest, sort_keys=True),
            "signature": _signature(manifest),
            "height": str(values["height"]),
            "width": str(values["width"]),
            "num_frames": str(values["num_frames"]),
            "keyframe_anchors": json.dumps(list(values["keyframe_anchors"])),
            "condition_count": str(len(condition_latents)),
            "audio_condition_count": str(len(audio_condition_latents)),
            "reference_layout": json.dumps(
                [
                    {"kind": reference.kind, "has_audio": reference.has_audio}
                    for reference in values["normalized_references"]
                ]
            ),
        },
    )
    os.replace(temporary, path)
    _write_status(path, "conditioned")


def _load_conditioning(cache: MiniMaxH3PipelineCache, model_id: str) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = cache.conditioning_checkpoint
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = _signature(_conditioning_manifest(cache, model_id))
    if metadata.get("stage") != "conditioned" or metadata.get("signature") != expected:
        raise ValueError(
            f"conditioning checkpoint does not match this request: {path}; "
            "use --restart"
        )
    tensors = load_file(path, device="cpu")
    condition_count = int(metadata["condition_count"])
    audio_condition_count = int(metadata.get("audio_condition_count", "0"))
    reference_layout = json.loads(metadata.get("reference_layout", "[]"))
    return {
        "prompt_embeds": tensors["prompt_embeds"],
        "text_token_tags": tensors["text_token_tags"],
        "condition_latents": [
            tensors[f"condition_latents.{index}"] for index in range(condition_count)
        ],
        "audio_condition_latents": [
            tensors[f"audio_condition_latents.{index}"]
            for index in range(audio_condition_count)
        ],
        "normalized_references": [
            _ReferenceLayout(kind=reference["kind"], has_audio=reference["has_audio"])
            for reference in reference_layout
        ],
        "height": int(metadata["height"]),
        "width": int(metadata["width"]),
        "num_frames": int(metadata["num_frames"]),
        "keyframe_anchors": tuple(json.loads(metadata["keyframe_anchors"])),
    }


def _save_latents(
    cache: MiniMaxH3PipelineCache,
    model_id: str,
    latents: torch.Tensor,
) -> None:
    from safetensors.torch import save_file

    path = cache.latent_checkpoint
    temporary = path.with_name(path.name + ".tmp")
    manifest = _generation_manifest(cache, model_id)
    save_file(
        {"latents": latents.detach().cpu().contiguous()},
        str(temporary),
        metadata={
            "stage": "denoised",
            "manifest": json.dumps(manifest, sort_keys=True),
            "signature": _signature(manifest),
        },
    )
    os.replace(temporary, path)
    _write_status(path, "denoised")


def _save_latents_async(
    cache: MiniMaxH3PipelineCache,
    model_id: str,
    latents: torch.Tensor,
) -> Future[float]:
    """Persist the recovery checkpoint without blocking video decoding."""
    future: Future[float] = Future()

    def persist() -> None:
        if not future.set_running_or_notify_cancel():
            return
        started = time.monotonic()
        try:
            _save_latents(cache, model_id, latents)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(time.monotonic() - started)

    threading.Thread(
        target=persist,
        name="minimax-h3-latent-checkpoint",
        daemon=False,
    ).start()
    return future


def _finish_latent_checkpoint(cache: MiniMaxH3PipelineCache, *, wait: bool) -> None:
    """Collect a completed checkpoint write, optionally waiting for it."""
    future = cache.latent_checkpoint_future
    if future is None or (not wait and not future.done()):
        return
    cache.latent_checkpoint_seconds = future.result()
    cache.latent_checkpoint_future = None


def _load_latents(cache: MiniMaxH3PipelineCache, model_id: str) -> torch.Tensor:
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = cache.latent_checkpoint
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = _signature(_generation_manifest(cache, model_id))
    if metadata.get("stage") != "denoised" or metadata.get("signature") != expected:
        raise ValueError(
            f"latent checkpoint does not match this request: {path}; use --restart"
        )
    tensors = load_file(path, device="cpu")
    return tensors["latents"]


class MiniMaxH3Pipeline(StreamInferencePipeline[Any, Any, Any]):
    """FlashDreams H3 runtime with staged third-party conditioning and decode."""

    config: MiniMaxH3PipelineConfig

    def __init__(self, config: MiniMaxH3PipelineConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def device(self) -> torch.device:
        return cast(torch.Tensor, self._device_anchor).device

    def initialize_cache(
        self,
        *,
        prompt: str,
        image_path: Path | None,
        last_image_path: Path | None,
        references: tuple[MiniMaxH3ReferenceSpec, ...],
        output_path: Path,
        width: int,
        height: int,
        duration: float,
        steps: int,
        seed: int,
        low_ram: bool,
        restart: bool,
        attention: str,
        lora: str | None,
        lora_weight_name: str | None,
        lora_scale: float,
    ) -> MiniMaxH3PipelineCache:
        """Build a validated, checkpoint-aware workflow cache."""
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        workflow = self.config.workflow
        if workflow == "t2va" and (
            image_path is not None or last_image_path is not None or references
        ):
            raise ValueError("t2va does not accept keyframes or references")
        if workflow == "fl2va" and not (image_path or last_image_path):
            raise ValueError("fl2va requires --image-path and/or --last-image-path")
        if workflow == "ref2va" and not references:
            raise ValueError("ref2va requires at least one --reference")
        if workflow != "ref2va" and references:
            raise ValueError(f"{workflow} does not accept ordered references")
        for label, path in (
            ("first-frame", image_path),
            ("last-frame", last_image_path),
        ):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f"{label} image not found: {path}")
        if steps < 2:
            raise ValueError("steps must be at least 2 scheduler points")
        if attention not in {"auto", "flash", "default"}:
            raise ValueError(f"unsupported attention backend: {attention}")
        if not 0 <= lora_scale <= 4:
            raise ValueError("LoRA scale must be between 0 and 4")
        validate_canvas(width, height)
        align_num_frames(duration)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latent_checkpoint = output_path.with_suffix(
            output_path.suffix + ".latents.safetensors"
        )
        conditioning_checkpoint = output_path.with_suffix(
            output_path.suffix + ".conditioning.safetensors"
        )
        return MiniMaxH3PipelineCache(
            prompt=prompt,
            workflow=workflow,
            image_path=image_path,
            last_image_path=last_image_path,
            references=references,
            output_path=output_path,
            width=width,
            height=height,
            duration=duration,
            steps=steps,
            seed=seed,
            low_ram=low_ram,
            restart=restart,
            attention=attention,
            lora=lora,
            lora_weight_name=lora_weight_name,
            lora_scale=lora_scale,
            latent_checkpoint=latent_checkpoint,
            conditioning_checkpoint=conditioning_checkpoint,
        )

    @torch.no_grad()
    def generate(
        self,
        autoregressive_index: int,
        cache: MiniMaxH3PipelineCache,
        input: Any = None,
    ) -> torch.Tensor:
        """Generate and video-decode the single non-streaming H3 step."""
        del input
        if autoregressive_index != 0 or cache.generated:
            raise ValueError("MiniMax H3 supports exactly one runtime step per cache")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()

        if cache.latent_checkpoint.is_file() and not cache.restart:
            cache.resumed_stage = "decode"
            latents = _load_latents(cache, self.config.model_id)
        else:
            if cache.low_ram:
                latents = self._generate_low_ram(cache)
            else:
                latents = self._generate_standard(cache)
            cache.latent_checkpoint_future = _save_latents_async(
                cache, self.config.model_id, latents
            )

        decode_started = time.monotonic()
        try:
            frames = self._decode_video(cache, latents)
        except BaseException:
            try:
                _finish_latent_checkpoint(cache, wait=True)
            except BaseException:
                logger.exception("Latent checkpoint also failed after decode failure")
            raise
        cache.decode_seconds = time.monotonic() - decode_started
        cache.elapsed_seconds = time.monotonic() - started
        cache.peak_gpu_memory_gib = torch.cuda.max_memory_allocated() / 2**30
        cache.generated = True
        return frames

    def mark_complete(self, cache: MiniMaxH3PipelineCache) -> None:
        """Record completion only after the runtime output target closes."""
        if not cache.generated or not cache.output_path.is_file():
            raise RuntimeError("cannot complete H3 job before its MP4 is written")
        _finish_latent_checkpoint(cache, wait=True)
        _write_status(
            cache.latent_checkpoint, "complete", output=str(cache.output_path)
        )

    def finalize(
        self,
        autoregressive_index: int,
        cache: MiniMaxH3PipelineCache,
    ) -> dict[str, float]:
        """Return runtime metrics for the completed H3 rollout."""
        if autoregressive_index != 0 or not cache.generated:
            raise ValueError("finalize requires the completed H3 runtime step")
        _finish_latent_checkpoint(cache, wait=False)
        return {
            "conditioning_seconds": cache.conditioning_seconds,
            "denoise_seconds": cache.denoise_seconds,
            "denoise_prepare_seconds": cache.denoise_prepare_seconds,
            "transformer_load_seconds": cache.transformer_load_seconds,
            "denoise_compute_seconds": cache.denoise_compute_seconds,
            "denoise_cleanup_seconds": cache.denoise_cleanup_seconds,
            "latent_checkpoint_seconds": cache.latent_checkpoint_seconds,
            "decode_seconds": cache.decode_seconds,
            "total_seconds": cache.elapsed_seconds,
            "peak_gpu_memory_gib": cache.peak_gpu_memory_gib,
        }

    def _cache_dir(self) -> str | None:
        return str(self.config.cache_dir) if self.config.cache_dir is not None else None

    def _apply_lora(self, transformer: Any, cache: MiniMaxH3PipelineCache) -> None:
        if cache.lora is None:
            return
        converted = load_lora(
            transformer,
            cache.lora,
            cache.lora_scale,
            cache.lora_weight_name,
        )
        logger.info("Loaded LoRA {} at scale {:g}", converted, cache.lora_scale)

    def _generate_low_ram(self, cache: MiniMaxH3PipelineCache) -> torch.Tensor:
        if cache.conditioning_checkpoint.is_file() and not cache.restart:
            cache.resumed_stage = "denoise"
            conditioned = _load_conditioning(cache, self.config.model_id)
        else:
            conditioning_started = time.monotonic()
            _write_status(cache.latent_checkpoint, "conditioning")
            conditioned = self._condition(cache)
            _save_conditioning(cache, self.config.model_id, conditioned)
            cache.conditioning_seconds = time.monotonic() - conditioning_started
        return self._run_native_denoise(cache, conditioned)

    def _condition(self, cache: MiniMaxH3PipelineCache) -> dict[str, Any]:
        from diffusers.utils import load_image

        num_frames = align_num_frames(cache.duration)
        media: dict[str, Any]
        if cache.workflow == "t2va":
            media = {
                "height": cache.height,
                "width": cache.width,
                "num_frames": num_frames,
                "keyframe_anchors": (),
                "normalized_references": [],
            }
            text_inputs = {"prompt": cache.prompt}
            encoded = {
                "condition_latents": [],
                "audio_condition_latents": [],
            }
        elif cache.workflow == "fl2va":
            resize_inputs: dict[str, Any] = {
                "height": cache.height,
                "width": cache.width,
            }
            if cache.image_path is not None:
                resize_inputs["image"] = load_image(str(cache.image_path))
            if cache.last_image_path is not None:
                resize_inputs["last_image"] = load_image(str(cache.last_image_path))
            media = self._run_conditioning_stage(
                cache.workflow,
                ("before_encode",),
                resize_inputs,
                ["height", "width", "keyframes", "keyframe_anchors"],
            )
            media["num_frames"] = num_frames
            media["normalized_references"] = []
            text_inputs = {"prompt": cache.prompt, "keyframes": media["keyframes"]}
            encoded = self._run_conditioning_stage(
                cache.workflow,
                ("vae_encoder",),
                {"keyframes": media["keyframes"]},
                ["condition_latents"],
                cuda_components=("vae",),
            )
            encoded["audio_condition_latents"] = []
        else:
            references = load_references(cache.references)
            media = self._run_conditioning_stage(
                cache.workflow,
                ("before_encode",),
                {
                    "references": references,
                    "height": cache.height,
                    "width": cache.width,
                    "num_frames": num_frames,
                },
                ["height", "width", "num_frames", "normalized_references"],
            )
            media["keyframe_anchors"] = ()
            text_inputs = {
                "prompt": cache.prompt,
                "normalized_references": media["normalized_references"],
            }
            encoded = self._run_conditioning_stage(
                cache.workflow,
                ("vae_encoder",),
                {"normalized_references": media["normalized_references"]},
                ["condition_latents", "audio_condition_latents"],
                cuda_components=("vae", "audio_vae"),
            )

        text = self._run_conditioning_stage(
            cache.workflow,
            ("text_encoder",),
            text_inputs,
            ["prompt_embeds", "text_token_tags"],
            cuda_components=("text_encoder",),
        )
        return {
            **text,
            **encoded,
            "height": media["height"],
            "width": media["width"],
            "keyframe_anchors": media["keyframe_anchors"],
            "normalized_references": media["normalized_references"],
            "num_frames": media["num_frames"],
        }

    def _run_conditioning_stage(
        self,
        workflow: MiniMaxH3Workflow,
        blocks: tuple[str, ...],
        inputs: dict[str, Any],
        outputs: list[str],
        *,
        cuda_components: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

        pipe = ModularPipeline.from_pretrained(
            self.config.model_id,
            workflow=workflow,
            cache_dir=self._cache_dir(),
        )
        _replace_blocks(pipe, blocks)
        required = list(
            dict.fromkeys(spec.name for spec in pipe._blocks.expected_components)
        )
        cpu_components = [name for name in required if name not in cuda_components]
        pipe.load_components(names=cpu_components, dtype=torch.bfloat16)
        for name in cuda_components:
            pipe.load_components(
                names=[name],
                dtype=torch.bfloat16,
                device_map="cuda",
                low_cpu_mem_usage=True,
            )
        try:
            return dict(pipe(**inputs, output=outputs))
        finally:
            _release_pipeline(pipe)

    def _build_prepare_pipeline(self, workflow: MiniMaxH3Workflow) -> Any:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

        pipe = ModularPipeline.from_pretrained(
            self.config.model_id,
            workflow=workflow,
            cache_dir=self._cache_dir(),
        )
        blocks = {
            "t2va": (
                "denoise.no_keyframe_anchors",
                "denoise.prepare_layout",
                "denoise.prepare_latents",
            ),
            "fl2va": (
                "denoise.prepare_layout",
                "denoise.prepare_condition_latents",
                "denoise.prepare_latents",
                "denoise.prepare_latents_fl2va",
            ),
            "ref2va": (
                "denoise.prepare_layout",
                "denoise.prepare_condition_latents",
                "denoise.prepare_latents",
                "denoise.prepare_latents_ref2va",
            ),
        }[workflow]
        _replace_blocks(pipe, blocks)
        if workflow != "t2va":
            pipe.load_components(names=["scheduler"], dtype=torch.bfloat16)
        return pipe

    def _prepare_denoise_state(
        self,
        cache: MiniMaxH3PipelineCache,
        conditioned: dict[str, Any],
    ) -> MiniMaxH3DenoiseState:
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        pipe = self._build_prepare_pipeline(cache.workflow)
        state = PipelineState()
        for name, value in conditioned.items():
            state.set(name, value)
        state.set("generator", torch.Generator(device="cpu").manual_seed(cache.seed))
        fields = [
            "latents",
            "audio_latents",
            "prompt_embeds",
            "position_ids",
            "token_tags",
            "video_indices",
            "audio_indices",
            "text_indices",
            "num_condition_video_rows",
            "num_condition_audio_rows",
            "num_latent_frames",
            "latent_height",
            "latent_width",
        ]
        try:
            results = pipe(state=state, output=fields)
        finally:
            _release_pipeline(pipe)
        return MiniMaxH3DenoiseState(**results)

    def _run_native_denoise(
        self, cache: MiniMaxH3PipelineCache, conditioned: dict[str, Any]
    ) -> torch.Tensor:
        denoise_started = time.monotonic()
        _write_status(cache.latent_checkpoint, "denoising-native-flashdreams")
        prepare_started = time.monotonic()
        state = self._prepare_denoise_state(cache, conditioned)
        cache.denoise_prepare_seconds = time.monotonic() - prepare_started
        backend = "cudnn" if cache.attention == "default" else "flash"
        cache.attention_backend = backend
        base_transformer = cast(
            MiniMaxH3TransformerConfig, self.config.diffusion_model.transformer
        )
        transformer_config = replace(
            base_transformer,
            attention_backend=backend,
            device="cuda",
            execution_device="cuda",
            sequential_cpu_offload=False,
        )
        model_config = replace(
            self.config.diffusion_model,
            transformer=transformer_config,
            scheduler=replace(
                self.config.diffusion_model.scheduler,
                num_inference_steps=cache.steps,
            ),
            audio_scheduler=replace(
                self.config.diffusion_model.audio_scheduler,
                num_inference_steps=cache.steps,
            ),
            seed=cache.seed,
        )
        transformer_load_started = time.monotonic()
        model = model_config.setup()
        cache.transformer_load_seconds = time.monotonic() - transformer_load_started
        try:
            self._apply_lora(model.transformer, cache)
            compute_started = time.monotonic()
            latents = model.generate_joint(state)
            if latents.is_cuda:
                torch.cuda.synchronize(latents.device)
            cache.denoise_compute_seconds = time.monotonic() - compute_started
        finally:
            cleanup_started = time.monotonic()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            cache.denoise_cleanup_seconds = time.monotonic() - cleanup_started
        cache.denoise_seconds = time.monotonic() - denoise_started
        return latents

    def _generate_standard(self, cache: MiniMaxH3PipelineCache) -> torch.Tensor:
        conditioning_started = time.monotonic()
        conditioned = self._condition(cache)
        cache.conditioning_seconds = time.monotonic() - conditioning_started
        return self._run_native_denoise(cache, conditioned)

    def _decode_video(
        self, cache: MiniMaxH3PipelineCache, latents: torch.Tensor
    ) -> torch.Tensor:
        from diffusers.modular_pipelines.modular_pipeline import (
            ModularPipeline,
            PipelineState,
            SequentialPipelineBlocks,
        )

        _write_status(cache.latent_checkpoint, "decoding-video")
        pipe = ModularPipeline.from_pretrained(
            self.config.model_id,
            workflow=cache.workflow,
            cache_dir=self._cache_dir(),
        )
        video_block = pipe._blocks.sub_blocks.get("decode.video")
        if video_block is None:
            raise RuntimeError("MiniMax H3 workflow has no video decode block")
        pipe._blocks = SequentialPipelineBlocks.from_blocks_dict({"video": video_block})
        pipe.load_components(
            names=["vae", "video_processor"],
            dtype={"vae": torch.float32},
        )
        pipe.vae.encoder.to_empty(device="cpu")
        pipe.vae.quant_conv.to_empty(device="cpu")
        pipe.vae.post_quant_conv.to("cuda")
        pipe.vae.decoder.to("cuda")
        first_encoder_parameter = next(pipe.vae.encoder.parameters())
        first_encoder_parameter.data = torch.empty_like(
            first_encoder_parameter, device="cuda"
        )

        state = PipelineState()
        state.set("latents", latents.to("cuda"))
        state.set("output_type", "np")
        results = pipe(state=state, output=["videos"])
        video = np.asarray(results["videos"][0])
        frames = torch.from_numpy(video).permute(0, 3, 1, 2).float().mul(2).sub(1)
        _release_pipeline(pipe)
        return frames.contiguous()
