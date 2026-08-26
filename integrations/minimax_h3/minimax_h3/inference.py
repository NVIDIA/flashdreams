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

"""Staged, Diffusers-free MiniMax H3 inference over native components."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import torch
from loguru import logger
from PIL import Image
from torch import Tensor, nn

from flashdreams.runtime_v2.audio_output import AudioOutput
from minimax_h3.audio_vae import MiniMaxH3AudioVAEConfig
from minimax_h3.conditioning import (
    audio_latent_num_frames,
    prepare_denoise_state,
    prepare_ref2va_denoise_state,
)
from minimax_h3.constants import (
    AUDIO_SAMPLE_RATE,
    FPS,
    MODEL_ID,
    align_num_frames,
    validate_canvas,
)
from minimax_h3.keyframes import encode_keyframes, prepare_keyframes
from minimax_h3.latent_checkpoint import (
    MiniMaxH3AssetIdentity,
    MiniMaxH3CheckpointIdentity,
    MiniMaxH3LatentCheckpointStore,
)
from minimax_h3.lora import load_lora
from minimax_h3.model import (
    MiniMaxH3DiffusionModelConfig,
    MiniMaxH3DenoiseProgress,
    MiniMaxH3DenoiseState,
    MiniMaxH3JointLatents,
)
from minimax_h3.reference_conditioning import (
    MiniMaxH3EncodedReferences,
    MiniMaxH3Reference,
    encode_references,
    normalize_references,
)
from minimax_h3.scheduler import MiniMaxH3SchedulerConfig
from minimax_h3.text_encoder import (
    MiniMaxH3TextCondition,
    build_fl2va_presentation,
    build_ref2va_presentation,
    build_t2va_presentation,
    encode_presentation,
)
from minimax_h3.transformer import (
    H3_REF_TRANSFORMER_CHECKPOINT,
    H3_TRANSFORMER_CHECKPOINT,
    MiniMaxH3TransformerConfig,
)
from minimax_h3.video_vae import MiniMaxH3VideoVAEConfig

MiniMaxH3Workflow = Literal["t2va", "fl2va", "ref2va"]

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
"""Immutable MiniMax H3 model revision used by every native component."""

_MIN_FREE_GPU_GIB = 80.0
"""Staged transformer weight plus measured activation reserve."""

_MIN_FREE_CACHE_GIB = 150.0
"""Converted root components needed for one workflow at the pinned revision."""


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3InferenceConfig:
    """Immutable model and execution settings shared across H3 sessions."""

    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    device: str = "cuda"
    attention_backend: Literal["flash", "cudnn", "efficient", "math"] = "flash"
    cache_dir: Path | None = None
    checkpoint_min_free_gb: float | None = _MIN_FREE_CACHE_GIB

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.revision != MODEL_REVISION:
            raise ValueError(f"MiniMax H3 requires pinned revision {MODEL_REVISION}")
        if not self.device.strip():
            raise ValueError("device cannot be empty")
        if self.attention_backend not in ("flash", "cudnn", "efficient", "math"):
            raise ValueError(
                f"unsupported attention backend {self.attention_backend!r}"
            )
        if self.checkpoint_min_free_gb is not None and (
            isinstance(self.checkpoint_min_free_gb, bool)
            or not isinstance(self.checkpoint_min_free_gb, (int, float))
            or not math.isfinite(self.checkpoint_min_free_gb)
            or self.checkpoint_min_free_gb < 0
        ):
            raise ValueError("checkpoint_min_free_gb must be finite and non-negative")


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3InferenceRequest:
    """One fully decoded H3 request, independent of paths and output targets."""

    workflow: MiniMaxH3Workflow
    prompt: str
    width: int
    height: int
    duration: float = 5.0
    num_inference_steps: int = 30
    seed: int = 0
    first_image: Image.Image | None = None
    last_image: Image.Image | None = None
    references: tuple[MiniMaxH3Reference, ...] = ()
    checkpoint_store: MiniMaxH3LatentCheckpointStore | None = None
    checkpoint_inputs: tuple[MiniMaxH3AssetIdentity, ...] = ()
    restart: bool = False
    lora_path: Path | None = None
    lora_weight_name: str | None = None
    lora_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.workflow not in ("t2va", "fl2va", "ref2va"):
            raise ValueError(f"unsupported MiniMax H3 workflow {self.workflow!r}")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        validate_canvas(self.width, self.height)
        align_num_frames(self.duration)
        if type(self.num_inference_steps) is not int or self.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.first_image is not None and not isinstance(
            self.first_image, Image.Image
        ):
            raise ValueError("first_image must be a PIL image")
        if self.last_image is not None and not isinstance(
            self.last_image, Image.Image
        ):
            raise ValueError("last_image must be a PIL image")
        has_keyframes = self.first_image is not None or self.last_image is not None
        if self.workflow == "t2va" and (has_keyframes or self.references):
            raise ValueError("t2va does not accept keyframes or references")
        if self.workflow == "fl2va" and not has_keyframes:
            raise ValueError("fl2va requires a first image, a last image, or both")
        if self.workflow == "fl2va" and self.references:
            raise ValueError("fl2va does not accept ordered references")
        if self.workflow == "ref2va" and (has_keyframes or not self.references):
            raise ValueError("ref2va requires references and does not accept keyframes")
        if self.lora_path is None:
            if self.lora_weight_name is not None or self.lora_scale != 1.0:
                raise ValueError("LoRA options require lora_path")
        else:
            resolved = self.lora_path.expanduser().resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f"LoRA source is not a file: {resolved}")
            if self.lora_weight_name is not None:
                raise ValueError("lora_weight_name applies only to Hub LoRA sources")
            if (
                isinstance(self.lora_scale, bool)
                or not isinstance(self.lora_scale, (int, float))
                or not 0 <= self.lora_scale <= 4
            ):
                raise ValueError("lora_scale must be between 0 and 4")

    @property
    def num_frames(self) -> int:
        """Return the final ``17n+5`` frame count, including 362 at 15 s."""
        return align_num_frames(self.duration)


@dataclass(frozen=True, kw_only=True, slots=True)
class MiniMaxH3InferenceResult:
    """Decoded synchronized media plus rollout measurements."""

    video: Tensor
    """CPU ``[-1, 1]`` frames shaped ``[time, 3, height, width]``."""

    audio: AudioOutput
    """Decoded stereo 32 kHz audio at absolute sample offset zero."""

    metrics: dict[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.video.ndim != 4
            or self.video.shape[0] <= 0
            or self.video.shape[1] != 3
            or any(size <= 0 for size in self.video.shape[2:])
        ):
            raise ValueError("video must have shape [time, 3, height, width]")
        if self.video.device.type != "cpu" or self.video.dtype != torch.float32:
            raise ValueError("video must be CPU float32")
        if not self.video.is_contiguous() or not bool(torch.isfinite(self.video).all()):
            raise ValueError("video must be contiguous and finite")
        if bool((self.video < -1).any()) or bool((self.video > 1).any()):
            raise ValueError("video must stay within [-1, 1]")


class MiniMaxH3Resources(Protocol):
    """Factory surface for staged heavyweight H3 components."""

    tokenizer: Any
    processor: Any

    def load_text_encoder(self) -> Any: ...

    def load_video_vae(self) -> Any: ...

    def load_audio_vae(self) -> Any: ...

    def load_diffusion_model(
        self, workflow: MiniMaxH3Workflow, num_inference_steps: int
    ) -> Any: ...

    def release(self, module: Any) -> None: ...

    def close(self) -> None: ...


class DefaultMiniMaxH3Resources:
    """Load one pinned heavyweight component at a time on the execution GPU."""

    def __init__(self, config: MiniMaxH3InferenceConfig) -> None:
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoTokenizer,
            Qwen2VLImageProcessor,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        self.config = config
        self._snapshot_dir = Path(
            snapshot_download(
                repo_id=config.model_id,
                revision=config.revision,
                cache_dir=(
                    None if config.cache_dir is None else str(config.cache_dir)
                ),
                allow_patterns=["tokenizer/*", "processor/*"],
            )
        )
        tokenizer_dir = self._snapshot_dir / "tokenizer"
        processor_dir = self._snapshot_dir / "processor"
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
        )
        image_processor = Qwen2VLImageProcessor.from_pretrained(
            processor_dir,
            local_files_only=True,
        )
        video_processor = Qwen3VLVideoProcessor.from_pretrained(
            processor_dir,
            local_files_only=True,
        )
        chat_template = json.loads(
            (processor_dir / "chat_template.json").read_text(encoding="utf-8")
        )["chat_template"]
        self.processor = Qwen3VLProcessor(
            image_processor=image_processor,
            video_processor=video_processor,
            tokenizer=self.tokenizer,
            chat_template=chat_template,
        )

    def _component_dir(self, component: str) -> Path:
        """Download one pinned allowlisted component and return its local path."""
        from huggingface_hub import snapshot_download

        snapshot_dir = Path(
            snapshot_download(
                repo_id=self.config.model_id,
                revision=self.config.revision,
                cache_dir=(
                    None
                    if self.config.cache_dir is None
                    else str(self.config.cache_dir)
                ),
                allow_patterns=[f"{component}/*"],
            )
        )
        if snapshot_dir != self._snapshot_dir:
            raise RuntimeError("Pinned H3 components resolved to different snapshots")
        return snapshot_dir / component

    def load_text_encoder(self) -> nn.Module:
        """Materialize only the Qwen3-VL conditioner on the execution device."""
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(
            self._component_dir("text_encoder"),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": self.config.device},
            low_cpu_mem_usage=True,
        ).eval()

    def load_video_vae(self) -> nn.Module:
        """Materialize the native full-precision video VAE."""
        return MiniMaxH3VideoVAEConfig(
            device=self.config.device,
            checkpoint_min_free_gb=self.config.checkpoint_min_free_gb,
        ).setup()

    def load_audio_vae(self) -> nn.Module:
        """Materialize the native full-precision waveform VAE."""
        return MiniMaxH3AudioVAEConfig(
            device=self.config.device,
            checkpoint_min_free_gb=self.config.checkpoint_min_free_gb,
        ).setup()

    def load_diffusion_model(
        self, workflow: MiniMaxH3Workflow, num_inference_steps: int
    ) -> nn.Module:
        """Materialize the workflow's native transformer and paired schedules."""
        checkpoint = (
            H3_REF_TRANSFORMER_CHECKPOINT
            if workflow == "ref2va"
            else H3_TRANSFORMER_CHECKPOINT
        )
        transformer = MiniMaxH3TransformerConfig(
            checkpoint_path=checkpoint,
            checkpoint_min_free_gb=self.config.checkpoint_min_free_gb,
            device=self.config.device,
            execution_device=self.config.device,
            sequential_cpu_offload=False,
            attention_backend=self.config.attention_backend,
        )
        return MiniMaxH3DiffusionModelConfig(
            transformer=transformer,
            scheduler=MiniMaxH3SchedulerConfig(
                num_inference_steps=num_inference_steps, shift=12.0
            ),
            audio_scheduler=MiniMaxH3SchedulerConfig(
                num_inference_steps=num_inference_steps, shift=3.0
            ),
        ).setup()

    def release(self, module: Any) -> None:
        """Drop a completed stage without staging its weights in host RAM."""
        try:
            module.to_empty(device="cpu")
        except (AttributeError, RuntimeError):
            logger.debug("H3 stage did not support to_empty during release")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Release small shared tokenizer/processor metadata."""
        del self.tokenizer
        del self.processor


@dataclass(frozen=True, kw_only=True, slots=True)
class _ConditionedRequest:
    text: MiniMaxH3TextCondition
    keyframe_anchors: tuple[str, ...] = ()
    encoded_video: tuple[Tensor, ...] = ()
    normalized_references: tuple[MiniMaxH3Reference, ...] = ()
    encoded_references: MiniMaxH3EncodedReferences | None = None


class MiniMaxH3InferenceEngine:
    """Run each H3 stage once while keeping heavyweight weights disjoint."""

    def __init__(
        self,
        config: MiniMaxH3InferenceConfig | None = None,
        *,
        resources: MiniMaxH3Resources | None = None,
    ) -> None:
        self.config = config or MiniMaxH3InferenceConfig()
        if resources is None:
            validate_execution_capacity(self.config)
            resources = DefaultMiniMaxH3Resources(self.config)
        self.resources: MiniMaxH3Resources = resources
        self._closed = False

    @torch.no_grad()
    def generate(self, request: MiniMaxH3InferenceRequest) -> MiniMaxH3InferenceResult:
        """Condition, jointly denoise, and decode one finite H3 request."""
        if self._closed:
            raise RuntimeError("MiniMax H3 inference engine is closed")
        cuda_device = (
            torch.device(self.config.device)
            if torch.cuda.is_available() and self.config.device.startswith("cuda")
            else None
        )

        def synchronized_time() -> float:
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
            return time.monotonic()

        started = synchronized_time()
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        conditioned_started = started
        conditioned = self._condition(request)
        conditioned_finished = synchronized_time()
        conditioning_seconds = conditioned_finished - conditioned_started

        state_started = conditioned_finished
        state = self._prepare_state(request, conditioned)
        state_finished = synchronized_time()
        prepare_seconds = state_finished - state_started

        denoise_started = state_finished
        joint = self._denoise(request, state)
        denoise_finished = synchronized_time()
        denoise_seconds = denoise_finished - denoise_started

        video_started = denoise_finished
        video = self._decode_video(joint)
        video_finished = synchronized_time()
        video_decode_seconds = video_finished - video_started
        audio_started = video_finished
        audio = self._decode_audio(joint)
        finished = synchronized_time()
        audio_decode_seconds = finished - audio_started
        peak = 0.0
        if cuda_device is not None:
            peak = torch.cuda.max_memory_allocated(cuda_device) / 2**30
        total_seconds = finished - started
        metrics: dict[str, float | int] = {
            "conditioning_s": conditioning_seconds,
            "prepare_s": prepare_seconds,
            "denoise_s": denoise_seconds,
            "video_decode_s": video_decode_seconds,
            "audio_decode_s": audio_decode_seconds,
            "total_s": total_seconds,
            "generated_fps": request.num_frames / total_seconds,
            "peak_gpu_memory_gib": peak,
            "aligned_frame_count": request.num_frames,
            "audio_sample_count": audio.samples.shape[1],
        }
        return MiniMaxH3InferenceResult(video=video, audio=audio, metrics=metrics)

    def _condition(self, request: MiniMaxH3InferenceRequest) -> _ConditionedRequest:
        keyframes = None
        normalized_references: tuple[MiniMaxH3Reference, ...] = ()
        if request.workflow == "fl2va":
            keyframes = prepare_keyframes(
                first_image=request.first_image,
                last_image=request.last_image,
                height=request.height,
                width=request.width,
            )
            presentation = build_fl2va_presentation(
                self.resources.tokenizer,
                self.resources.processor,
                request.prompt,
                keyframes.images,
            )
        elif request.workflow == "ref2va":
            normalized_references = normalize_references(
                list(request.references), num_frames=request.num_frames
            )
            presentation = build_ref2va_presentation(
                self.resources.tokenizer,
                self.resources.processor,
                request.prompt,
                normalized_references,
            )
        else:
            presentation = build_t2va_presentation(
                self.resources.tokenizer, request.prompt
            )

        text_encoder = self.resources.load_text_encoder()
        try:
            text = encode_presentation(
                text_encoder,
                self.resources.processor,
                presentation,
                device=self.config.device,
                dtype=torch.bfloat16,
            )
            text = MiniMaxH3TextCondition(
                prompt_embeds=text.prompt_embeds.detach().cpu(),
                text_token_tags=text.text_token_tags.detach().cpu(),
            )
        finally:
            self.resources.release(text_encoder)
            del text_encoder

        encoded_video: tuple[Tensor, ...] = ()
        encoded_references = None
        if keyframes is not None:
            video_vae = self.resources.load_video_vae()
            try:
                encoded_video = tuple(
                    value.detach().cpu()
                    for value in encode_keyframes(video_vae, keyframes)
                )
            finally:
                self.resources.release(video_vae)
                del video_vae
        elif normalized_references:
            video_vae = self.resources.load_video_vae()
            audio_vae = None
            try:
                if any(reference.has_audio for reference in normalized_references):
                    audio_vae = self.resources.load_audio_vae()
                encoded = encode_references(
                    video_vae,
                    audio_vae,
                    normalized_references,
                    device=self.config.device,
                )
                encoded_references = MiniMaxH3EncodedReferences(
                    video=tuple(value.detach().cpu() for value in encoded.video),
                    audio=tuple(value.detach().cpu() for value in encoded.audio),
                )
            finally:
                if audio_vae is not None:
                    self.resources.release(audio_vae)
                self.resources.release(video_vae)
                del audio_vae, video_vae
        return _ConditionedRequest(
            text=text,
            keyframe_anchors=() if keyframes is None else keyframes.anchors,
            encoded_video=encoded_video,
            normalized_references=normalized_references,
            encoded_references=encoded_references,
        )

    def _prepare_state(
        self,
        request: MiniMaxH3InferenceRequest,
        conditioned: _ConditionedRequest,
    ) -> MiniMaxH3DenoiseState:
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        if request.workflow == "ref2va":
            if conditioned.encoded_references is None:
                raise RuntimeError("REF2VA reference encoding is missing")
            return prepare_ref2va_denoise_state(
                conditioned.text.prompt_embeds,
                conditioned.text.text_token_tags,
                conditioned.normalized_references,
                conditioned.encoded_references,
                num_frames=request.num_frames,
                height=request.height,
                width=request.width,
                generator=generator,
                device="cpu",
            )
        return prepare_denoise_state(
            conditioned.text.prompt_embeds,
            conditioned.text.text_token_tags,
            condition_latents=conditioned.encoded_video,
            keyframe_anchors=conditioned.keyframe_anchors,
            num_frames=request.num_frames,
            height=request.height,
            width=request.width,
            generator=generator,
            device="cpu",
        )

    def _checkpoint_identity(
        self, request: MiniMaxH3InferenceRequest
    ) -> MiniMaxH3CheckpointIdentity:
        lora = (
            None
            if request.lora_path is None
            else MiniMaxH3AssetIdentity.from_file(request.lora_path)
        )
        return MiniMaxH3CheckpointIdentity(
            workflow=request.workflow,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            aligned_num_frames=request.num_frames,
            num_audio_latents=audio_latent_num_frames(request.num_frames),
            seed=request.seed,
            num_inference_steps=request.num_inference_steps,
            video_scheduler_shift=12.0,
            audio_scheduler_shift=3.0,
            model=MiniMaxH3AssetIdentity(
                source=self.config.model_id,
                resolved_revision=self.config.revision,
            ),
            inputs=request.checkpoint_inputs,
            lora=lora,
            lora_scale=request.lora_scale,
        )

    def _denoise(
        self,
        request: MiniMaxH3InferenceRequest,
        state: MiniMaxH3DenoiseState,
    ) -> MiniMaxH3JointLatents:
        model = self.resources.load_diffusion_model(
            request.workflow, request.num_inference_steps
        )
        try:
            if request.lora_path is not None:
                load_lora(
                    model.transformer,
                    str(request.lora_path),
                    request.lora_scale,
                    request.lora_weight_name,
                )
            resume = None
            checkpoint: Callable[[MiniMaxH3DenoiseProgress], None] | None = None
            if request.checkpoint_store is not None:
                store = request.checkpoint_store
                identity = self._checkpoint_identity(request)
                if store.path.is_file() and not request.restart:
                    resume = store.load(identity)

                def save_checkpoint(progress: MiniMaxH3DenoiseProgress) -> None:
                    store.save(identity, progress)

                checkpoint = save_checkpoint
            return model.generate_joint(
                state, resume=resume, checkpoint=checkpoint
            )
        finally:
            self.resources.release(model)
            del model

    def _decode_video(self, joint: MiniMaxH3JointLatents) -> Tensor:
        video_vae = self.resources.load_video_vae()
        try:
            pixels = video_vae.decode_output(
                joint.video.to(self.config.device)
            )
            return (
                pixels[0]
                .permute(1, 0, 2, 3)
                .mul(2.0)
                .sub(1.0)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
            )
        finally:
            self.resources.release(video_vae)
            del video_vae

    def _decode_audio(self, joint: MiniMaxH3JointLatents) -> AudioOutput:
        audio_vae = self.resources.load_audio_vae()
        try:
            decoded = audio_vae.decode_output(
                joint.audio.to(self.config.device)
            )
            return AudioOutput(
                samples=decoded.samples.detach().to("cpu").contiguous(),
                sample_rate=decoded.sample_rate,
                sample_offset=decoded.sample_offset,
            )
        finally:
            self.resources.release(audio_vae)
            del audio_vae

    def close(self) -> None:
        """Release shared metadata and reject further generation."""
        if self._closed:
            return
        self._closed = True
        self.resources.close()


def validate_execution_capacity(config: MiniMaxH3InferenceConfig) -> None:
    """Reject unsupported devices and insufficient staged disk/GPU capacity."""
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native MiniMax H3 inference requires a CUDA device")
    free_gpu, _ = torch.cuda.mem_get_info(device)
    required_gpu = int(_MIN_FREE_GPU_GIB * 2**30)
    if free_gpu < required_gpu:
        raise RuntimeError(
            f"MiniMax H3 staged inference requires at least {_MIN_FREE_GPU_GIB:g} "
            f"GiB free GPU memory, found {free_gpu / 2**30:.1f} GiB"
        )
    if config.cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
    else:
        cache_dir = config.cache_dir.expanduser()
    existing = cache_dir
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    free_disk = shutil.disk_usage(existing).free
    required_disk = int(_MIN_FREE_CACHE_GIB * 2**30)
    if free_disk < required_disk:
        raise RuntimeError(
            f"Pulling the pinned H3 components requires at least "
            f"{_MIN_FREE_CACHE_GIB:g} GiB free in the model cache filesystem, "
            f"found {free_disk / 2**30:.1f} GiB"
        )


__all__ = [
    "MODEL_REVISION",
    "DefaultMiniMaxH3Resources",
    "MiniMaxH3InferenceConfig",
    "MiniMaxH3InferenceEngine",
    "MiniMaxH3InferenceRequest",
    "MiniMaxH3InferenceResult",
    "MiniMaxH3Resources",
    "MiniMaxH3Workflow",
    "validate_execution_capacity",
]
