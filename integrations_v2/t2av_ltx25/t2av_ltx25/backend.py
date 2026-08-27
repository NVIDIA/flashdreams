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

"""Model-neutral backend seam and Diffusers implementation for LTX 2.5."""

import gc
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
"""Official Diffusers-packaged LTX 2.5 checkpoint."""

MODEL_REVISION = "426936f8b22dc28e4def61e515478b0b7e4a53cc"
"""Immutable model snapshot validated by this integration."""

MODEL_ACCESS_URL = "https://huggingface.co/Lightricks/LTX-2.5-Diffusers"
"""Checkpoint gate whose terms users must accept before downloading weights."""

DEFAULT_AUDIO_SAMPLE_RATE = 48_000
"""LTX 2.5 bandwidth-extended vocoder output sample rate."""

DEFAULT_AUDIO_CHANNELS = 2
"""Stereo channels emitted by the LTX 2.5 vocoder."""

OffloadMode = Literal["model", "sequential", "none"]


@dataclass(frozen=True, kw_only=True, slots=True)
class BackendLoadConfig:
    """Configuration that affects model construction and residency."""

    device: str
    """Torch device used for inference."""

    offload: OffloadMode
    """Diffusers CPU-offload policy."""

    local_files_only: bool
    """Whether checkpoint loading must avoid network access."""


@dataclass(frozen=True, kw_only=True, slots=True)
class GenerationRequest:
    """One joint audio-video generation request."""

    prompt: str
    """Text prompt conditioning both output modalities."""

    seed: int
    """Seed for the CUDA random generator."""

    num_frames: int
    """Requested video frame count on the LTX temporal grid."""

    width: int
    """Output width in pixels."""

    height: int
    """Output height in pixels."""

    frame_rate: int
    """Playback frame rate in frames per second."""


@dataclass(frozen=True, kw_only=True, slots=True)
class GeneratedMedia:
    """Decoded media returned by an LTX backend."""

    video: Tensor
    """Video with shape ``[T, C, H, W]`` and dtype ``uint8``."""

    audio: Tensor
    """Normalized PCM with shape ``[channels, samples]``."""

    metrics: dict[str, float | int] = field(default_factory=dict)
    """Backend measurements using runtime-recognized unit suffixes."""


class LTX25Backend(Protocol):
    """Shared model surface consumed by the V2 application."""

    @property
    def sample_rate(self) -> int:
        """Return decoded PCM samples per second."""
        ...

    @property
    def audio_channels(self) -> int:
        """Return the decoded PCM channel count."""
        ...

    def generate(self, request: GenerationRequest) -> GeneratedMedia:
        """Generate synchronized media for ``request``."""
        ...

    def close(self) -> None:
        """Release model resources."""
        ...


class DiffusersLTX25Backend:
    """Pinned Diffusers LTX 2.5 pipeline with normalized tensor output."""

    def __init__(self, pipeline: Any, device: str) -> None:
        self._pipeline = pipeline
        self._device = device
        self._sample_rate = int(pipeline.vocoder.config.output_sampling_rate)
        if self._sample_rate != DEFAULT_AUDIO_SAMPLE_RATE:
            raise RuntimeError(
                "The pinned LTX 2.5 checkpoint must emit "
                f"{DEFAULT_AUDIO_SAMPLE_RATE} Hz audio, got {self._sample_rate}."
            )

    @property
    def sample_rate(self) -> int:
        """Return decoded PCM samples per second."""
        return self._sample_rate

    @property
    def audio_channels(self) -> int:
        """Return the decoded PCM channel count."""
        return DEFAULT_AUDIO_CHANNELS

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> GeneratedMedia:
        """Generate one distilled LTX 2.5 clip and move it to bounded CPU tensors."""
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

        cuda_device = _cuda_device(self._device)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)

        started_at = time.perf_counter()
        generator = torch.Generator(device=self._device).manual_seed(request.seed)
        pipeline = self._pipeline
        if pipeline is None:
            raise RuntimeError("The LTX 2.5 backend is closed.")
        result = pipeline(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_frames=request.num_frames,
            frame_rate=float(request.frame_rate),
            sigmas=DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            audio_guidance_scale=1.0,
            generator=generator,
            output_type="pt",
        )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        pipeline_s = time.perf_counter() - started_at

        video = _video_to_uint8(result.frames)
        audio = _audio_to_normalized_pcm(result.audio)
        if audio.shape[0] != self.audio_channels:
            raise RuntimeError(
                f"LTX 2.5 returned {audio.shape[0]} audio channels; "
                f"expected {self.audio_channels}."
            )

        metrics: dict[str, float | int] = {
            "pipeline_s": pipeline_s,
            "audio_samples_count": int(audio.shape[1]),
        }
        if cuda_device is not None:
            metrics["peak_cuda_memory_gib"] = float(
                torch.cuda.max_memory_allocated(cuda_device) / 1024**3
            )
        return GeneratedMedia(video=video, audio=audio, metrics=metrics)

    def close(self) -> None:
        """Release the pipeline and return cached CUDA allocations to the allocator."""
        pipeline = self._pipeline
        self._pipeline = None
        maybe_free = getattr(pipeline, "maybe_free_model_hooks", None)
        if maybe_free is not None:
            maybe_free()
        del pipeline
        gc.collect()
        cuda_device = _cuda_device(self._device)
        if cuda_device is not None:
            torch.cuda.empty_cache()


def load_diffusers_backend(config: BackendLoadConfig) -> LTX25Backend:
    """Load the pinned LTX 2.5 Diffusers pipeline.

    Raises:
        RuntimeError: The model gate has not been accepted or the checkpoint's
            audio contract differs from the pinned integration.
    """
    from diffusers import LTX2Pipeline
    from huggingface_hub.errors import GatedRepoError

    try:
        pipeline = LTX2Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            local_files_only=config.local_files_only,
            prompt_enhancer=None,
            processor=None,
        )
    except GatedRepoError as error:
        raise RuntimeError(
            "LTX 2.5 checkpoint access is gated. Request access at "
            f"{MODEL_ACCESS_URL}, then retry."
        ) from error

    pipeline.vae.enable_tiling()
    if config.offload == "model":
        pipeline.enable_model_cpu_offload(device=config.device)
    elif config.offload == "sequential":
        pipeline.enable_sequential_cpu_offload(device=config.device)
    else:
        pipeline.to(config.device)
    return DiffusersLTX25Backend(pipeline, config.device)


def _cuda_device(device: str) -> torch.device | None:
    """Return a usable CUDA device, or none for a non-CUDA execution target."""
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.empty(0, device=resolved)
    return resolved


def _video_to_uint8(video: Any) -> Tensor:
    """Validate and convert Diffusers output to contiguous CPU TCHW RGB."""
    frames = torch.as_tensor(video).detach().to(device="cpu", dtype=torch.float32)
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[2] != 3:
        raise RuntimeError(
            f"LTX 2.5 video must have shape [1, T, 3, H, W], got {tuple(frames.shape)}."
        )
    if not bool(torch.isfinite(frames).all()):
        raise RuntimeError("LTX 2.5 video contains non-finite values.")
    if bool((frames < -1e-4).any()) or bool((frames > 1.0001).any()):
        raise RuntimeError("LTX 2.5 postprocessed video must stay within [0, 1].")
    return frames[0].clamp(0, 1).mul(255).round().to(torch.uint8).contiguous()


def _audio_to_normalized_pcm(audio: Any) -> Tensor:
    """Validate and convert Diffusers output to contiguous channel-major PCM."""
    samples = torch.as_tensor(audio).detach().to(device="cpu", dtype=torch.float32)
    if samples.ndim != 3 or samples.shape[0] != 1:
        raise RuntimeError(
            "LTX 2.5 audio must have shape [1, channels, samples], got "
            f"{tuple(samples.shape)}."
        )
    samples = samples[0]
    if not bool(torch.isfinite(samples).all()):
        raise RuntimeError("LTX 2.5 audio contains non-finite values.")
    if bool((samples < -1.0001).any()) or bool((samples > 1.0001).any()):
        raise RuntimeError("LTX 2.5 audio must stay within [-1, 1].")
    return samples.clamp(-1, 1).contiguous()


__all__ = [
    "DEFAULT_AUDIO_CHANNELS",
    "DEFAULT_AUDIO_SAMPLE_RATE",
    "MODEL_ID",
    "MODEL_REVISION",
    "BackendLoadConfig",
    "GeneratedMedia",
    "GenerationRequest",
    "LTX25Backend",
    "OffloadMode",
    "load_diffusers_backend",
]
