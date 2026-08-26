# SPDX-FileCopyrightText: Copyright (c) 2026 Hongyu Zhou
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

"""Session-owned LingBot-VA Robotwin inference engine."""

from __future__ import annotations

import gc
import html
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from flashdreams.infra.config import derive_config
from lingbot_va._loaders import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
    resolve_checkpoint_root,
)
from lingbot_va.action import LingbotVAActionProcessor
from lingbot_va.config import PIPELINE_LINGBOT_VA_ROBOTWIN_I2AV
from lingbot_va.constants import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_INPUT_IMAGE_DIR,
    DEFAULT_PROMPT,
    ROBOTWIN_ACTION_GUIDANCE_SCALE,
    ROBOTWIN_ACTION_INFERENCE_STEPS,
    ROBOTWIN_ACTION_SNR_SHIFT,
    ROBOTWIN_COMPOSITE_HEIGHT,
    ROBOTWIN_FRAME_CHUNK_SIZE,
    ROBOTWIN_GUIDANCE_SCALE,
    ROBOTWIN_HEIGHT,
    ROBOTWIN_OBS_CAM_KEYS,
    ROBOTWIN_SNR_SHIFT,
    ROBOTWIN_VAE_TEMPORAL_SCALE,
    ROBOTWIN_VIDEO_INFERENCE_STEPS,
    ROBOTWIN_WIDTH,
)
from lingbot_va.pipeline import LingbotVAInferencePipelineConfig
from lingbot_va.utils import resolve_prompt


@dataclass(frozen=True, slots=True, kw_only=True)
class LingbotVAEngineConfig:
    """Resolved model and input settings for one destructive engine run."""

    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT
    checkpoint_revision: str | None = None
    input_image_dir: Path = DEFAULT_INPUT_IMAGE_DIR
    prompt: str | Path = DEFAULT_PROMPT
    num_chunks: int = 10
    seed: int = 42
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    enable_offload: bool = False
    compile_network: bool = True
    guidance_scale: float = ROBOTWIN_GUIDANCE_SCALE
    action_guidance_scale: float = ROBOTWIN_ACTION_GUIDANCE_SCALE
    video_inference_steps: int = ROBOTWIN_VIDEO_INFERENCE_STEPS
    action_inference_steps: int = ROBOTWIN_ACTION_INFERENCE_STEPS
    video_snr_shift: float = ROBOTWIN_SNR_SHIFT
    action_snr_shift: float = ROBOTWIN_ACTION_SNR_SHIFT

    def __post_init__(self) -> None:
        """Reject invalid user-controlled settings before loading weights."""
        if not str(self.checkpoint_root):
            raise ValueError("checkpoint_root must not be empty")
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        if self.video_inference_steps <= 0 or self.action_inference_steps <= 0:
            raise ValueError("inference step counts must be positive")
        if self.video_snr_shift <= 0 or self.action_snr_shift <= 0:
            raise ValueError("SNR shifts must be positive")
        if self.guidance_scale < 0 or self.action_guidance_scale < 0:
            raise ValueError("guidance scales must be non-negative")


@dataclass(frozen=True, slots=True)
class LingbotVAEngineOutput:
    """CPU outputs from one complete Robotwin rollout."""

    video: Tensor
    """Decoded video in ``[T, C, H, W]`` layout and the ``[-1, 1]`` range."""

    actions: Tensor
    """Denormalized actions in ``[step, channel]`` layout."""

    metrics: Mapping[str, float]
    """Measured phase durations and peak allocated CUDA bytes."""


class LingbotVAEngineState(Enum):
    """Legal lifecycle states for a one-run engine."""

    NEW = auto()
    RUNNING = auto()
    FINISHED = auto()
    CLOSED = auto()


def required_image_paths(input_image_dir: Path) -> dict[str, Path]:
    """Return the canonical Robotwin camera-key-to-PNG mapping."""
    return {key: input_image_dir / f"{key}.png" for key in ROBOTWIN_OBS_CAM_KEYS}


def validate_input_images(input_image_dir: Path) -> dict[str, Path]:
    """Validate all three Robotwin camera PNGs without loading model state."""
    image_paths = required_image_paths(input_image_dir)
    missing = [path for path in image_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Robotwin camera PNGs: " + ", ".join(str(path) for path in missing)
        )
    return image_paths


def validate_device(device_name: str) -> torch.device:
    """Resolve a requested device and fail early when CUDA is unavailable."""
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {device}"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
    return device


def build_pipeline_config(
    config: LingbotVAEngineConfig,
    checkpoint_root: Path,
) -> LingbotVAInferencePipelineConfig:
    """Apply every effective engine override to a copied pipeline config."""
    return derive_config(
        PIPELINE_LINGBOT_VA_ROBOTWIN_I2AV,
        checkpoint_root=str(checkpoint_root),
        enable_sync_and_profile=False,
        diffusion_model={
            "seed": config.seed,
            "transformer": {
                "checkpoint_root": str(checkpoint_root),
                "dtype": config.dtype,
                "compile_network": config.compile_network,
                "guidance_scale": config.guidance_scale,
                "action_guidance_scale": config.action_guidance_scale,
            },
            "scheduler": {
                "num_inference_steps": config.video_inference_steps,
                "shift": config.video_snr_shift,
            },
        },
        action_scheduler={
            "num_inference_steps": config.action_inference_steps,
            "shift": config.action_snr_shift,
        },
    )


def _prompt_clean(text: str) -> str:
    """Apply the upstream double HTML decode and whitespace cleanup."""
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    return re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip()


class LingbotVAEngine:
    """Own model components for exactly one complete LingBot-VA rollout.

    Video decoding requires releasing the denoising pipeline first. Therefore
    this object cannot be reused after :meth:`run`; a session reset creates a
    new engine.
    """

    def __init__(self, config: LingbotVAEngineConfig) -> None:
        """
        Args:
            config: Immutable settings for the rollout.
        """
        self.config = config
        self._state = LingbotVAEngineState.NEW
        self._device: torch.device | None = None
        self._pipeline: Any | None = None
        self._pipeline_cache: Any | None = None
        self._vae: Any | None = None
        self._text_encoder: Any | None = None
        self._tokenizer: Any | None = None
        self._streaming_vae: WanVAEStreamingWrapper | None = None
        self._streaming_vae_half: WanVAEStreamingWrapper | None = None

    @property
    def state(self) -> LingbotVAEngineState:
        """Return the current one-run lifecycle state."""
        return self._state

    def run(self) -> LingbotVAEngineOutput:
        """Run one rollout and return decoded video, actions, and metrics."""
        if self._state is not LingbotVAEngineState.NEW:
            raise RuntimeError(
                f"LingbotVAEngine.run() requires NEW state, got {self._state.name}."
            )
        self._state = LingbotVAEngineState.RUNNING
        try:
            output = self._run_impl()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise
        self._state = LingbotVAEngineState.FINISHED
        return output

    def _run_impl(self) -> LingbotVAEngineOutput:
        """Execute the real model path after lifecycle validation."""
        started = time.perf_counter()
        image_paths = validate_input_images(self.config.input_image_dir)
        prompt = resolve_prompt(self.config.prompt)
        device = validate_device(self.config.device)
        checkpoint_root = resolve_checkpoint_root(
            self.config.checkpoint_root,
            revision=self.config.checkpoint_revision,
        )
        self._device = device

        if device.type == "cuda":
            # PyTorch 2.12 can reject reset_peak_memory_stats before the CUDA
            # context exists. Resolve the current device inside its context
            # first so every session reports its own peak rather than a
            # process-lifetime high-water mark.
            with torch.cuda.device(device):
                torch.cuda.current_device()
                torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(self.config.seed)
        self._load_models(checkpoint_root, device)

        prompt_started = time.perf_counter()
        prompt_embeds, negative_prompt_embeds = self._prepare_prompt(prompt, device)
        prompt_seconds = time.perf_counter() - prompt_started

        observation_started = time.perf_counter()
        init_latent = self._prepare_observation(image_paths, device)
        observation_seconds = time.perf_counter() - observation_started

        denoise_started = time.perf_counter()
        latents, actions = self._generate_chunks(
            prompt_embeds,
            negative_prompt_embeds,
            init_latent,
            device,
        )
        denoise_seconds = time.perf_counter() - denoise_started

        del prompt_embeds, negative_prompt_embeds, init_latent
        self._release_denoising_state()

        decode_started = time.perf_counter()
        video = self._decode_video(latents, device)
        decode_seconds = time.perf_counter() - decode_started
        self._release_vae()

        peak_allocated = 0.0
        if device.type == "cuda":
            peak_allocated = float(torch.cuda.max_memory_allocated(device))
        return LingbotVAEngineOutput(
            video=video,
            actions=actions,
            metrics={
                "prompt_encode_s": prompt_seconds,
                "observation_encode_s": observation_seconds,
                "denoise_s": denoise_seconds,
                "decode_s": decode_seconds,
                "total_s": time.perf_counter() - started,
                "peak_allocated_bytes": peak_allocated,
            },
        )

    def _load_models(self, checkpoint_root: Path, device: torch.device) -> None:
        """Load shared VAE, text components, and the native transformer."""
        component_device = torch.device("cpu") if self.config.enable_offload else device
        self._vae = load_vae(checkpoint_root, self.config.dtype, component_device)
        self._streaming_vae = WanVAEStreamingWrapper(self._vae)
        self._streaming_vae_half = WanVAEStreamingWrapper(self._vae)
        self._tokenizer = load_tokenizer(checkpoint_root)
        self._text_encoder = load_text_encoder(
            checkpoint_root,
            self.config.dtype,
            component_device,
        )
        pipeline_config = build_pipeline_config(self.config, checkpoint_root)
        self._pipeline = pipeline_config.setup()
        self._pipeline.transformer.load_model(device)

    def _prepare_prompt(
        self,
        prompt: str,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Encode positive and, when required, negative text embeddings."""
        if self._text_encoder is None:
            raise RuntimeError("text encoder is not loaded")
        if self.config.enable_offload:
            self._text_encoder.to(device)
        positive = self._encode_prompt(prompt, device)
        use_cfg = (
            self.config.guidance_scale > 1.0 or self.config.action_guidance_scale > 1.0
        )
        negative = self._encode_prompt("", device) if use_cfg else positive
        if self.config.enable_offload:
            self._text_encoder.to("cpu")
            self._empty_cuda_cache()
        return positive, negative

    def _encode_prompt(self, prompt: str, device: torch.device) -> Tensor:
        """Encode one cleaned prompt with the upstream 512-token contract."""
        if self._tokenizer is None or self._text_encoder is None:
            raise RuntimeError("text components are not loaded")
        text_inputs = self._tokenizer(
            [_prompt_clean(prompt)],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = text_inputs.input_ids
        mask = text_inputs.attention_mask
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        encoder_device = next(self._text_encoder.parameters()).device
        embeddings = self._text_encoder(
            ids.to(encoder_device),
            mask.to(encoder_device),
        ).last_hidden_state
        embeddings = embeddings.to(dtype=self.config.dtype, device=device)
        trimmed = [item[:length] for item, length in zip(embeddings, sequence_lengths)]
        padded = [
            torch.cat(
                [item, item.new_zeros(512 - item.size(0), item.size(1))],
                dim=0,
            )
            for item in trimmed
        ]
        return torch.stack(padded, dim=0)

    def _prepare_observation(
        self,
        image_paths: Mapping[str, Path],
        device: torch.device,
    ) -> Tensor:
        """Load and encode the high and two wrist camera observations."""
        if self._vae is None:
            raise RuntimeError("VAE is not loaded")
        if self.config.enable_offload:
            self._vae.to(device)
        images = {
            key: np.asarray(Image.open(path).convert("RGB"))
            for key, path in image_paths.items()
        }
        output = self._encode_observation(images, device)
        if self.config.enable_offload:
            self._vae.to("cpu")
            self._empty_cuda_cache()
        return output

    def _encode_observation(
        self,
        observation_images: Mapping[str, np.ndarray],
        device: torch.device,
    ) -> Tensor:
        """Encode the three-camera Robotwin spatial arrangement."""
        if (
            self._vae is None
            or self._streaming_vae is None
            or self._streaming_vae_half is None
        ):
            raise RuntimeError("VAE streaming wrappers are not loaded")
        videos = []
        for camera_index, key in enumerate(ROBOTWIN_OBS_CAM_KEYS):
            if camera_index == 0:
                height, width = ROBOTWIN_HEIGHT, ROBOTWIN_WIDTH
            else:
                height, width = ROBOTWIN_HEIGHT // 2, ROBOTWIN_WIDTH // 2
            image = (
                torch.from_numpy(observation_images[key].copy())
                .float()
                .permute(2, 0, 1)
                .unsqueeze(1)
            )
            image = F.interpolate(
                image,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            videos.append(image.unsqueeze(0))

        high_video = videos[0] / 255.0 * 2.0 - 1.0
        wrist_video = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
        vae_device = next(self._vae.parameters()).device
        encoded_high = self._streaming_vae.encode_chunk(
            high_video.to(device=vae_device, dtype=self.config.dtype)
        )
        encoded_wrist = self._streaming_vae_half.encode_chunk(
            wrist_video.to(device=vae_device, dtype=self.config.dtype)
        )
        encoded = torch.cat(
            [
                torch.cat(encoded_wrist.split(1, dim=0), dim=-1),
                encoded_high,
            ],
            dim=-2,
        )
        mean, _ = torch.chunk(encoded, 2, dim=1)
        latent_mean = torch.as_tensor(
            self._vae.config.latents_mean,
            device=mean.device,
        ).view(1, -1, 1, 1, 1)
        latent_inverse_std = (
            1.0
            / torch.as_tensor(
                self._vae.config.latents_std,
                device=mean.device,
            )
        ).view(1, -1, 1, 1, 1)
        normalized = ((mean.float() - latent_mean) * latent_inverse_std).to(mean)
        return normalized.to(device)

    def _generate_chunks(
        self,
        prompt_embeddings: Tensor,
        negative_prompt_embeddings: Tensor,
        init_latent: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Generate all video/action chunks before destructive decode teardown."""
        if self._pipeline is None:
            raise RuntimeError("pipeline is not loaded")
        use_cfg = (
            self.config.guidance_scale > 1.0 or self.config.action_guidance_scale > 1.0
        )
        self._pipeline_cache = self._pipeline.initialize_cache(
            text_embeddings=prompt_embeddings,
            negative_text_embeddings=(negative_prompt_embeddings if use_cfg else None),
            batch_size=1,
        )
        action_processor = LingbotVAActionProcessor()
        action_mask = action_processor.action_mask(device=device)
        predicted_latents: list[Tensor] = []
        predicted_actions: list[Tensor] = []
        for chunk_index in range(self.config.num_chunks):
            output = self._pipeline.generate(
                autoregressive_index=chunk_index,
                cache=self._pipeline_cache,
                input={
                    "init_latent": init_latent,
                    "action_mask": action_mask,
                    "device": device,
                    "dtype": self.config.dtype,
                },
            )
            predicted_latents.append(output.latent.detach().cpu())
            predicted_actions.append(action_processor.postprocess(output.action))
            del output
            self._empty_cuda_cache()
        return (
            torch.cat(predicted_latents, dim=2),
            torch.cat(predicted_actions, dim=0),
        )

    def _release_denoising_state(self) -> None:
        """Release cache, DiT, text, and streaming encoder state before decode."""
        cache = self._pipeline_cache
        if cache is not None:
            transformer_cache = getattr(cache, "transformer_cache", None)
            if transformer_cache is not None:
                for network_cache in (
                    getattr(transformer_cache, "network_cache", None),
                    getattr(transformer_cache, "network_cache_uncond", None),
                ):
                    if network_cache is None:
                        continue
                    for block_cache in network_cache.block_caches:
                        block_cache.self_attn.reset()
                        block_cache.cross_attn.text.k = torch.empty(0)
                        block_cache.cross_attn.text.v = torch.empty(0)
        self._pipeline_cache = None

        for wrapper in (self._streaming_vae, self._streaming_vae_half):
            if wrapper is not None:
                wrapper.clear_cache()
        self._streaming_vae = None
        self._streaming_vae_half = None

        if self._pipeline is not None:
            transformer = self._pipeline.transformer
            network = getattr(transformer, "_network", None)
            if network is not None:
                network.to("cpu")
                object.__setattr__(transformer, "_network", None)
        self._pipeline = None
        if self._text_encoder is not None:
            self._text_encoder.to("cpu")
        self._text_encoder = None
        self._tokenizer = None
        gc.collect()
        self._empty_cuda_cache()

    def _decode_video(self, latents: Tensor, device: torch.device) -> Tensor:
        """Decode accumulated latent frames while offloading each frame to CPU."""
        if self._vae is None:
            raise RuntimeError("VAE is not loaded")
        vae = self._vae.to(device=device, dtype=self.config.dtype)
        latent = latents.to(device=device, dtype=self.config.dtype)
        latent_mean = torch.as_tensor(
            vae.config.latents_mean,
            device=device,
            dtype=self.config.dtype,
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latent_inverse_std = (
            1.0
            / torch.as_tensor(
                vae.config.latents_std,
                device=device,
                dtype=self.config.dtype,
            )
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latent = latent / latent_inverse_std + latent_mean

        vae.clear_cache()
        decoded_input = vae.post_quant_conv(latent)
        del latent
        decoded_frames: list[Tensor] = []
        for frame_index in range(decoded_input.shape[2]):
            vae._conv_idx = [0]
            decoder_kwargs: dict[str, Any] = {
                "feat_cache": vae._feat_map,
                "feat_idx": vae._conv_idx,
            }
            if frame_index == 0:
                decoder_kwargs["first_chunk"] = True
            frame = vae.decoder(
                decoded_input[:, :, frame_index : frame_index + 1],
                **decoder_kwargs,
            )
            decoded_frames.append(frame.detach().cpu())
            del frame
        del decoded_input
        vae.clear_cache()
        self._empty_cuda_cache()

        decoded = torch.cat(decoded_frames, dim=2)
        patch_size = getattr(vae.config, "patch_size", None)
        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            decoded = unpatchify(decoded, patch_size=patch_size)
        if tuple(decoded.shape[-2:]) != (ROBOTWIN_COMPOSITE_HEIGHT, ROBOTWIN_WIDTH):
            raise ValueError(
                "LingBot-VA VAE returned composite spatial shape "
                f"{tuple(decoded.shape[-2:])}; expected "
                f"{(ROBOTWIN_COMPOSITE_HEIGHT, ROBOTWIN_WIDTH)}."
            )
        high_camera = decoded[..., -ROBOTWIN_HEIGHT:, :]
        return high_camera[0].clamp(-1.0, 1.0).permute(1, 0, 2, 3).contiguous()

    def _release_vae(self) -> None:
        """Release the final model component after decoded tensors reach CPU."""
        if self._vae is not None:
            self._vae.to("cpu")
            if hasattr(self._vae, "clear_cache"):
                self._vae.clear_cache()
        self._vae = None
        gc.collect()
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        """Drop unused CUDA allocations when this engine targets CUDA."""
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Idempotently release partially or fully initialized model state."""
        if self._state is LingbotVAEngineState.CLOSED:
            return
        try:
            self._release_denoising_state()
        finally:
            try:
                self._release_vae()
            finally:
                self._state = LingbotVAEngineState.CLOSED


def expected_output_shape(config: LingbotVAEngineConfig) -> tuple[int, int, int, int]:
    """Return the fixed natural decoded video shape for one rollout."""
    latent_frames = config.num_chunks * ROBOTWIN_FRAME_CHUNK_SIZE
    decoded_frames = (latent_frames - 1) * ROBOTWIN_VAE_TEMPORAL_SCALE + 1
    return (
        decoded_frames,
        3,
        ROBOTWIN_HEIGHT,
        ROBOTWIN_WIDTH,
    )


__all__ = [
    "LingbotVAEngine",
    "LingbotVAEngineConfig",
    "LingbotVAEngineOutput",
    "LingbotVAEngineState",
    "build_pipeline_config",
    "expected_output_shape",
    "required_image_paths",
    "validate_device",
    "validate_input_images",
]
