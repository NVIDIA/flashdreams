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

"""SANA-WM prompt, first-frame, and camera conditioning components."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from cam2v import Cam2VConditioning
from PIL import Image
from torch import Tensor

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.infra.encoder import (
    Encoder,
    EncoderConfig,
    StreamingEncoder,
    StreamingEncoderCache,
)
from sana_wm.impl.camera import (
    default_intrinsics_vec4,
    load_intrinsics,
    prepare_camera,
    resize_center_crop_geometry,
    transform_intrinsics_for_crop,
)
from sana_wm.impl.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_CONFIG_PATH,
    SANA_WM_STREAMING_CONFIG_PATH,
    SANA_WM_STREAMING_LATENT_CHUNK_SIZE,
)
from sana_wm.impl.transformer import (
    QuantBackend,
    SanaWMStage1Conditioning,
    SanaWMStreamingStage1Conditioning,
    _chunk_index_from_config,
    _get_tokenizer_and_text_encoder,
    _get_vae,
    _get_weight_dtype,
    _load_inference_config,
    _vae_encode_ltx2,
)

_EXAMPLE_DATA_BASE_URL = (
    "https://raw.githubusercontent.com/NVlabs/Sana/main/asset/sana_wm"
)
_EXAMPLE_DATA_DIR = default_flashdreams_cache_dir() / "example_data/sana_wm"
_EXAMPLE_DATA_INDICES = frozenset(range(5))


@dataclass(kw_only=True)
class SanaWMTextPromptRequest:
    """Raw text prompt inputs for SANA-WM Stage 1."""

    prompt: str
    negative_prompt: str = ""


@dataclass(kw_only=True)
class SanaWMTextConditioning:
    """Encoded positive and negative prompt tensors."""

    condition: Tensor
    condition_mask: Tensor
    negative: Tensor
    negative_mask: Tensor


@dataclass(kw_only=True)
class SanaWMCameraRequest:
    """Raw camera trajectory inputs for SANA-WM conditioning."""

    poses_c2w: np.ndarray
    intrinsics_vec4: np.ndarray


@dataclass(kw_only=True)
class SanaWMI2VConditioning:
    """Encoded first-frame and camera tensors for Stage-1 diffusion."""

    first_latent: Tensor
    camera: dict[str, Tensor]
    num_frames: int


@dataclass(kw_only=True)
class SanaWMI2VConditioningRequest:
    """Raw one-shot SANA-WM I2V rollout inputs."""

    image: Any
    prompt: str
    poses_c2w: np.ndarray
    intrinsics_vec4: np.ndarray
    num_frames: int
    fps: int
    steps: int
    cfg_scale: float
    flow_shift: float | None
    seed: int
    negative_prompt: str = ""


@dataclass(kw_only=True)
class SanaWMStreamingI2VConditioningRequest(SanaWMI2VConditioningRequest):
    """Raw SANA-WM streaming rollout inputs."""

    num_frame_per_block: int = SANA_WM_STREAMING_LATENT_CHUNK_SIZE
    """Latent frames generated per steady-state AR block."""


@dataclass(kw_only=True)
class SanaWMStreamingRolloutState:
    """Cached full-rollout conditioning shared by every streaming AR step."""

    request: SanaWMStreamingI2VConditioningRequest
    """Most recently encoded request; identity preserves the offline fast path."""

    text: SanaWMTextConditioning
    first_latent: Tensor
    model_kwargs: dict[str, object]
    total_latent_shape: tuple[int, int, int, int, int]
    chunk_boundaries: tuple[int, ...]
    flow_shift: float
    steps: int
    seed: int
    cfg_scale: float


@dataclass(kw_only=True)
class SanaWMTextPromptEncoderConfig(EncoderConfig):
    """Config for the Stage-1 prompt encoder component."""

    _target: type["SanaWMTextPromptEncoder"] = field(
        default_factory=lambda: SanaWMTextPromptEncoder
    )

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    stage1_precision: str = "bf16"
    """Stage-1 precision; quantized paths pad text tokens for scaled-MM."""

    quant_backend: QuantBackend = "auto"
    """Low-precision backend selector, included in the prompt cache key."""

    offload_text_encoder: bool = False
    """Move the text encoder back to CPU after prompt encoding."""


class SanaWMTextPromptEncoder(Encoder):
    """Encode SANA-WM Stage-1 positive and negative prompts."""

    config: SanaWMTextPromptEncoderConfig

    def __init__(self, config: SanaWMTextPromptEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self._dummy = nn.Parameter(torch.empty(0))
        self._runtime_config: Any | None = None
        self._text_encoder_built = False
        self._prompt_cache: dict[
            tuple[object, ...], tuple[Tensor, Tensor, Tensor, Tensor]
        ] = {}

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def forward(self, input: SanaWMTextPromptRequest) -> SanaWMTextConditioning:
        """Encode prompt strings into Stage-1 text embeddings and masks."""
        cond, cond_mask, neg, neg_mask = self._encode_prompts(
            input.prompt,
            input.negative_prompt,
        )
        cond, cond_mask, neg, neg_mask = self._pad_text_for_quant(
            cond,
            cond_mask,
            neg,
            neg_mask,
        )
        return SanaWMTextConditioning(
            condition=cond,
            condition_mask=cond_mask,
            negative=neg,
            negative_mask=neg_mask,
        )

    def release_runtime(self) -> None:
        """Release prompt encoder tensors."""
        self._prompt_cache.clear()
        for attr in ("text_encoder", "tokenizer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._text_encoder_built = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_runtime_config(self) -> Any:
        if self._runtime_config is None:
            self._runtime_config = _load_inference_config(self.config.config_path)
        return self._runtime_config

    def _ensure_weight_dtype(self) -> torch.dtype:
        return _get_weight_dtype(self._ensure_runtime_config().model.mixed_precision)

    def _ensure_text_encoder(self) -> None:
        if self._text_encoder_built:
            return
        cfg = self._ensure_runtime_config()
        self.tokenizer, self.text_encoder = _get_tokenizer_and_text_encoder(
            name=cfg.text_encoder.text_encoder_name,
            device=self.device,
        )
        if self.config.offload_text_encoder:
            self.text_encoder.to("cpu")
        self._text_encoder_built = True

    def _encode_prompts(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        cfg = self._ensure_runtime_config()
        self._ensure_text_encoder()
        max_length = cfg.text_encoder.model_max_length
        chi_prompt = "\n".join(cfg.text_encoder.chi_prompt or [])
        if chi_prompt:
            prompt = chi_prompt + prompt
            max_length_all = len(self.tokenizer.encode(chi_prompt)) + max_length - 2
        else:
            max_length_all = max_length

        key = (
            prompt,
            negative_prompt,
            str(self.device),
            str(self._ensure_weight_dtype()),
            self.config.stage1_precision,
            self.config.quant_backend,
        )
        if key in self._prompt_cache:
            return self._prompt_cache[key]

        move_text_encoder = self.config.offload_text_encoder or (
            _module_device(self.text_encoder) != self.device
        )
        if move_text_encoder:
            self.text_encoder.to(self.device)

        def encode(text: str, length: int) -> tuple[Tensor, Tensor]:
            tokens = self.tokenizer(
                [text],
                max_length=length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            return self.text_encoder(tokens.input_ids, tokens.attention_mask)[0], (
                tokens.attention_mask
            )

        try:
            cond, cond_mask = encode(prompt, max_length_all)
            select = [0] + list(range(-max_length + 1, 0))
            cond = cond[:, None][:, :, select]
            cond_mask = cond_mask[:, select]
            neg, neg_mask = encode(negative_prompt, max_length)
            result = (cond, cond_mask, neg[:, None], neg_mask)
            self._prompt_cache.clear()
            self._prompt_cache[key] = result
            return result
        finally:
            if move_text_encoder:
                self.text_encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _pad_text_for_quant(
        self,
        cond: Tensor,
        cond_mask: Tensor,
        neg: Tensor,
        neg_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.config.stage1_precision == "bf16":
            return cond, cond_mask, neg, neg_mask
        multiple = int(os.environ.get("SANA_WM_STAGE1_NVFP4_TEXT_PAD_MULTIPLE", "8"))
        if multiple <= 1:
            return cond, cond_mask, neg, neg_mask

        def pad_pair(text: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
            pad = (-text.shape[-2]) % multiple
            if pad == 0:
                return text, mask
            text_shape = list(text.shape)
            text_shape[-2] = pad
            mask_shape = list(mask.shape)
            mask_shape[-1] = pad
            return (
                torch.cat([text, text.new_zeros(text_shape)], dim=-2),
                torch.cat([mask, mask.new_zeros(mask_shape)], dim=-1),
            )

        cond, cond_mask = pad_pair(cond, cond_mask)
        neg, neg_mask = pad_pair(neg, neg_mask)
        return cond, cond_mask, neg, neg_mask


@dataclass(kw_only=True)
class SanaWMFirstFrameEncoderConfig(EncoderConfig):
    """Config for the first-frame VAE encoder component."""

    _target: type["SanaWMFirstFrameEncoder"] = field(
        default_factory=lambda: SanaWMFirstFrameEncoder
    )

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    offload_vae: bool = False
    """Move the VAE to CPU after first-frame encoding."""


class SanaWMFirstFrameEncoder(Encoder):
    """Encode the input first frame with the LTX-2 VAE."""

    config: SanaWMFirstFrameEncoderConfig

    def __init__(self, config: SanaWMFirstFrameEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self._dummy = nn.Parameter(torch.empty(0))
        self._runtime_config: Any | None = None
        self._vae_built = False

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def forward(self, input: Any) -> Tensor:
        """Encode a PIL-like RGB image into a single latent frame."""
        from torchvision import transforms as T

        cfg = self._ensure_runtime_config()
        weight_dtype = _get_weight_dtype(cfg.model.mixed_precision)
        self._ensure_vae()
        if self.config.offload_vae:
            self.vae.to(self.device)

        if isinstance(input, Tensor):
            image = input
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
                raise ValueError(
                    "SANA-WM first-frame tensors must have shape [1, 3, H, W]."
                )
            image = image.unsqueeze(2)
        else:
            image = (T.ToTensor()(input) * 2.0 - 1.0).unsqueeze(0).unsqueeze(2)
        latent = _vae_encode_ltx2(
            cfg.vae.vae_type,
            self.vae,
            image.to(self.device, dtype=self.vae_dtype),
            device=self.device,
        ).to(weight_dtype)
        if self.config.offload_vae:
            self.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return latent

    def _ensure_runtime_config(self) -> Any:
        if self._runtime_config is None:
            self._runtime_config = _load_inference_config(self.config.config_path)
        return self._runtime_config

    def _ensure_vae(self) -> None:
        if self._vae_built:
            return
        cfg = self._ensure_runtime_config()
        self.vae_dtype = _get_weight_dtype(cfg.vae.weight_dtype)
        from sana_wm.impl._tools import resolve_hf_path

        cfg.vae.vae_pretrained = resolve_hf_path(cfg.vae.vae_pretrained)
        self.vae = _get_vae(
            cfg.vae.vae_type,
            cfg.vae.vae_pretrained,
            device=self.device,
            dtype=self.vae_dtype,
            config=cfg.vae,
        )
        self._vae_built = True


@dataclass(kw_only=True)
class SanaWMCameraConditioningEncoderConfig(EncoderConfig):
    """Config for the camera/raymap conditioning component."""

    _target: type["SanaWMCameraConditioningEncoder"] = field(
        default_factory=lambda: SanaWMCameraConditioningEncoder
    )

    height: int = DEFAULT_VIDEO_HEIGHT
    width: int = DEFAULT_VIDEO_WIDTH


class SanaWMCameraConditioningEncoder(Encoder):
    """Build SANA-WM raymap and chunk-Plucker camera tensors."""

    config: SanaWMCameraConditioningEncoderConfig

    def __init__(self, config: SanaWMCameraConditioningEncoderConfig) -> None:
        super().__init__(config)
        self.config = config

    def forward(self, input: SanaWMCameraRequest) -> dict[str, Tensor]:
        """Encode raw poses and intrinsics into Stage-1 camera tensors."""
        return prepare_camera(
            input.poses_c2w,
            input.intrinsics_vec4,
            target_size=(self.config.height, self.config.width),
        )


@dataclass(kw_only=True)
class SanaWMConditioningEncoderConfig(EncoderConfig):
    """Config for the pipeline-level SANA-WM conditioning encoder."""

    _target: type["SanaWMConditioningEncoder"] = field(
        default_factory=lambda: SanaWMConditioningEncoder
    )

    config_path: str = SANA_WM_CONFIG_PATH
    """SANA-WM inference YAML path or ``hf://`` URI."""

    text_encoder: SanaWMTextPromptEncoderConfig = field(
        default_factory=SanaWMTextPromptEncoderConfig
    )
    first_frame_encoder: SanaWMFirstFrameEncoderConfig = field(
        default_factory=SanaWMFirstFrameEncoderConfig
    )
    camera_encoder: SanaWMCameraConditioningEncoderConfig = field(
        default_factory=SanaWMCameraConditioningEncoderConfig
    )
    height: int = DEFAULT_VIDEO_HEIGHT
    width: int = DEFAULT_VIDEO_WIDTH


class SanaWMConditioningEncoder(StreamingEncoder[StreamingEncoderCache]):
    """Prepare all per-rollout inputs for the Sana Stage-1 sampler."""

    config: SanaWMConditioningEncoderConfig

    def __init__(self, config: SanaWMConditioningEncoderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.text_encoder = config.text_encoder.setup()
        self.first_frame_encoder = config.first_frame_encoder.setup()
        self.camera_encoder = config.camera_encoder.setup()
        self._runtime_config: Any | None = None

    def initialize_autoregressive_cache(self, **_context: Any) -> StreamingEncoderCache:
        """Return an empty cache for the one-shot conditioning encoder."""
        return StreamingEncoderCache()

    @torch.inference_mode()
    def forward(
        self,
        input: SanaWMI2VConditioningRequest,
        autoregressive_index: int = 0,
        cache: StreamingEncoderCache | None = None,
    ) -> SanaWMStage1Conditioning:
        """Encode raw rollout inputs into Stage-1 conditioning."""
        del cache
        if autoregressive_index != 0:
            raise ValueError("SANA-WM bidirectional inference has one AR step.")

        cfg = self._ensure_runtime_config()
        text = self.text_encoder(
            SanaWMTextPromptRequest(
                prompt=input.prompt,
                negative_prompt=input.negative_prompt,
            )
        )
        first_latent = self.first_frame_encoder(input.image)
        weight_dtype = _get_weight_dtype(cfg.model.mixed_precision)
        camera = self.camera_encoder(
            SanaWMCameraRequest(
                poses_c2w=input.poses_c2w,
                intrinsics_vec4=input.intrinsics_vec4,
            )
        )
        raymap = (
            camera["raymap"]
            .unsqueeze(0)
            .to(
                first_latent.device,
                dtype=weight_dtype,
            )
        )
        chunk_plucker = (
            camera["chunk_plucker"]
            .unsqueeze(0)
            .to(
                first_latent.device,
                dtype=weight_dtype,
            )
        )

        model_kwargs_extra: dict[str, object] = {}
        if input.cfg_scale > 1.0:
            model_kwargs_extra["negative_mask"] = text.negative_mask
            uncondition = text.negative
        else:
            uncondition = None

        vae_stride = cfg.vae.vae_stride
        latent_t = (input.num_frames - 1) // int(vae_stride[0]) + 1
        latent_h = self.config.height // int(vae_stride[-1])
        latent_w = self.config.width // int(vae_stride[-1])
        chunk_index = _chunk_index_from_config(cfg, num_frames=latent_t)
        model_kwargs: dict[str, object] = {
            "data_info": {
                "img_hw": torch.tensor(
                    [[self.config.height, self.config.width]],
                    dtype=torch.float,
                    device=first_latent.device,
                ),
                "condition_frame_info": {0: 0.0},
            },
            "mask": text.condition_mask,
            "camera_conditions": raymap,
            "chunk_plucker": chunk_plucker,
            **model_kwargs_extra,
        }
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index

        return SanaWMStage1Conditioning(
            condition=text.condition,
            uncondition=uncondition,
            model_kwargs=model_kwargs,
            first_latent=first_latent,
            latent_shape=(
                1,
                int(first_latent.shape[1]),
                latent_t,
                latent_h,
                latent_w,
            ),
            cfg_scale=float(input.cfg_scale),
            flow_shift=self._resolve_flow_shift(input.flow_shift),
            steps=int(input.steps),
            seed=int(input.seed),
        )

    def _ensure_runtime_config(self) -> Any:
        if self._runtime_config is None:
            self._runtime_config = _load_inference_config(self.config.config_path)
        return self._runtime_config

    def _resolve_flow_shift(self, override: float | None) -> float:
        cfg = self._ensure_runtime_config()
        if override is not None:
            return float(override)
        if cfg.scheduler.inference_flow_shift is not None:
            return float(cfg.scheduler.inference_flow_shift)
        return float(cfg.scheduler.flow_shift)


@dataclass(kw_only=True)
class SanaWMStreamingConditioningEncoderCache(StreamingEncoderCache):
    """Per-rollout cache for streaming SANA-WM conditioning."""

    rollout: SanaWMStreamingRolloutState | None = None


@dataclass(kw_only=True)
class SanaWMStreamingConditioningEncoderConfig(SanaWMConditioningEncoderConfig):
    """Config for the pipeline-level streaming SANA-WM conditioning encoder."""

    _target: type["SanaWMStreamingConditioningEncoder"] = field(
        default_factory=lambda: SanaWMStreamingConditioningEncoder
    )

    config_path: str = SANA_WM_STREAMING_CONFIG_PATH
    """SANA-WM streaming inference config path or built-in identifier."""

    text_encoder: SanaWMTextPromptEncoderConfig = field(
        default_factory=lambda: SanaWMTextPromptEncoderConfig(
            config_path=SANA_WM_STREAMING_CONFIG_PATH
        )
    )
    first_frame_encoder: SanaWMFirstFrameEncoderConfig = field(
        default_factory=lambda: SanaWMFirstFrameEncoderConfig(
            config_path=SANA_WM_STREAMING_CONFIG_PATH
        )
    )


class SanaWMStreamingConditioningEncoder(SanaWMConditioningEncoder):
    """Prepare SANA-WM streaming rollout conditioning once, then slice AR chunks."""

    config: SanaWMStreamingConditioningEncoderConfig

    def initialize_autoregressive_cache(
        self,
        **_context: Any,
    ) -> SanaWMStreamingConditioningEncoderCache:
        """Return an initially empty streaming conditioning cache."""
        return SanaWMStreamingConditioningEncoderCache()

    @torch.inference_mode()
    def forward(
        self,
        input: SanaWMStreamingI2VConditioningRequest,
        autoregressive_index: int = 0,
        cache: SanaWMStreamingConditioningEncoderCache | None = None,
    ) -> SanaWMStreamingStage1Conditioning:
        """Return the Stage-1 conditioning payload for one streaming chunk."""
        if cache is None:
            cache = SanaWMStreamingConditioningEncoderCache()
        if cache.rollout is None:
            if autoregressive_index != 0:
                raise ValueError(
                    "SANA-WM streaming conditioning must start at AR step 0."
                )
            cache.rollout = self._build_rollout(input)
        elif cache.rollout.request is not input:
            cache.rollout = self._update_rollout(cache.rollout, input)

        rollout = cache.rollout
        chunk_count = len(rollout.chunk_boundaries) - 1
        if autoregressive_index < 0 or autoregressive_index >= chunk_count:
            raise ValueError(
                f"autoregressive_index={autoregressive_index} is outside the "
                f"streaming rollout with {chunk_count} chunks."
            )
        start = rollout.chunk_boundaries[autoregressive_index]
        end = rollout.chunk_boundaries[autoregressive_index + 1]
        latent_shape = (
            rollout.total_latent_shape[0],
            rollout.total_latent_shape[1],
            end - start,
            rollout.total_latent_shape[3],
            rollout.total_latent_shape[4],
        )
        model_kwargs = dict(rollout.model_kwargs)
        data_info = model_kwargs.get("data_info")
        if isinstance(data_info, dict):
            model_kwargs["data_info"] = dict(data_info)

        return SanaWMStreamingStage1Conditioning(
            condition=rollout.text.condition,
            uncondition=(rollout.text.negative if rollout.cfg_scale > 1.0 else None),
            model_kwargs=model_kwargs,
            first_latent=rollout.first_latent,
            latent_shape=latent_shape,
            cfg_scale=rollout.cfg_scale,
            flow_shift=rollout.flow_shift,
            steps=rollout.steps,
            seed=rollout.seed,
            total_latent_shape=rollout.total_latent_shape,
            start_frame=start,
            end_frame=end,
            chunk_index=autoregressive_index,
            chunk_boundaries=rollout.chunk_boundaries,
        )

    def _build_rollout(
        self,
        input: SanaWMStreamingI2VConditioningRequest,
    ) -> SanaWMStreamingRolloutState:
        if input.num_frames != len(input.poses_c2w) or input.num_frames != len(
            input.intrinsics_vec4
        ):
            raise ValueError("SANA-WM streaming camera history must match num_frames.")
        cfg = self._ensure_runtime_config()
        text = self.text_encoder(
            SanaWMTextPromptRequest(
                prompt=input.prompt,
                negative_prompt=input.negative_prompt,
            )
        )
        first_latent = self.first_frame_encoder(input.image)
        weight_dtype = _get_weight_dtype(cfg.model.mixed_precision)
        camera = self.camera_encoder(
            SanaWMCameraRequest(
                poses_c2w=input.poses_c2w,
                intrinsics_vec4=input.intrinsics_vec4,
            )
        )
        raymap = (
            camera["raymap"]
            .unsqueeze(0)
            .to(
                first_latent.device,
                dtype=weight_dtype,
            )
        )
        chunk_plucker = (
            camera["chunk_plucker"]
            .unsqueeze(0)
            .to(
                first_latent.device,
                dtype=weight_dtype,
            )
        )

        vae_stride = cfg.vae.vae_stride
        latent_t = (input.num_frames - 1) // int(vae_stride[0]) + 1
        latent_h = self.config.height // int(vae_stride[-1])
        latent_w = self.config.width // int(vae_stride[-1])
        chunk_boundaries = streaming_chunk_boundaries(
            latent_t,
            int(input.num_frame_per_block),
        )

        model_kwargs_extra: dict[str, object] = {}
        if input.cfg_scale > 1.0:
            model_kwargs_extra["negative_mask"] = text.negative_mask

        chunk_index = _chunk_index_from_config(cfg, num_frames=latent_t)
        model_kwargs: dict[str, object] = {
            "data_info": {
                "img_hw": torch.tensor(
                    [[self.config.height, self.config.width]],
                    dtype=torch.float,
                    device=first_latent.device,
                ),
                "condition_frame_info": {0: 0.0},
            },
            "mask": text.condition_mask,
            "camera_conditions": raymap,
            "chunk_plucker": chunk_plucker,
            **model_kwargs_extra,
        }
        if chunk_index is not None:
            model_kwargs["chunk_index"] = chunk_index

        return SanaWMStreamingRolloutState(
            request=input,
            text=text,
            first_latent=first_latent,
            model_kwargs=model_kwargs,
            total_latent_shape=(
                1,
                int(first_latent.shape[1]),
                latent_t,
                latent_h,
                latent_w,
            ),
            chunk_boundaries=chunk_boundaries,
            flow_shift=self._resolve_flow_shift(input.flow_shift),
            steps=int(input.steps),
            seed=int(input.seed),
            cfg_scale=float(input.cfg_scale),
        )

    def _update_rollout(
        self,
        rollout: SanaWMStreamingRolloutState,
        input: SanaWMStreamingI2VConditioningRequest,
    ) -> SanaWMStreamingRolloutState:
        """Extend cached static conditioning with the latest camera history."""
        if input.num_frames != len(input.poses_c2w) or input.num_frames != len(
            input.intrinsics_vec4
        ):
            raise ValueError("SANA-WM streaming camera history must match num_frames.")

        cfg = self._ensure_runtime_config()
        camera = self.camera_encoder(
            SanaWMCameraRequest(
                poses_c2w=input.poses_c2w,
                intrinsics_vec4=input.intrinsics_vec4,
            )
        )
        weight_dtype = _get_weight_dtype(cfg.model.mixed_precision)
        device = rollout.first_latent.device
        raymap = camera["raymap"].unsqueeze(0).to(device, dtype=weight_dtype)
        chunk_plucker = (
            camera["chunk_plucker"]
            .unsqueeze(0)
            .to(
                device,
                dtype=weight_dtype,
            )
        )

        vae_stride = cfg.vae.vae_stride
        latent_t = (input.num_frames - 1) // int(vae_stride[0]) + 1
        chunk_boundaries = streaming_chunk_boundaries(
            latent_t,
            int(input.num_frame_per_block),
        )
        model_kwargs = dict(rollout.model_kwargs)
        model_kwargs["camera_conditions"] = raymap
        model_kwargs["chunk_plucker"] = chunk_plucker
        chunk_index = _chunk_index_from_config(cfg, num_frames=latent_t)
        if chunk_index is None:
            model_kwargs.pop("chunk_index", None)
        else:
            model_kwargs["chunk_index"] = chunk_index

        rollout.request = input
        rollout.model_kwargs = model_kwargs
        rollout.total_latent_shape = (
            rollout.total_latent_shape[0],
            rollout.total_latent_shape[1],
            latent_t,
            rollout.total_latent_shape[3],
            rollout.total_latent_shape[4],
        )
        rollout.chunk_boundaries = chunk_boundaries
        rollout.flow_shift = self._resolve_flow_shift(input.flow_shift)
        rollout.steps = int(input.steps)
        rollout.seed = int(input.seed)
        rollout.cfg_scale = float(input.cfg_scale)
        return rollout


def streaming_chunk_boundaries(total_frames: int, chunk_size: int) -> tuple[int, ...]:
    """Return SANA-WM streaming latent-frame AR chunk boundaries."""
    if total_frames <= 1:
        raise ValueError(
            f"SANA-WM streaming requires more than one latent frame; got {total_frames}."
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    active = total_frames - 1
    if active % chunk_size != 0:
        raise ValueError(
            "SANA-WM streaming active latent frames must divide the chunk size: "
            f"active={active}, chunk_size={chunk_size}."
        )
    boundaries = [0, 1 + chunk_size]
    while boundaries[-1] < total_frames:
        boundaries.append(boundaries[-1] + chunk_size)
    return tuple(min(boundary, total_frames) for boundary in boundaries)


def resolve_sana_wm_conditioning(
    values: Mapping[str, Any],
) -> Cam2VConditioning:
    """Resolve SANA-WM first-frame, prompt, and camera calibration inputs."""
    example_idx = int(values.get("example_idx", 0))
    if example_idx not in _EXAMPLE_DATA_INDICES:
        raise ValueError(
            f"SANA-WM example_idx must be one of {sorted(_EXAMPLE_DATA_INDICES)}."
        )

    image_path = _optional_path(values.get("image_path"))
    prompt = " ".join(str(values.get("prompt") or "").split())
    prompt_path = _optional_path(values.get("prompt_path"))
    if _as_bool(values.get("example_data", False)):
        example_dir = _ensure_example_data(example_idx)
        image_path = image_path or example_dir / f"demo_{example_idx}.png"
        if not prompt and prompt_path is None:
            prompt_path = example_dir / f"demo_{example_idx}.txt"

    if image_path is None:
        raise ValueError("SANA-WM Cam2V requires image_path.")
    if not image_path.is_file():
        raise FileNotFoundError(f"SANA-WM Cam2V missing image_path: {image_path}")

    if not prompt and prompt_path is not None:
        if not prompt_path.is_file():
            raise FileNotFoundError(f"SANA-WM Cam2V missing prompt_path: {prompt_path}")
        lines = [
            line.strip()
            for line in prompt_path.read_text(encoding="utf-8").splitlines()
        ]
        prompt = next((line for line in lines if line), "")
    if not prompt:
        raise ValueError("SANA-WM Cam2V requires prompt or prompt_path.")

    with Image.open(image_path) as source:
        source_size = source.size
    pixel_height = int(values.get("pixel_height", DEFAULT_VIDEO_HEIGHT))
    pixel_width = int(values.get("pixel_width", DEFAULT_VIDEO_WIDTH))
    resized_size, crop_offset = resize_center_crop_geometry(
        source_size,
        target_h=pixel_height,
        target_w=pixel_width,
    )
    intrinsic_path = _optional_path(values.get("intrinsic_path"))
    if intrinsic_path is None:
        intrinsics = default_intrinsics_vec4(source_size, 1)
    else:
        if not intrinsic_path.is_file():
            raise FileNotFoundError(
                f"SANA-WM Cam2V missing intrinsic_path: {intrinsic_path}"
            )
        intrinsics = load_intrinsics(intrinsic_path, 1)
    intrinsics = transform_intrinsics_for_crop(
        intrinsics,
        source_size,
        resized_size,
        crop_offset,
    )

    world_scale_value = values.get("world_scale")
    world_scale = 1.0 if world_scale_value in (None, "") else float(world_scale_value)
    if world_scale <= 0:
        raise ValueError("SANA-WM Cam2V world_scale must be > 0.")
    return Cam2VConditioning(
        prompt=prompt,
        first_frame_path=image_path,
        base_intrinsics=torch.from_numpy(intrinsics[0]),
        world_scale=world_scale,
    )


def _ensure_example_data(example_idx: int) -> Path:
    cache_dir = _EXAMPLE_DATA_DIR / f"{example_idx:02d}"
    distributed = torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        for suffix in ("png", "txt"):
            filename = f"demo_{example_idx}.{suffix}"
            download_to_cache(
                f"{_EXAMPLE_DATA_BASE_URL}/{filename}",
                cache_dir=cache_dir,
                filename=filename,
            )
    if distributed:
        torch.distributed.barrier()
    return cache_dir


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _optional_path(value: object) -> Path | None:
    return None if value is None or value == "" else Path(str(value))


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


__all__ = [
    "SanaWMCameraConditioningEncoder",
    "SanaWMCameraConditioningEncoderConfig",
    "SanaWMCameraRequest",
    "SanaWMConditioningEncoder",
    "SanaWMConditioningEncoderConfig",
    "SanaWMFirstFrameEncoder",
    "SanaWMFirstFrameEncoderConfig",
    "SanaWMI2VConditioning",
    "SanaWMI2VConditioningRequest",
    "SanaWMStreamingConditioningEncoder",
    "SanaWMStreamingConditioningEncoderCache",
    "SanaWMStreamingConditioningEncoderConfig",
    "SanaWMStreamingI2VConditioningRequest",
    "resolve_sana_wm_conditioning",
    "streaming_chunk_boundaries",
    "SanaWMTextConditioning",
    "SanaWMTextPromptEncoder",
    "SanaWMTextPromptEncoderConfig",
    "SanaWMTextPromptRequest",
]
