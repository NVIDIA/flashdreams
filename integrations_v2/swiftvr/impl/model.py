# SPDX-FileCopyrightText: Copyright (c) 2026 SwiftVR Authors.
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

"""Minimal SwiftVR streaming inference runtime.

The ReAE and causal streaming protocol are adapted from SwiftVR commit
``5ca168cef6ca7200f135fdfea85e5e13d12c5b53``. The WAN transformer itself is
provided by the installed Diffusers package and receives SwiftVR attention via
:mod:`swiftvr.impl.attention`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import Tensor, nn

from flashdreams.core.io.hf import maybe_download_hf_repo_on_rank0
from swiftvr.impl.attention import prepare_transformer

_CHECKPOINT_PATTERNS = (
    "reae.safetensors",
    "prompt_embedding.safetensors",
    "transformer/config.json",
    "transformer/*.safetensors",
)
_INFERENCE_TIMESTEP = 1000.0
_ROPE_EXTEND_MARGIN = 256


def _convolution(input_channels: int, output_channels: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(input_channels, output_channels, 3, padding=1, **kwargs)


class _Clamp(nn.Module):
    def forward(self, tensor: Tensor) -> Tensor:
        """Soft-clamp autoencoder latents."""
        return torch.tanh(tensor / 3) * 3


class _MemoryBlock(nn.Module):
    """Fuse each frame with the preceding frame."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            _convolution(input_channels * 2, output_channels),
            nn.ReLU(inplace=True),
            _convolution(output_channels, output_channels),
            nn.ReLU(inplace=True),
            _convolution(output_channels, output_channels),
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1, bias=False)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, tensor: Tensor, previous: Tensor) -> Tensor:
        """Apply the residual block with one preceding frame."""
        return self.act(
            self.conv(torch.cat([tensor, previous], dim=1)) + self.skip(tensor)
        )


class _TemporalPool(nn.Module):
    """Reduce the temporal axis through a channel projection."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels * stride, channels, 1, bias=False)

    def forward(self, tensor: Tensor) -> Tensor:
        """Pool ``stride`` adjacent frames."""
        _, channels, height, width = tensor.shape
        return self.conv(tensor.reshape(-1, self.stride * channels, height, width))


class _TemporalGrow(nn.Module):
    """Expand the temporal axis through nearest interpolation and projection."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.n_f = channels
        self.proj = (
            nn.Conv2d(channels, channels, 1, bias=False) if stride == 1 else None
        )
        self.conv3d = (
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            )
            if stride != 1
            else None
        )

    def forward(self, tensor: Tensor) -> Tensor:
        """Grow one flattened frame batch."""
        if self.stride == 1:
            assert self.proj is not None
            return self.proj(tensor)
        assert self.conv3d is not None
        frames, channels, height, width = tensor.shape
        tensor = F.interpolate(
            tensor.unsqueeze(2),
            size=(self.stride, height, width),
            mode="nearest",
        )
        tensor = self.conv3d(tensor)
        return tensor.permute(0, 2, 1, 3, 4).reshape(
            frames * self.stride, channels, height, width
        )


class RestorationAutoencoder(nn.Module):
    """SwiftVR restoration-aware streaming autoencoder."""

    def __init__(self) -> None:
        super().__init__()
        encoder_channels = 64
        decoder_channels = (512, 256, 128, 64)
        self.patch_size = 2
        self.frames_to_trim = 3
        self.encoder = nn.Sequential(
            _convolution(12, encoder_channels),
            nn.ReLU(inplace=True),
            _TemporalPool(encoder_channels, 2),
            _convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            *[_MemoryBlock(encoder_channels, encoder_channels) for _ in range(3)],
            _TemporalPool(encoder_channels, 2),
            _convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            *[_MemoryBlock(encoder_channels, encoder_channels) for _ in range(3)],
            _TemporalPool(encoder_channels, 1),
            _convolution(encoder_channels, encoder_channels, stride=2, bias=False),
            *[_MemoryBlock(encoder_channels, encoder_channels) for _ in range(3)],
            _convolution(encoder_channels, 48),
        )
        first, second, third, fourth = decoder_channels
        self.decoder = nn.Sequential(
            _Clamp(),
            _convolution(48, first),
            nn.ReLU(inplace=True),
            *[_MemoryBlock(first, first) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(first, 1),
            _convolution(first, second, bias=False),
            *[_MemoryBlock(second, second) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(second, 2),
            _convolution(second, third, bias=False),
            *[_MemoryBlock(third, third) for _ in range(3)],
            nn.Upsample(scale_factor=2),
            _TemporalGrow(third, 2),
            _convolution(third, fourth, bias=False),
            nn.ReLU(inplace=True),
            _convolution(fourth, 12),
        )


def _apply_with_boundaries(
    model: nn.Sequential,
    tensor: Tensor,
    state: dict[str, Tensor | None] | None,
) -> tuple[Tensor | None, dict[str, Tensor | None]]:
    state = state or {}
    next_state: dict[str, Tensor | None] = {}
    batch, time, channels, height, width = tensor.shape
    tensor = tensor.reshape(batch * time, channels, height, width)
    for index, block in enumerate(model):
        if isinstance(block, _MemoryBlock):
            _, channels, height, width = tensor.shape
            time = tensor.shape[0] // batch
            video = tensor.reshape(batch, time, channels, height, width)
            key = f"memory_{index}"
            previous_state = state.get(key)
            if previous_state is not None:
                previous = torch.cat([previous_state, video[:, :-1]], dim=1)
            else:
                previous = F.pad(video, (0, 0, 0, 0, 0, 0, 1, 0))[:, :time]
            next_state[key] = video[:, -1:].detach().clone()
            tensor = block(tensor, previous.reshape_as(tensor))
        elif isinstance(block, _TemporalPool):
            _, channels, height, width = tensor.shape
            time = tensor.shape[0] // batch
            video = tensor.reshape(batch, time, channels, height, width)
            key = f"pool_{index}"
            previous = state.get(key)
            if previous is not None:
                video = torch.cat([previous, video], dim=1)
                time = video.shape[1]
            full_frames = time // block.stride * block.stride
            next_state[key] = (
                video[:, full_frames:].detach().clone() if full_frames != time else None
            )
            if full_frames == 0:
                return None, next_state
            tensor = block(
                video[:, :full_frames].reshape(
                    batch * full_frames, channels, height, width
                )
            )
        else:
            tensor = block(tensor)
    _, channels, height, width = tensor.shape
    return (
        tensor.view(batch, tensor.shape[0] // batch, channels, height, width),
        next_state,
    )


class _StreamingAutoencoder:
    def __init__(self, model: RestorationAutoencoder) -> None:
        self.model = model
        self.encoder_state: dict[str, Tensor | None] | None = None
        self.decoder_state: dict[str, Tensor | None] | None = None
        self.encoder_tail: Tensor | None = None
        self.first_decode = True

    def encode(self, tensor: Tensor) -> Tensor | None:
        """Encode all complete four-frame groups and retain the tail."""
        batch, time, channels, height, width = tensor.shape
        tensor = F.pixel_unshuffle(
            tensor.reshape(batch * time, channels, height, width),
            self.model.patch_size,
        ).reshape(batch, time, -1, height // 2, width // 2)
        if self.encoder_tail is not None:
            tensor = torch.cat([self.encoder_tail, tensor], dim=1)
        remainder = tensor.shape[1] % 4
        if remainder:
            self.encoder_tail = tensor[:, -remainder:].detach().clone()
            tensor = tensor[:, :-remainder]
        else:
            self.encoder_tail = None
        if tensor.shape[1] == 0:
            return None
        encoded, self.encoder_state = _apply_with_boundaries(
            self.model.encoder, tensor, self.encoder_state
        )
        return encoded

    def flush_encoder(self) -> Tensor | None:
        """Replicate-pad and encode the final partial temporal group."""
        if self.encoder_tail is None:
            return None
        tensor = self.encoder_tail
        self.encoder_tail = None
        padding = (-tensor.shape[1]) % 4
        if padding:
            tensor = torch.cat(
                [tensor, tensor[:, -1:].expand(-1, padding, -1, -1, -1)], dim=1
            )
        encoded, self.encoder_state = _apply_with_boundaries(
            self.model.encoder, tensor, self.encoder_state
        )
        return encoded

    def decode(self, tensor: Tensor) -> Tensor | None:
        """Decode one latent chunk while carrying temporal boundary state."""
        decoded, self.decoder_state = _apply_with_boundaries(
            self.model.decoder, tensor, self.decoder_state
        )
        if decoded is None:
            return None
        decoded = decoded.clamp_(0, 1)
        batch, time, channels, height, width = decoded.shape
        decoded = F.pixel_shuffle(
            decoded.reshape(batch * time, channels, height, width),
            self.model.patch_size,
        ).reshape(batch, time, 3, height * 2, width * 2)
        if self.first_decode:
            decoded = decoded[:, self.model.frames_to_trim :]
            self.first_decode = False
        return decoded


def _extend_rope(rope: Any, required_length: int) -> None:
    cosine, sine = rope.freqs_cos, rope.freqs_sin
    old_length = cosine.shape[0]
    if required_length <= old_length:
        return
    if old_length < 2:
        raise RuntimeError(f"Cannot extend SwiftVR RoPE cache of length {old_length}.")
    with torch.no_grad():
        cosine64 = cosine.to(torch.float64)
        sine64 = sine.to(torch.float64)
        cosine_delta = cosine64[1:2] * cosine64[0:1] + sine64[1:2] * sine64[0:1]
        sine_delta = sine64[1:2] * cosine64[0:1] - cosine64[1:2] * sine64[0:1]
        start = torch.atan2(sine64[0:1], cosine64[0:1])
        step = torch.atan2(sine_delta, cosine_delta)
        positions = torch.arange(
            old_length,
            required_length,
            device=cosine.device,
            dtype=torch.float64,
        ).view(-1, 1)
        angles = start + positions * step
        rope.freqs_cos = torch.cat(
            [cosine, torch.cos(angles).to(cosine.dtype)], dim=0
        ).contiguous()
        rope.freqs_sin = torch.cat(
            [sine, torch.sin(angles).to(sine.dtype)], dim=0
        ).contiguous()


def _rope_with_offset(
    rope: Any,
    frames: int,
    height: int,
    width: int,
    time_offset: int,
) -> tuple[Tensor, Tensor]:
    required_length = max(time_offset + frames, height, width)
    if required_length > rope.freqs_cos.shape[0]:
        _extend_rope(rope, required_length + _ROPE_EXTEND_MARGIN)
    splits = (rope.t_dim, rope.h_dim, rope.w_dim)
    cosine = rope.freqs_cos.split(splits, dim=1)
    sine = rope.freqs_sin.split(splits, dim=1)

    def grid(parts: tuple[Tensor, ...]) -> Tensor:
        time = parts[0][time_offset : time_offset + frames].view(frames, 1, 1, -1)
        vertical = parts[1][:height].view(1, height, 1, -1)
        horizontal = parts[2][:width].view(1, 1, width, -1)
        return torch.cat(
            [
                time.expand(frames, height, width, -1),
                vertical.expand(frames, height, width, -1),
                horizontal.expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(1, frames * height * width, 1, -1)

    return grid(cosine), grid(sine)


def _condition(
    transformer: Any,
    prompt: Tensor,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    prompt = prompt.to(device=device, dtype=dtype)
    if prompt.ndim == 2:
        prompt = prompt.unsqueeze(0).expand(batch, -1, -1)
    timestep = torch.full(
        (batch,), _INFERENCE_TIMESTEP, device=device, dtype=torch.float32
    )
    embedding, projection, encoder, image = transformer.condition_embedder(
        timestep, prompt, None
    )
    projection = projection.unflatten(1, (6, -1))
    if image is not None:
        encoder = torch.cat([image, encoder], dim=1)
    return embedding, projection, encoder


def _transformer_chunk(
    transformer: Any,
    tensor: Tensor,
    condition: tuple[Tensor, Tensor, Tensor],
    *,
    time_offset: int,
) -> Tensor:
    embedding, projection, encoder = condition
    patch_time, patch_height, patch_width = transformer.config.patch_size
    batch, _, frames, height, width = tensor.shape
    frames //= patch_time
    height //= patch_height
    width //= patch_width
    rope = _rope_with_offset(transformer.rope, frames, height, width, time_offset)
    hidden = transformer.patch_embedding(tensor).flatten(2).transpose(1, 2)
    shape = (frames, height, width)
    for block in transformer.blocks:
        underlying = getattr(block, "_orig_mod", block)
        underlying.attn1._swiftvr_shape = shape
    for block in transformer.blocks:
        hidden = block(hidden, encoder, projection, rope)
    if embedding.ndim == 3:
        shift, scale = (
            transformer.scale_shift_table.unsqueeze(0) + embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
    else:
        shift, scale = (transformer.scale_shift_table + embedding.unsqueeze(1)).chunk(
            2, dim=1
        )
    hidden = (
        transformer.norm_out(hidden.float()) * (1 + scale.to(hidden.device))
        + shift.to(hidden.device)
    ).type_as(hidden)
    hidden = transformer.proj_out(hidden)
    hidden = hidden.reshape(
        batch,
        frames,
        height,
        width,
        patch_time,
        patch_height,
        patch_width,
        -1,
    ).permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden.flatten(6, 7).flatten(4, 5).flatten(2, 3)


class _StreamingTransformer:
    def __init__(self, transformer: Any, prompt: Tensor, overlap: int) -> None:
        self.transformer = transformer
        self.prompt = prompt
        self.overlap = overlap
        self.previous_input: Tensor | None = None
        self.previous_output: Tensor | None = None
        self.time_offset = 0
        self.condition: tuple[Tensor, Tensor, Tensor] | None = None

    @torch.inference_mode()
    def denoise(self, tensor: Tensor) -> Tensor:
        """Restore one ``[B, C, T, H, W]`` latent chunk."""
        batch, _, frames, _, _ = tensor.shape
        overlap = 0
        if self.previous_input is not None and self.overlap:
            overlap = self.previous_input.shape[2]
            extended = torch.cat([self.previous_input.to(tensor.device), tensor], dim=2)
        else:
            extended = tensor
        if self.condition is None:
            self.condition = _condition(
                self.transformer,
                self.prompt,
                batch,
                tensor.device,
                tensor.dtype,
            )
        prediction = _transformer_chunk(
            self.transformer,
            extended,
            self.condition,
            time_offset=self.time_offset - overlap,
        )
        restored = extended - prediction
        if overlap and self.previous_output is not None:
            blend = torch.linspace(
                0, 1, overlap, device=tensor.device, dtype=tensor.dtype
            ).view(1, 1, overlap, 1, 1)
            restored[:, :, :overlap] = (
                self.previous_output.to(tensor.device) * (1 - blend)
                + restored[:, :, :overlap] * blend
            )
            restored = restored[:, :, overlap:]
        retained = min(self.overlap, frames)
        if retained:
            self.previous_input = tensor[:, :, -retained:].detach().cpu().clone()
            self.previous_output = restored[:, :, -retained:].detach().cpu().clone()
        self.time_offset += frames
        return restored


class SwiftVRStream:
    """Per-video causal SwiftVR state."""

    def __init__(
        self,
        pipeline: "SwiftVRPipeline",
        *,
        output_height: int,
        output_width: int,
        overlap: int,
    ) -> None:
        self.pipeline = pipeline
        self.output_height = output_height
        self.output_width = output_width
        self.pad_height = (-output_height) % 32
        self.pad_width = (-output_width) % 32
        self.autoencoder = _StreamingAutoencoder(pipeline.autoencoder)
        self.transformer = _StreamingTransformer(
            pipeline.transformer, pipeline.prompt_embedding, overlap
        )

    def _preprocess(self, frames: Tensor) -> Tensor:
        frames = frames.to(self.pipeline.device)
        frames = frames.permute(0, 3, 1, 2).to(dtype=self.pipeline.dtype)
        frames = F.interpolate(
            frames,
            size=(self.output_height, self.output_width),
            mode="bilinear",
            align_corners=False,
        ).div_(255)
        if self.pad_height or self.pad_width:
            frames = F.pad(frames, (0, self.pad_width, 0, self.pad_height))
        return frames.unsqueeze(0)

    def _run_latents(self, latents: Tensor) -> Tensor | None:
        latent_channels_first = latents.permute(0, 2, 1, 3, 4).contiguous()
        restored = self.transformer.denoise(latent_channels_first)
        decoded = self.autoencoder.decode(restored.permute(0, 2, 1, 3, 4).contiguous())
        if decoded is None:
            return None
        return decoded[
            :,
            :,
            :,
            : self.output_height,
            : self.output_width,
        ]

    @torch.inference_mode()
    def step(self, frames_uint8: Tensor) -> Tensor | None:
        """Process ``[T, H, W, 3]`` uint8 frames and return ``[B, T, C, H, W]``."""
        latents = self.autoencoder.encode(self._preprocess(frames_uint8))
        return None if latents is None else self._run_latents(latents)

    @torch.inference_mode()
    def flush(self) -> Tensor | None:
        """Flush the final autoencoder temporal group."""
        latents = self.autoencoder.flush_encoder()
        return None if latents is None else self._run_latents(latents)


class SwiftVRPipeline:
    """Resident SwiftVR weights with cheap per-stream state construction."""

    def __init__(
        self,
        autoencoder: RestorationAutoencoder,
        transformer: Any,
        prompt_embedding: Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.autoencoder = autoencoder
        self.transformer = transformer
        self.prompt_embedding = prompt_embedding
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None,
        device: str,
        dtype: torch.dtype,
        attention_window: tuple[int, int],
        compile_blocks: bool,
    ) -> "SwiftVRPipeline":
        """Load and prepare one SwiftVR checkpoint."""
        checkpoint_root = _resolve_checkpoint(checkpoint, revision=revision)
        autoencoder = RestorationAutoencoder()
        autoencoder.load_state_dict(
            load_file(str(checkpoint_root / "reae.safetensors"), device="cpu"),
            strict=True,
        )
        transformer: Any = WanTransformer3DModel.from_pretrained(
            checkpoint_root,
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=dtype,
        )
        prompt = load_file(
            str(checkpoint_root / "prompt_embedding.safetensors"), device="cpu"
        )["prompt_emb"][0]
        resolved_device = torch.device(device)
        autoencoder.to(device=resolved_device, dtype=dtype).eval()
        transformer.to(device=resolved_device, dtype=dtype).eval()
        prompt = prompt.to(device=resolved_device, dtype=dtype)
        if resolved_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        prepare_transformer(
            transformer,
            window=attention_window,
            compile_blocks=compile_blocks,
        )
        return cls(
            autoencoder,
            transformer,
            prompt,
            device=resolved_device,
            dtype=dtype,
        )

    def start_stream(
        self, *, output_height: int, output_width: int, overlap: int
    ) -> SwiftVRStream:
        """Create isolated temporal state while reusing resident weights."""
        return SwiftVRStream(
            self,
            output_height=output_height,
            output_width=output_width,
            overlap=overlap,
        )


def _resolve_checkpoint(checkpoint: str, *, revision: str | None) -> Path:
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        return local
    maybe_download_hf_repo_on_rank0(
        checkpoint,
        revision=revision,
        allow_patterns=_CHECKPOINT_PATTERNS,
    )
    return Path(
        snapshot_download(
            checkpoint,
            revision=revision,
            allow_patterns=list(_CHECKPOINT_PATTERNS),
            local_files_only=True,
        )
    )


__all__ = ["RestorationAutoencoder", "SwiftVRPipeline", "SwiftVRStream"]
