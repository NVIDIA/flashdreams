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

"""Real-ESRGAN checkpoint loading and tensor upsampling."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flashdreams.core.io.download import download_to_cache
from realesrgan.archs import RRDBNet, SRVGGNetCompact

RealESRGANModelName = Literal[
    "RealESRGAN_x4plus",
    "RealESRNet_x4plus",
    "RealESRGAN_x4plus_anime_6B",
    "RealESRGAN_x2plus",
    "realesr-animevideov3",
    "realesr-general-x4v3",
]

MODEL_URLS: dict[str, str] = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "RealESRNet_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "realesr-animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    "realesr-general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
}
"""Public Real-ESRGAN checkpoint URLs. Files are cached at runtime."""

MODEL_CONFIGS: dict[str, dict[str, int | str]] = {
    "RealESRGAN_x4plus": {
        "arch": "RRDBNet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 4,
    },
    "RealESRNet_x4plus": {
        "arch": "RRDBNet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 4,
    },
    "RealESRGAN_x4plus_anime_6B": {
        "arch": "RRDBNet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 6,
        "num_grow_ch": 32,
        "scale": 4,
    },
    "RealESRGAN_x2plus": {
        "arch": "RRDBNet",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_block": 23,
        "num_grow_ch": 32,
        "scale": 2,
    },
    "realesr-animevideov3": {
        "arch": "SRVGGNetCompact",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_conv": 16,
        "upscale": 4,
        "act_type": "prelu",
        "scale": 4,
    },
    "realesr-general-x4v3": {
        "arch": "SRVGGNetCompact",
        "num_in_ch": 3,
        "num_out_ch": 3,
        "num_feat": 64,
        "num_conv": 32,
        "upscale": 4,
        "act_type": "prelu",
        "scale": 4,
    },
}
"""Model architecture table for public Real-ESRGAN checkpoints."""

CACHE_DIR = (
    Path(os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")))
    / "realesrgan"
)
"""User-writable cache directory for Real-ESRGAN checkpoint downloads."""


def default_model_name(scale: Literal[2, 4]) -> RealESRGANModelName:
    """Return the default public Real-ESRGAN model for ``scale``."""
    return "RealESRGAN_x2plus" if scale == 2 else "RealESRGAN_x4plus"


def create_model(model_name: str) -> tuple[nn.Module, int]:
    """Create the architecture for a named Real-ESRGAN checkpoint."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown Real-ESRGAN model {model_name!r}; "
            f"available models: {sorted(MODEL_CONFIGS)}"
        )
    config = MODEL_CONFIGS[model_name]
    arch = str(config["arch"])
    scale = int(config["scale"])
    if arch == "RRDBNet":
        return (
            RRDBNet(
                num_in_ch=int(config["num_in_ch"]),
                num_out_ch=int(config["num_out_ch"]),
                num_feat=int(config["num_feat"]),
                num_block=int(config["num_block"]),
                num_grow_ch=int(config["num_grow_ch"]),
                scale=scale,
            ),
            scale,
        )
    if arch == "SRVGGNetCompact":
        return (
            SRVGGNetCompact(
                num_in_ch=int(config["num_in_ch"]),
                num_out_ch=int(config["num_out_ch"]),
                num_feat=int(config["num_feat"]),
                num_conv=int(config["num_conv"]),
                upscale=int(config["upscale"]),
                act_type=str(config["act_type"]),
            ),
            scale,
        )
    raise ValueError(f"Unknown Real-ESRGAN architecture {arch!r}.")


def resolve_model_path(model_name: str, model_path: str | Path | None = None) -> Path:
    """Return a local checkpoint path, downloading public weights if needed."""
    if model_path is not None:
        return Path(model_path).expanduser()
    if model_name not in MODEL_URLS:
        raise ValueError(f"No public checkpoint URL for model {model_name!r}.")
    return download_to_cache(MODEL_URLS[model_name], cache_dir=CACHE_DIR)


def load_weights(model: nn.Module, checkpoint_path: str | Path) -> None:
    """Load Real-ESRGAN weights from ``checkpoint_path`` into ``model``."""
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser(),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state = checkpoint["params"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


class RealESRGANUpsampler:
    """Real-ESRGAN RGB tensor upsampler.

    Args:
        scale: Output upsample factor.
        model_name: Public Real-ESRGAN model name. Defaults to the general
            x2 or x4 RRDBNet model matching ``scale``.
        model_path: Optional local checkpoint path.
        tile: Optional input tile size. ``0`` runs each frame as one tensor.
        tile_pad: Input overlap on every tile edge.
        pre_pad: Reflection padding applied before model inference.
        half: Use fp16 on CUDA.
        compile_model: Compile the model with ``torch.compile`` after loading.
        compile_mode: Mode passed to ``torch.compile``.
        device: Torch device string or object.
    """

    def __init__(
        self,
        *,
        scale: Literal[2, 4] = 2,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = True,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        device: str | torch.device | None = None,
        load_checkpoint: bool = True,
    ) -> None:
        self.scale = scale
        self.model_name = model_name or default_model_name(scale)
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.model_scale = create_model(self.model_name)
        if self.model_scale != scale:
            raise ValueError(
                f"Model {self.model_name!r} has native scale {self.model_scale}; "
                f"requested scale {scale}."
            )
        if load_checkpoint:
            load_weights(self.model, resolve_model_path(self.model_name, model_path))
        self.model.eval().to(self.device)
        if self.half and self.device.type == "cuda":
            self.model.half()
        if self.compile_model:
            self.model = torch.compile(self.model, mode=self.compile_mode)
        self.dtype = (
            torch.float16 if self.half and self.device.type == "cuda" else torch.float32
        )

    @torch.no_grad()
    def upsample_video_tensor(self, video: torch.Tensor) -> torch.Tensor:
        """Upsample ``video`` shaped ``[T, C, H, W]`` in ``[-1, 1]``."""
        if video.ndim != 4 or video.shape[1] != 3:
            raise ValueError(
                f"Expected [T, 3, H, W] video tensor; got {tuple(video.shape)}"
            )
        frames = [self.upsample_frame_tensor(frame) for frame in video]
        return torch.stack(frames, dim=0)

    @torch.no_grad()
    def upsample_frame_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        """Upsample one RGB frame shaped ``[3, H, W]`` in ``[-1, 1]``."""
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError(
                f"Expected [3, H, W] frame tensor; got {tuple(frame.shape)}"
            )
        image = ((frame.to(self.device, dtype=self.dtype) + 1.0) * 0.5).clamp(0, 1)
        output = self._process_image(image.unsqueeze(0))
        return (output.squeeze(0).float().cpu().clamp(0, 1) * 2.0) - 1.0

    @torch.no_grad()
    def upsample_bgr_image(self, image: np.ndarray) -> tuple[np.ndarray, str]:
        """Upsample an OpenCV image array and return ``(image, mode)``."""
        mode = "RGB"
        alpha: np.ndarray | None = None
        if image.ndim == 2:
            mode = "L"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            mode = "RGBA"
            alpha = image[:, :, 3]
            image_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)

        max_range = 65535.0 if image.dtype == np.uint16 else 255.0
        tensor = torch.from_numpy(
            (image_rgb.astype(np.float32) / max_range).transpose(2, 0, 1)
        )
        output = (self.upsample_frame_tensor(tensor * 2.0 - 1.0) + 1.0) * 0.5
        if mode == "L":
            return _rgb_tensor_to_gray_image(output, image.dtype), mode
        output_bgr = _rgb_tensor_to_bgr_image(output, image.dtype)
        if mode == "RGBA" and alpha is not None:
            alpha_out = cv2.resize(
                alpha,
                (output_bgr.shape[1], output_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            output_bgr = _bgr_to_bgra(output_bgr, alpha_out)

        return output_bgr, mode

    def _process_image(self, image: torch.Tensor) -> torch.Tensor:
        padded, crop = self._pad_input(image)
        output = (
            self._process_tiled(padded)
            if self.tile > 0
            else self.model(padded.to(dtype=self.dtype))
        )
        return self._crop_output(output, crop)

    def _pad_input(self, image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.pre_pad:
            image = F.pad(image, (0, self.pre_pad, 0, self.pre_pad), "reflect")
        mod_scale = 2 if self.model_scale == 2 else None
        mod_pad_h = 0
        mod_pad_w = 0
        if mod_scale is not None:
            _, _, height, width = image.shape
            mod_pad_h = (mod_scale - height % mod_scale) % mod_scale
            mod_pad_w = (mod_scale - width % mod_scale) % mod_scale
            if mod_pad_h or mod_pad_w:
                image = F.pad(image, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return image, (
            (self.pre_pad + mod_pad_h) * self.model_scale,
            (self.pre_pad + mod_pad_w) * self.model_scale,
        )

    def _crop_output(self, output: torch.Tensor, crop: tuple[int, int]) -> torch.Tensor:
        crop_h, crop_w = crop
        if crop_h:
            output = output[:, :, :-crop_h, :]
        if crop_w:
            output = output[:, :, :, :-crop_w]
        return output

    def _process_tiled(self, image: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = image.shape
        output = image.new_zeros(
            batch,
            channels,
            height * self.model_scale,
            width * self.model_scale,
        )
        tiles_x = math.ceil(width / self.tile)
        tiles_y = math.ceil(height / self.tile)
        for y in range(tiles_y):
            for x in range(tiles_x):
                in_x0 = x * self.tile
                in_y0 = y * self.tile
                in_x1 = min(in_x0 + self.tile, width)
                in_y1 = min(in_y0 + self.tile, height)
                pad_x0 = max(in_x0 - self.tile_pad, 0)
                pad_y0 = max(in_y0 - self.tile_pad, 0)
                pad_x1 = min(in_x1 + self.tile_pad, width)
                pad_y1 = min(in_y1 + self.tile_pad, height)
                tile = image[:, :, pad_y0:pad_y1, pad_x0:pad_x1]
                output_tile = self.model(tile.to(dtype=self.dtype))

                out_x0 = in_x0 * self.model_scale
                out_y0 = in_y0 * self.model_scale
                out_x1 = in_x1 * self.model_scale
                out_y1 = in_y1 * self.model_scale
                tile_x0 = (in_x0 - pad_x0) * self.model_scale
                tile_y0 = (in_y0 - pad_y0) * self.model_scale
                tile_x1 = tile_x0 + (in_x1 - in_x0) * self.model_scale
                tile_y1 = tile_y0 + (in_y1 - in_y0) * self.model_scale
                output[:, :, out_y0:out_y1, out_x0:out_x1] = output_tile[
                    :, :, tile_y0:tile_y1, tile_x0:tile_x1
                ]
        return output


def write_bgr_image(path: str | Path, image: np.ndarray) -> None:
    """Write ``image`` to ``path`` and raise if OpenCV rejects it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image to {path}")


def _rgb_tensor_to_bgr_image(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    """Convert a CPU ``[3, H, W]`` RGB float tensor in ``[0, 1]`` to BGR image."""
    image_dtype = _output_image_dtype(dtype)
    scale = _dtype_max(image_dtype)
    torch_dtype = _torch_dtype(image_dtype)
    return (
        tensor.clamp(0, 1)
        .mul(scale)
        .round()
        .to(torch_dtype)[[2, 1, 0]]
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )


def _rgb_tensor_to_gray_image(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
    """Convert a CPU ``[3, H, W]`` RGB float tensor in ``[0, 1]`` to grayscale."""
    image_dtype = _output_image_dtype(dtype)
    scale = _dtype_max(image_dtype)
    torch_dtype = _torch_dtype(image_dtype)
    tensor = tensor.clamp(0, 1)
    gray = tensor[0] * 0.299 + tensor[1] * 0.587 + tensor[2] * 0.114
    return gray.mul(scale).round().to(torch_dtype).contiguous().numpy()


def _bgr_to_bgra(bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Append ``alpha`` to a BGR image without routing through float color convert."""
    output = np.empty((*bgr.shape[:2], 4), dtype=bgr.dtype)
    output[:, :, :3] = bgr
    output[:, :, 3] = alpha.astype(bgr.dtype, copy=False)
    return output


def _output_image_dtype(dtype: np.dtype) -> np.dtype:
    return (
        np.dtype(np.uint16)
        if np.dtype(dtype) == np.dtype(np.uint16)
        else np.dtype(np.uint8)
    )


def _dtype_max(dtype: np.dtype) -> float:
    return 65535.0 if np.dtype(dtype) == np.dtype(np.uint16) else 255.0


def _torch_dtype(dtype: np.dtype) -> torch.dtype:
    return torch.uint16 if np.dtype(dtype) == np.dtype(np.uint16) else torch.uint8
