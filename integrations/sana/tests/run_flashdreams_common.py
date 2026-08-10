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

"""Shared helpers for FlashDreams SANA-WM test runners."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from sana_wm.camera import (
    action_string_to_c2w,
    default_intrinsics_vec4,
    fit_camera_trajectory,
    load_intrinsics,
    resize_center_crop_geometry,
    snap_num_frames,
    transform_intrinsics_for_crop,
)
from sana_wm.constants import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    SANA_WM_VAE_TEMPORAL_COMPRESSION,
)


def apply_backend_defaults() -> None:
    """Apply the stack-matched CUDA defaults used by the test harness."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        import torch._inductor.config as inductor_config

        inductor_config.coordinate_descent_tuning = True
        inductor_config.epilogue_fusion = True
    except Exception:
        pass


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto``/``cuda`` to the local rank's CUDA device when present."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        return torch.device("cpu")
    if device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    return torch.device(device)


def prepare_bidirectional_inputs(
    *,
    image_path: Path,
    camera_path: Path,
    intrinsics_path: Path | None,
    intrinsics_hfov_deg: float,
    num_frames_requested: int,
) -> tuple[Image.Image, np.ndarray, np.ndarray, int]:
    """Prepare the legacy bidirectional input contract."""
    image = Image.open(image_path).convert("RGB")
    c2w_full = np.load(camera_path).astype(np.float32)
    if c2w_full.ndim != 3 or c2w_full.shape[1:] != (4, 4):
        raise ValueError(f"--camera-path must be [F,4,4]; got {c2w_full.shape}.")

    num_frames = min(num_frames_requested, c2w_full.shape[0])
    num_frames = snap_num_frames(
        num_frames,
        stride=SANA_WM_VAE_TEMPORAL_COMPRESSION,
        upper_bound=c2w_full.shape[0],
    )
    c2w = c2w_full[:num_frames]
    cropped, intrinsics_vec4 = _prepare_image_and_intrinsics(
        image,
        intrinsics_path=intrinsics_path,
        intrinsics_hfov_deg=intrinsics_hfov_deg,
        num_frames=num_frames,
    )
    return cropped, c2w, intrinsics_vec4, num_frames


def prepare_streaming_inputs(
    *,
    image_path: Path,
    camera_path: Path | None,
    camera_source: str,
    action: str | None,
    translation_speed: float,
    rotation_speed_deg: float,
    intrinsics_path: Path | None,
    intrinsics_hfov_deg: float,
    num_frames_requested: int,
    snap_stride: int,
) -> tuple[Image.Image, np.ndarray, np.ndarray, int]:
    """Prepare the streaming action/camera input contract."""
    image = Image.open(image_path).convert("RGB")
    if camera_source == "action":
        if not action:
            raise ValueError("--camera-source=action requires --action.")
        num_frames = snap_num_frames(num_frames_requested, stride=snap_stride)
        c2w = action_string_to_c2w(
            action,
            translation_speed=translation_speed,
            rotation_speed_deg=rotation_speed_deg,
            num_frames=num_frames,
        )
    else:
        if camera_path is None:
            raise ValueError("--camera-source=camera requires --camera-path.")
        c2w_full = np.load(camera_path).astype(np.float32)
        if c2w_full.ndim != 3 or c2w_full.shape[1:] != (4, 4):
            raise ValueError(f"--camera-path must be [F,4,4]; got {c2w_full.shape}.")
        num_frames = min(num_frames_requested, c2w_full.shape[0])
        num_frames = snap_num_frames(
            num_frames,
            stride=snap_stride,
            upper_bound=c2w_full.shape[0],
        )
        c2w = fit_camera_trajectory(c2w_full, num_frames)

    cropped, intrinsics_vec4 = _prepare_image_and_intrinsics(
        image,
        intrinsics_path=intrinsics_path,
        intrinsics_hfov_deg=intrinsics_hfov_deg,
        num_frames=num_frames,
    )
    return cropped, c2w, intrinsics_vec4, num_frames


def preload_pipeline_components(pipeline: Any) -> None:
    """Load model components before timed generation begins."""
    encoder = getattr(pipeline, "encoder", None)
    if encoder is not None:
        text_encoder = getattr(encoder, "text_encoder", None)
        if text_encoder is not None:
            text_encoder._ensure_text_encoder()
        first_frame_encoder = getattr(encoder, "first_frame_encoder", None)
        if first_frame_encoder is not None:
            first_frame_encoder._ensure_vae()
    pipeline.diffusion_model.transformer._ensure_model()
    decoder = getattr(pipeline, "decoder", None)
    vae_decoder = getattr(decoder, "vae_decoder", None)
    if vae_decoder is not None:
        vae_decoder._ensure_vae()
    refiner = getattr(decoder, "refiner", None)
    if refiner is not None:
        refiner._ensure_refiner()


def install_stage1_compile_hook(pipeline: Any) -> None:
    """Compile Stage 1 lazily after checkpoint load."""
    transformer = pipeline.diffusion_model.transformer
    original_ensure_model = transformer._ensure_model
    compiled = {"done": False}

    def _ensure_model_with_compile() -> None:
        original_ensure_model()
        if compiled["done"]:
            return
        transformer.model = torch.compile(
            transformer.model,
            mode="max-autotune-no-cudagraphs",
        )
        compiled["done"] = True

    transformer._ensure_model = _ensure_model_with_compile


def compile_streaming_refiner(pipeline: Any) -> None:
    """Match upstream streaming's default refiner ``torch.compile`` path."""
    decoder = getattr(pipeline, "decoder", None)
    refiner = getattr(decoder, "refiner", None)
    if refiner is None:
        return
    refiner._ensure_refiner()
    compiled = getattr(refiner.refiner, "_flashdreams_compiled", False)
    if compiled:
        return
    compile_mode = os.environ.get(
        "SANA_WM_TORCH_COMPILE_MODE",
        "max-autotune-no-cudagraphs",
    ).strip()
    compile_dynamic_raw = (
        os.environ.get(
            "SANA_WM_TORCH_COMPILE_DYNAMIC",
            "1",
        )
        .strip()
        .lower()
    )
    compile_dynamic = compile_dynamic_raw not in {"0", "false", "no", "off"}
    refiner.refiner.transformer = torch.compile(
        refiner.refiner.transformer,
        mode=compile_mode,
        dynamic=compile_dynamic,
    )
    refiner.refiner._flashdreams_compiled = True


def sum_stage_ms(rows: list[dict[str, float]], key: str) -> float | None:
    """Sum optional per-chunk CUDA stage timings."""
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return float(sum(values)) if values else None


def write_video(output_dir: Path, name: str, frames: np.ndarray, fps: int) -> Path:
    """Write HWC uint8 frames to an MP4 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_generated.mp4"
    iio.imwrite(path, frames, fps=fps)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a benchmark JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _prepare_image_and_intrinsics(
    image: Image.Image,
    *,
    intrinsics_path: Path | None,
    intrinsics_hfov_deg: float,
    num_frames: int,
) -> tuple[Image.Image, np.ndarray]:
    resized_size, crop_offset = resize_center_crop_geometry(
        image.size,
        target_h=DEFAULT_VIDEO_HEIGHT,
        target_w=DEFAULT_VIDEO_WIDTH,
    )
    resized = image.resize(resized_size, Image.Resampling.LANCZOS)
    left, top = crop_offset
    cropped = resized.crop(
        (left, top, left + DEFAULT_VIDEO_WIDTH, top + DEFAULT_VIDEO_HEIGHT)
    )
    if intrinsics_path is None:
        intrinsics_src = default_intrinsics_vec4(
            image.size,
            num_frames,
            hfov_deg=intrinsics_hfov_deg,
        )
    else:
        intrinsics_src = load_intrinsics(intrinsics_path, num_frames)
    intrinsics_vec4 = transform_intrinsics_for_crop(
        intrinsics_src,
        image.size,
        resized_size,
        crop_offset,
    )
    return cropped, intrinsics_vec4
