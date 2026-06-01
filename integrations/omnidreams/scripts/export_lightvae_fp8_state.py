#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a calibrated LightVAE FP8 encoder state for OmniDreams native VAE."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omnidreams.config import OMNIDREAMS_CONFIGS
from omnidreams.runner import (
    DEFAULT_EXAMPLE_DATA_UUID_1V,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
)

from flashdreams.infra.config import derive_config

DEFAULT_CONFIG = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae"
VAE_FP8_VERSION_KEY = "__omnidreams_vae_fp8_version__"
MODEL_KIND_KEY = "__omnidreams_vae_fp8_model_kind__"
STATE_SCALE_MAX_KEY = "__omnidreams_vae_fp8_scale_max__"
MODEL_KIND_LIGHTVAE_ENCODER = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path.")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help="OmniDreams config whose encoder checkpoint should be calibrated.",
    )
    parser.add_argument("--calibration-video", type=Path, default=None)
    parser.add_argument(
        "--example-data",
        action="store_true",
        help="Fetch the bundled single-view HDMap sample and use it for calibration.",
    )
    parser.add_argument(
        "--example-data-uuid",
        default=DEFAULT_EXAMPLE_DATA_UUID_1V,
        help="Single-view sample UUID used with --example-data.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=DEFAULT_VIDEO_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_VIDEO_WIDTH)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--scale-max", type=float, default=24.0)
    return parser.parse_args()


def _require_float8() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("PyTorch float8_e4m3fn is required for FP8 state export")
    return dtype


def _scale_view_shape(tensor: torch.Tensor, channel_dim: int) -> tuple[int, ...]:
    return tuple(
        tensor.shape[i] if i == channel_dim else 1 for i in range(tensor.dim())
    )


def _quantize_fp8_per_channel(
    tensor: torch.Tensor,
    *,
    channel_dim: int = 0,
    scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"expected floating tensor, got {tensor.dtype}")
    if tensor.dim() == 0:
        raise ValueError("per-channel quantization requires a non-scalar tensor")
    if scale_max <= 0:
        raise ValueError(f"scale_max must be positive, got {scale_max}")

    fp8_dtype = _require_float8()
    if channel_dim < 0:
        channel_dim += tensor.dim()
    reduce_dims = tuple(i for i in range(tensor.dim()) if i != channel_dim)
    tensor_fp32 = tensor.detach().float()
    amax = tensor_fp32.abs().amax(dim=reduce_dims) if reduce_dims else tensor_fp32.abs()
    scale = (amax / float(scale_max)).clamp(min=1.0e-6)
    scaled = tensor_fp32 / scale.reshape(_scale_view_shape(tensor, channel_dim))
    return scaled.to(fp8_dtype).contiguous().view(torch.uint8), scale.to(torch.float16)


def _channel_amax(value: torch.Tensor, channel_dim: int) -> torch.Tensor:
    reduce_dims = tuple(dim for dim in range(value.dim()) if dim != channel_dim)
    return value.detach().float().abs().amax(dim=reduce_dims).cpu()


def _collect_activation_amax(
    model: Any,
    video_bcthw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if video_bcthw.dim() != 5:
        raise ValueError(
            f"expected calibration video [B,C,T,H,W], got {tuple(video_bcthw.shape)}"
        )

    amax: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    cache_step_originals: list[tuple[torch.nn.Module, Any]] = []

    def record(name: str, value: torch.Tensor) -> None:
        if value.dim() not in (4, 5):
            return
        current = _channel_amax(value, 1)
        previous = amax.get(name)
        amax[name] = current if previous is None else torch.maximum(previous, current)

    def hook(name: str):
        def _hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            if isinstance(output, torch.Tensor):
                record(name, output)

        return _hook

    def pre_hook(name: str):
        def _pre_hook(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                record(name, inputs[0])

        return _pre_hook

    record("encoder.conv1.input", video_bcthw)
    record("encoder.input", video_bcthw)
    record("input", video_bcthw)
    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(hook(name)))
        if name == "encoder.middle.1.proj":
            handles.append(
                module.register_forward_pre_hook(pre_hook("encoder.middle.1.sdpa"))
            )
        cache_step = getattr(module, "cache_step", None)
        if callable(cache_step):
            cache_step_originals.append((module, cache_step))

            def wrapped_cache_step(
                *args: Any, _name: str = name, _orig: Any = cache_step, **kwargs: Any
            ) -> Any:
                output = _orig(*args, **kwargs)
                if isinstance(output, torch.Tensor):
                    record(_name, output)
                return output

            setattr(module, "cache_step", wrapped_cache_step)

    try:
        cache = model.prepare_cache()
        latent = model.encode(video_bcthw, cache=cache)
    finally:
        for handle in handles:
            handle.remove()
        for module, original in cache_step_originals:
            setattr(module, "cache_step", original)

    record("latent", latent)
    return amax


def _activation_scales(
    amax: Mapping[str, torch.Tensor],
    *,
    scale_max: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in amax.items():
        out[f"{name}.activation_scale"] = (
            (value.float().abs() / float(scale_max))
            .clamp(min=1.0e-6)
            .to(torch.float16)
            .contiguous()
        )
    latent = out.get("latent.activation_scale")
    if latent is not None:
        out["latent.activation_scale"] = latent
    return out


def _build_fp8_state(
    state_dict: Mapping[str, torch.Tensor],
    activation_scales: Mapping[str, torch.Tensor],
    *,
    scale_max: float,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        VAE_FP8_VERSION_KEY: torch.tensor([1], dtype=torch.int32),
        MODEL_KIND_KEY: torch.tensor([MODEL_KIND_LIGHTVAE_ENCODER], dtype=torch.int32),
        STATE_SCALE_MAX_KEY: torch.tensor([float(scale_max)], dtype=torch.float32),
    }

    for name, tensor in state_dict.items():
        if (
            name.endswith(".weight")
            and torch.is_floating_point(tensor)
            and tensor.dim() >= 2
        ):
            q, scale = _quantize_fp8_per_channel(
                tensor.detach(),
                channel_dim=0,
                scale_max=scale_max,
            )
            state[name] = q.cpu()
            state[name.replace(".weight", ".weight_scale")] = scale.cpu()
        elif torch.is_floating_point(tensor):
            state[name] = (
                tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
            )
        else:
            state[name] = tensor.detach().cpu().contiguous()

    for name, scale in activation_scales.items():
        if scale.dim() != 1:
            raise ValueError(
                f"{name} must be a 1D scale tensor, got {tuple(scale.shape)}"
            )
        state[name] = scale.detach().to(dtype=torch.float16, device="cpu").contiguous()
    return state


def _resolve_video(args: argparse.Namespace) -> Path:
    if args.calibration_video is not None:
        return args.calibration_video.expanduser().resolve()
    if args.example_data:
        hdmaps, _first_frames = _ensure_hf_single_view_example_data_synced(
            args.example_data_uuid
        )
        return hdmaps[0]
    raise SystemExit("--calibration-video or --example-data is required")


def _load_video_prefix_bcthw(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - import-time gate
        raise ImportError(
            "OpenCV is required to load calibration video frames"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    images: list[torch.Tensor] = []
    while len(images) < frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    cap.release()
    if len(images) < frames:
        raise RuntimeError(f"{path} has {len(images)} readable frames; need {frames}")
    video = (
        torch.stack(images, dim=1).unsqueeze(0).to(device=device, dtype=torch.float16)
    )
    return (video / 127.5 - 1.0).contiguous()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )

    config = OMNIDREAMS_CONFIGS[args.config_name]
    if config.encoder is None:
        raise TypeError(f"{args.config_name} does not define a VAE encoder")
    encoder_cfg = derive_config(
        config.encoder,
        dtype=torch.float16,
        use_compile=False,
        use_cuda_graph=False,
    )
    encoder = encoder_cfg.setup().to(device).eval()
    model: Any = encoder.vae

    video_path = _resolve_video(args)
    video_bcthw = _load_video_prefix_bcthw(
        video_path,
        frames=args.frames,
        height=args.height,
        width=args.width,
        device=device,
    )
    amax = _collect_activation_amax(model, video_bcthw)
    state = _build_fp8_state(
        model.state_dict(),
        _activation_scales(amax, scale_max=args.scale_max),
        scale_max=args.scale_max,
    )

    args.out.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f"Wrote {args.out}")
    print(f"Calibration video: {video_path}")
    print(
        f"Activation scales: {len([k for k in state if k.endswith('.activation_scale')])}"
    )


if __name__ == "__main__":
    main()
