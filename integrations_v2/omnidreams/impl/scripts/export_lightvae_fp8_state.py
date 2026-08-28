#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a calibrated LightVAE FP8 encoder state for OmniDreams native VAE."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omnidreams.impl.vae_native import DEFAULT_LIGHTVAE_FP8_STATE_PATH

from flashdreams.infra.config import derive_config

DEFAULT_CONFIG = "omnidreams"
DEFAULT_STATE_PATH = Path(DEFAULT_LIGHTVAE_FP8_STATE_PATH)
EXAMPLE_DATA_HF_REPO = "nvidia/omni-dreams-samples"
DEFAULT_EXAMPLE_DATA_UUID = "239560dc-33d1-11ef-9720-00044bcbccac"
VAE_FP8_VERSION_KEY = "__omnidreams_vae_fp8_version__"
MODEL_KIND_KEY = "__omnidreams_vae_fp8_model_kind__"
STATE_SCALE_MAX_KEY = "__omnidreams_vae_fp8_scale_max__"
MODEL_KIND_LIGHTVAE_ENCODER = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Output .pt path (default: {DEFAULT_STATE_PATH}).",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help="OmniDreams config whose encoder checkpoint should be calibrated.",
    )
    parser.add_argument("--calibration-video", type=Path, default=None)
    parser.add_argument(
        "--example-data",
        action="store_true",
        help="Fetch the bundled single-view HDMap sample for calibration.",
    )
    parser.add_argument(
        "--example-data-uuid",
        default=DEFAULT_EXAMPLE_DATA_UUID,
        help="Single-view sample UUID used with --example-data.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
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


def _ensure_hf_calibration_video(
    uuid: str = DEFAULT_EXAMPLE_DATA_UUID,
) -> Path:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    from huggingface_hub.hf_api import RepoFile  # noqa: PLC0415

    subdir = f"data/single_view/{uuid}"
    entries = HfApi().list_repo_tree(
        repo_id=EXAMPLE_DATA_HF_REPO,
        repo_type="dataset",
        path_in_repo=subdir,
        recursive=False,
    )
    candidates = [
        entry.path
        for entry in entries
        if isinstance(entry, RepoFile) and entry.path.endswith("_hdmap.mp4")
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one *_hdmap.mp4 under {subdir!r} in "
            f"{EXAMPLE_DATA_HF_REPO!r}, found {candidates}"
        )
    return Path(
        hf_hub_download(
            repo_id=EXAMPLE_DATA_HF_REPO,
            repo_type="dataset",
            filename=candidates[0],
        )
    )


def _resolve_video(args: argparse.Namespace) -> Path:
    if args.calibration_video is not None:
        return args.calibration_video.expanduser().resolve()
    if args.example_data:
        return _ensure_hf_calibration_video(args.example_data_uuid)
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


def export_lightvae_fp8_state(
    out: Path,
    *,
    calibration_video: Path,
    config_name: str = DEFAULT_CONFIG,
    device: str | torch.device = "cuda",
    height: int = 704,
    width: int = 1280,
    frames: int = 13,
    scale_max: float = 24.0,
) -> Path:
    from omnidreams.config import OMNIDREAMS_CONFIGS  # noqa: PLC0415

    target = out.expanduser().resolve()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )

    config = OMNIDREAMS_CONFIGS[config_name]
    if config.encoder is None:
        raise TypeError(f"{config_name} does not define a VAE encoder")
    encoder_cfg = derive_config(
        config.encoder,
        dtype=torch.float16,
        use_compile=False,
        use_cuda_graph=False,
        native_vae_acceleration="disabled",
        native_vae_fp8_auto_export=False,
    )
    encoder = encoder_cfg.setup().to(torch_device).eval()
    model: Any = encoder.vae
    video_bcthw = _load_video_prefix_bcthw(
        calibration_video,
        frames=frames,
        height=height,
        width=width,
        device=torch_device,
    )
    with torch.inference_mode():
        amax = _collect_activation_amax(model, video_bcthw)
        state = _build_fp8_state(
            model.state_dict(),
            _activation_scales(amax, scale_max=scale_max),
            scale_max=scale_max,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

    activation_scale_count = sum(key.endswith(".activation_scale") for key in state)
    print(f"Wrote {target}")
    print(f"Calibration video: {calibration_video}")
    print(f"Activation scales: {activation_scale_count}")
    return target


def ensure_lightvae_fp8_state(
    out: Path = DEFAULT_STATE_PATH,
) -> Path:
    target = out.expanduser().resolve()
    if target.is_file():
        return target

    print(f"{target} is missing; exporting cached LightVAE FP8 state...")
    return export_lightvae_fp8_state(
        target,
        calibration_video=_ensure_hf_calibration_video(),
    )


def main() -> None:
    args = _parse_args()
    export_lightvae_fp8_state(
        args.out,
        calibration_video=_resolve_video(args),
        config_name=args.config_name,
        device=args.device,
        height=args.height,
        width=args.width,
        frames=args.frames,
        scale_max=args.scale_max,
    )


if __name__ == "__main__":
    main()
