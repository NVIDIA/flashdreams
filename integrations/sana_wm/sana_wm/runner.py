# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashDreams runner for the native SANA-WM pipeline."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flashdreams.infra.runner import RunnerConfig
from loguru import logger
from PIL import Image

from sana_wm.pipeline import (
    DEFAULT_CAMERA_PATH,
    DEFAULT_IMAGE_PATH,
    DEFAULT_INTRINSICS_PATH,
    DEFAULT_PROMPT_PATH,
    SanaWMGenerationParams,
    SanaWMNativePipeline,
    SanaWMNativePipelineConfig,
    SamplingAlgo,
    get_reference_module,
)


@dataclass(kw_only=True)
class SanaWMRunnerConfig(RunnerConfig):
    """CLI config for SANA-WM bidirectional image-to-video inference."""

    _target: type = field(default_factory=lambda: SanaWMRunner)

    pipeline: SanaWMNativePipelineConfig = field(
        default_factory=SanaWMNativePipelineConfig
    )

    image: Path = DEFAULT_IMAGE_PATH
    """First-frame RGB image."""

    prompt: str | Path = DEFAULT_PROMPT_PATH
    """Inline prompt text or a UTF-8 prompt file."""

    camera: Path | None = DEFAULT_CAMERA_PATH
    """Optional (F, 4, 4) camera-to-world trajectory. Mutually exclusive with
    action."""

    action: str | None = None
    """Optional WASD/IJKL action DSL, e.g. 'w-80,jw-40,w-40'."""

    translation_speed: float = 0.05
    rotation_speed_deg: float = 1.2

    intrinsics: Path | None = DEFAULT_INTRINSICS_PATH
    """Optional intrinsics .npy. If omitted, SANA-WM estimates intrinsics with
    Pi3X."""

    num_frames: int = 161
    fps: int = 16
    step: int = 60
    cfg_scale: float = 5.0
    flow_shift: float | None = None
    seed: int = 42
    negative_prompt: str = ""
    sampling_algo: SamplingAlgo = "flow_euler_ltx"

    no_action_overlay: bool = False

    name: str = "sana_wm"
    save_metadata: bool = True


class SanaWMRunner:
    """Drive the native SANA-WM pipeline from FlashDreams CLI config.

    SANA-WM's reference implementation is a monolithic image-to-video pipeline,
    not a FlashDreams ``StreamInferencePipeline``. This runner still uses the
    shared ``RunnerConfig`` registry shape, but owns construction order so the
    native path stays byte-for-byte comparable with standalone inference.
    """

    config: SanaWMRunnerConfig
    pipeline: SanaWMNativePipeline

    def __init__(self, config: SanaWMRunnerConfig) -> None:
        self.config = config
        if _is_torchrun_env():
            self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
        self.is_rank_zero = self.global_rank == 0

    def run(self) -> None:
        if self.world_size != 1:
            if not self.is_rank_zero:
                logger.info(
                    "SANA-WM runner is single-process; non-zero torchrun rank exits."
                )
                return
            logger.warning(
                "SANA-WM runner is single-process; only torchrun rank 0 will generate."
            )

        cfg = self.config
        refs = get_reference_module()
        device = _resolve_device(cfg.device, self.local_rank)
        image_path = _resolve_existing_path(cfg.image, "--image")
        image = Image.open(image_path).convert("RGB")
        prompt = _resolve_prompt(cfg.prompt)

        c2w_full = _resolve_trajectory(refs, cfg)
        num_frames = min(cfg.num_frames, c2w_full.shape[0])
        snapped = refs._snap_num_frames(
            num_frames, stride=8, upper_bound=c2w_full.shape[0]
        )
        if snapped != cfg.num_frames:
            logger.warning(
                "LTX-2 VAE requires num_frames = 8k+1; "
                f"--num-frames={cfg.num_frames} snapped to {snapped} "
                f"(trajectory has {c2w_full.shape[0]} frames)."
            )
        num_frames = snapped
        c2w = c2w_full[:num_frames]

        cropped, src_size, resized_size, crop_offset = refs.resize_and_center_crop(
            image
        )
        if cfg.intrinsics is not None:
            intrinsics_path = _resolve_existing_path(cfg.intrinsics, "--intrinsics")
            intr_src = refs.load_intrinsics(intrinsics_path, num_frames)
        else:
            intr_one = refs.estimate_intrinsics_with_pi3x(
                image, torch.device(device), refs.get_root_logger()
            )
            intr_src = np.broadcast_to(intr_one, (num_frames, 4)).copy()
        intrinsics_vec4 = refs.transform_intrinsics_for_crop(
            intr_src, src_size, resized_size, crop_offset
        )

        self.pipeline = cfg.pipeline.setup(device=device, logger=refs.get_root_logger())
        params = SanaWMGenerationParams(
            num_frames=num_frames,
            fps=cfg.fps,
            step=cfg.step,
            cfg_scale=cfg.cfg_scale,
            flow_shift=cfg.flow_shift,
            seed=cfg.seed,
            negative_prompt=cfg.negative_prompt,
            sampling_algo=cfg.sampling_algo,
        )

        out = self.pipeline.generate(cropped, prompt, c2w, intrinsics_vec4, params)
        video_hwc = out["video"]
        if not cfg.no_action_overlay:
            logger.info("Compositing action overlay onto the output video.")
            video_hwc = refs.apply_overlay(video_hwc, out["c2w"])

        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        video_path = refs.write_video(
            cfg.output_dir, cfg.name, video_hwc, params.fps, refs.get_root_logger()
        )
        if cfg.save_metadata:
            metadata_path = cfg.output_dir / f"{cfg.name}_metadata.json"
            metadata = {
                "runner_name": cfg.runner_name,
                "image": str(image_path),
                "camera": None if cfg.camera is None else str(cfg.camera),
                "action": cfg.action,
                "intrinsics": None if cfg.intrinsics is None else str(cfg.intrinsics),
                "pipeline": _pipeline_metadata(cfg.pipeline),
                "num_frames": num_frames,
                "fps": cfg.fps,
                "step": cfg.step,
                "cfg_scale": cfg.cfg_scale,
                "flow_shift": cfg.flow_shift,
                "seed": cfg.seed,
                "sampling_algo": cfg.sampling_algo,
                "refiner": cfg.pipeline.enable_refiner,
                "no_action_overlay": cfg.no_action_overlay,
                "video_path": str(video_path),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            logger.info(f"[{cfg.runner_name}] wrote metadata -> {metadata_path}")


def _is_torchrun_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _resolve_device(device: str, local_rank: int) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA is unavailable; falling back to CPU.")
            return "cpu"
        if "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}"
    return device


def _resolve_existing_path(value: Path, flag_name: str) -> Path:
    path = value.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{flag_name} file does not exist: {path}")
    return path


def _resolve_prompt(value: str | Path) -> str:
    if isinstance(value, Path):
        text = _resolve_existing_path(value, "--prompt").read_text(
            encoding="utf-8", errors="replace"
        )
    else:
        maybe_path = Path(value).expanduser()
        if maybe_path.is_file():
            text = maybe_path.read_text(encoding="utf-8", errors="replace")
        else:
            text = value

    text = text.strip()
    if not text:
        raise ValueError("--prompt must be inline text or a non-empty prompt file")
    return text


def _resolve_trajectory(refs: Any, cfg: SanaWMRunnerConfig) -> np.ndarray:
    if cfg.action is not None:
        return refs.action_string_to_c2w(
            cfg.action,
            translation_speed=cfg.translation_speed,
            rotation_speed_deg=cfg.rotation_speed_deg,
        )
    if cfg.camera is None:
        raise ValueError("One of --camera or --action is required.")
    camera_path = _resolve_existing_path(cfg.camera, "--camera")
    c2w_raw = np.load(camera_path).astype(np.float32)
    if c2w_raw.ndim != 3 or c2w_raw.shape[1:] != (4, 4):
        raise ValueError(f"--camera must be a (F, 4, 4) .npy; got {c2w_raw.shape}.")
    return c2w_raw


def _pipeline_metadata(pipeline: SanaWMNativePipelineConfig) -> dict[str, object]:
    return {
        "recipe_name": pipeline.recipe_name,
        "config_path": str(pipeline.config_path),
        "model_path": str(pipeline.model_path),
        "enable_refiner": pipeline.enable_refiner,
        "refiner_root": str(pipeline.refiner_root),
        "refiner_gemma_root": str(pipeline.refiner_gemma_root),
        "refiner_seed": pipeline.refiner_seed,
        "sink_size": pipeline.sink_size,
        "offload_vae": pipeline.offload_vae,
        "offload_refiner": pipeline.offload_refiner,
    }


__all__ = ["SanaWMRunner", "SanaWMRunnerConfig"]
