# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native FlashDreams pipeline wrapper for the vendored SANA-WM implementation."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from flashdreams.infra.config import InstantiateConfig
from PIL import Image
from torch import nn

SamplingAlgo = Literal["flow_euler_ltx", "flow_euler", "flow_dpm-solver"]

HF_REPO = "Efficient-Large-Model/SANA-WM_bidirectional"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "sana_wm_1600m_720p.yaml"
DEFAULT_IMAGE_PATH = PACKAGE_DIR / "assets" / "sana_wm" / "demo_0.png"
DEFAULT_PROMPT_PATH = PACKAGE_DIR / "assets" / "sana_wm" / "demo_0.txt"
DEFAULT_CAMERA_PATH = PACKAGE_DIR / "assets" / "sana_wm" / "demo_0_pose.npy"
DEFAULT_INTRINSICS_PATH = PACKAGE_DIR / "assets" / "sana_wm" / "demo_0_intrinsics.npy"
DEFAULT_MODEL_PATH = f"hf://{HF_REPO}/dit/sana_wm_1600m_720p.safetensors"
DEFAULT_REFINER_ROOT = f"hf://{HF_REPO}/refiner"
DEFAULT_REFINER_GEMMA_ROOT = f"hf://{HF_REPO}/refiner/text_encoder"


@dataclass(kw_only=True)
class SanaWMGenerationParams:
    """Per-call generation knobs matching the standalone SANA-WM CLI."""

    num_frames: int = 161
    fps: int = 16
    step: int = 60
    cfg_scale: float = 5.0
    flow_shift: float | None = None
    seed: int = 42
    negative_prompt: str = ""
    sampling_algo: SamplingAlgo = "flow_euler_ltx"


@dataclass(kw_only=True)
class SanaWMNativePipelineConfig(InstantiateConfig):
    """Config for the native SANA-WM pipeline.

    The model architecture, schedulers, VAE, camera conditioning, and refiner
    live under ``sana_wm._reference`` so the integration does not depend on an
    external NVlabs/Sana checkout at runtime.
    """

    _target: type["SanaWMNativePipeline"] = field(
        default_factory=lambda: SanaWMNativePipeline
    )

    recipe_name: str = "sana-wm-bidirectional"
    config_path: str | Path = DEFAULT_CONFIG_PATH
    model_path: str | Path = DEFAULT_MODEL_PATH

    enable_refiner: bool = True
    refiner_root: str | Path = DEFAULT_REFINER_ROOT
    refiner_gemma_root: str | Path = DEFAULT_REFINER_GEMMA_ROOT
    refiner_seed: int = 42
    sink_size: int = 1

    offload_vae: bool = False
    offload_refiner: bool = False


class SanaWMNativePipeline(nn.Module):
    """FlashDreams-owned entrypoint around the vendored SANA-WM pipeline."""

    config: SanaWMNativePipelineConfig

    def __init__(
        self,
        config: SanaWMNativePipelineConfig,
        *,
        device: str | torch.device = "cuda",
        logger: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.refs = _load_reference_module()
        self.logger = logger or self.refs.get_root_logger()

        inference_cfg = self.refs.pyrallis.parse(
            config_class=self.refs.InferenceConfig,
            config_path=_resolve_reference_path(self.refs, config.config_path),
            args=[],
        )
        _ensure_reference_model_registered(inference_cfg.model.model)
        refiner = (
            self.refs.RefinerSettings(
                root=config.refiner_root,
                gemma_root=config.refiner_gemma_root,
                sink_size=config.sink_size,
                seed=config.refiner_seed,
            )
            if config.enable_refiner
            else None
        )
        self.impl = self.refs.SanaWMPipeline(
            config=inference_cfg,
            model_path=_resolve_reference_path(self.refs, config.model_path),
            device=self.device,
            refiner=refiner,
            offload_vae=config.offload_vae,
            offload_refiner=config.offload_refiner,
            logger=self.logger,
        )

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        c2w: np.ndarray,
        intrinsics_vec4: np.ndarray,
        params: SanaWMGenerationParams,
    ) -> dict[str, object]:
        """Generate one SANA-WM rollout using the standalone reference math."""
        ref_params = self.refs.GenerationParams(
            num_frames=params.num_frames,
            fps=params.fps,
            step=params.step,
            cfg_scale=params.cfg_scale,
            flow_shift=params.flow_shift,
            seed=params.seed,
            negative_prompt=params.negative_prompt,
            sampling_algo=params.sampling_algo,
        )
        return self.impl.generate(image, prompt, c2w, intrinsics_vec4, ref_params)

    def to(self, *args: object, **kwargs: object) -> "SanaWMNativePipeline":
        """Keep ``nn.Module.to`` shape while construction owns device placement."""
        requested = kwargs.get("device", args[0] if args else None)
        if not isinstance(requested, (str, int, torch.device)):
            return self
        if torch.device(requested) != self.device:
            raise RuntimeError(
                "SanaWMNativePipeline must be constructed on its target device; "
                f"got to({requested!r}) after construction on {self.device}."
            )
        return self


@lru_cache(maxsize=1)
def _load_reference_module() -> Any:
    os.environ.setdefault("DISABLE_XFORMERS", "1")
    return importlib.import_module("sana_wm._reference.inference_sana_wm")


def _ensure_reference_model_registered(model_name: str) -> None:
    builder = importlib.import_module("sana_wm._reference.diffusion.model.builder")
    if builder.MODELS.get(model_name) is not None:
        return

    try:
        importlib.import_module(
            "sana_wm._reference.diffusion.model.nets.sana_multi_scale_video_camctrl"
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import SANA-WM model {model_name!r}. The native "
            "integration keeps standalone SANA-WM's CUDA model path, so the "
            "runtime environment must provide its required CUDA extras "
            "(for example Triton/flash-attn as in the standalone setup)."
        ) from exc

    if builder.MODELS.get(model_name) is None:
        raise RuntimeError(
            f"SANA-WM model {model_name!r} was imported but not registered."
        )


def _resolve_reference_path(refs: Any, value: str | Path) -> str:
    if isinstance(value, Path):
        return str(value.expanduser())
    return refs.resolve_hf_path(value)


def get_reference_module() -> Any:
    """Return the vendored SANA-WM reference module.

    This is intentionally lazy: CPU-only tests can import ``sana_wm.pipeline``
    without importing the heavy SANA-WM model stack.
    """
    return _load_reference_module()


__all__ = [
    "DEFAULT_CAMERA_PATH",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_IMAGE_PATH",
    "DEFAULT_INTRINSICS_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_PROMPT_PATH",
    "DEFAULT_REFINER_GEMMA_ROOT",
    "DEFAULT_REFINER_ROOT",
    "SanaWMGenerationParams",
    "SanaWMNativePipeline",
    "SanaWMNativePipelineConfig",
    "SamplingAlgo",
    "get_reference_module",
]
