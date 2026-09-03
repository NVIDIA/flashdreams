# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only Action2V application for runtime and input development."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from flashdreams.runtime_v2.session_desc import SessionDesc

from .application import Action2VApplication, Action2VApplicationDefaults
from .input import ActionSnapshot

_DUMMY_FRAME_PATH = Path(__file__).with_name("assets") / "dummy_frame.ppm"
"""Packaged first frame used by the dummy rollout."""


@dataclass
class _DummyDiffusionModelConfig:
    """Minimal seed config exercised by the shared application."""

    seed: int | None = 0
    """RNG seed accepted for Action2V pipeline compatibility."""


@dataclass
class _DummyPipelineConfig:
    """Construct a model-free Action2V pipeline."""

    diffusion_model: _DummyDiffusionModelConfig = field(
        default_factory=_DummyDiffusionModelConfig
    )
    """Seed settings accepted for Action2V pipeline compatibility."""

    def setup(self) -> "_DummyPipeline":
        """Build the model-free pipeline."""
        return _DummyPipeline(self)


class _DummyDiffusionModel:
    dtype = torch.float32

    def __init__(self, seed: int) -> None:
        self.rng = torch.Generator().manual_seed(seed)


class _DummyPipeline:
    """Render solid-color chunks through the standard pipeline API."""

    device = torch.device("cpu")

    def __init__(self, config: _DummyPipelineConfig) -> None:
        seed = config.diffusion_model.seed
        if seed is None:
            raise ValueError("Dummy Action2V requires a deterministic seed.")
        self.diffusion_model = _DummyDiffusionModel(seed)

    def to(self, device: str) -> "_DummyPipeline":
        """Validate the dummy pipeline device."""
        if torch.device(device).type != "cpu":
            raise ValueError("Dummy Action2V only supports the CPU device.")
        return self

    def eval(self) -> "_DummyPipeline":
        """Return the stateless evaluation pipeline."""
        return self

    def initialize_cache(self, *, seed_pixels: Tensor) -> dict[str, Any]:
        """Initialize output dimensions from the seed frames."""
        return {
            "autoregressive_index": 0,
            "height": seed_pixels.shape[-2],
            "width": seed_pixels.shape[-1],
        }

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: dict[str, Any],
        input: Any,
    ) -> Tensor:
        """Render one solid-color action chunk."""
        if not isinstance(input, ActionSnapshot):
            raise TypeError("Dummy Action2V actions must be ActionSnapshot values.")
        cache["autoregressive_index"] = autoregressive_index
        frames = torch.empty(
            (1, 4, 3, cache["height"], cache["width"]),
            dtype=torch.float32,
        )
        forward = 0.8 if "W" in input.keys else -0.8
        button = 0.8 if input.mouse_buttons else -0.8
        pointer = max(-1.0, min(1.0, input.mouse_dx * 8.0 + input.wheel_y * 0.2))
        frames[:, :, 0].fill_(forward)
        frames[:, :, 1].fill_(button)
        frames[:, :, 2].fill_(pointer)
        frames.add_(min(autoregressive_index, 10) * 0.01).clamp_(-1.0, 1.0)
        return frames

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: dict[str, Any],
    ) -> dict[str, int]:
        """Return the generated step as a dummy metric."""
        assert cache["autoregressive_index"] == autoregressive_index
        return {"dummy_step": autoregressive_index}


def _resolve_first_frame(values: Mapping[str, Any]) -> Path:
    image_path = values.get("image_path")
    if image_path is not None:
        return Path(image_path)
    if values.get("example_data"):
        return _DUMMY_FRAME_PATH
    raise ValueError("Action2V requires --image-path or --example-data.")


def _seed_loader(path: Path, session_desc: SessionDesc) -> Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"dummy seed does not exist: {path}")
    return torch.zeros(
        (4, 3, session_desc.video_height, session_desc.video_width),
        dtype=torch.float32,
    )


def _action_mapper(session_desc: SessionDesc, sensitivity: float):
    del session_desc

    def map_snapshot(snapshot: ActionSnapshot) -> ActionSnapshot:
        return ActionSnapshot(
            keys=snapshot.keys,
            mouse_buttons=snapshot.mouse_buttons,
            mouse_dx=snapshot.mouse_dx * sensitivity,
            mouse_dy=snapshot.mouse_dy * sensitivity,
            wheel_x=snapshot.wheel_x,
            wheel_y=snapshot.wheel_y,
        )

    return map_snapshot


DUMMY_ACTION2V_DEFAULTS = Action2VApplicationDefaults(
    slug="action2v-dummy",
    pipeline_config=_DummyPipelineConfig(),
    input_resolver=_resolve_first_frame,
    seed_loader=_seed_loader,
    action_mapper_factory=_action_mapper,
    total_blocks=10_000,
    pixel_width=320,
    pixel_height=180,
    fps=60,
    device="cpu",
    metadata={"model": "action2v-dummy", "frames_per_action": 4},
)
"""Defaults for the CPU-only action-to-video demonstration."""


def create_app() -> Action2VApplication:
    """Return the CPU-only Action2V demonstration application."""
    return Action2VApplication(defaults=DUMMY_ACTION2V_DEFAULTS)


__all__ = ["DUMMY_ACTION2V_DEFAULTS", "create_app"]
