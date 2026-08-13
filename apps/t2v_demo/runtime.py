# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility bridge from the legacy T2V demo registry to the shared shell."""

from __future__ import annotations

from t2v.t2v import (
    FIELD_FPS,
    FIELD_PIXEL_HEIGHT,
    FIELD_PIXEL_WIDTH,
    FIELD_PROMPT,
    FIELD_TOTAL_BLOCKS,
    T2VDemoAdapter,
    T2VInputProvider,
    T2VModelConfig,
    T2VRuntime,
    T2VScenario,
    T2VSession,
)

from .backends import T2VBackend, resolve_backend


def model_from_backend(
    backend: str | T2VBackend,
    preset_id: str | None = None,
) -> T2VModelConfig:
    """Adapt the existing backend registry to one neutral T2V model config."""
    resolved = resolve_backend(backend) if isinstance(backend, str) else backend
    preset = resolved.resolve_runner(preset_id)
    return T2VModelConfig(
        model_id="flashdreams-t2v",
        preset_id=preset.name,
        pipeline=preset.pipeline,
        prompt=preset.prompt,
        total_blocks=preset.total_blocks,
        pixel_height=preset.pixel_height,
        pixel_width=preset.pixel_width,
        fps=preset.fps,
        runtime_options={"backend": resolved.key},
    )


def make_adapter(backend: str, preset_id: str | None = None) -> T2VDemoAdapter:
    """Build an adapter from a CLI/UI backend key."""
    return T2VDemoAdapter(model=model_from_backend(backend, preset_id))


__all__ = [
    "FIELD_FPS",
    "FIELD_PIXEL_HEIGHT",
    "FIELD_PIXEL_WIDTH",
    "FIELD_PROMPT",
    "FIELD_TOTAL_BLOCKS",
    "T2VDemoAdapter",
    "T2VInputProvider",
    "T2VModelConfig",
    "T2VRuntime",
    "T2VScenario",
    "T2VSession",
    "make_adapter",
    "model_from_backend",
]
