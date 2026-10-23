# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared torch.compile mode normalization for the interactive demo."""

from __future__ import annotations

COMPILE_DENOISE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)
COMPILE_DENOISE_SCOPES = ("step", "loop")

COMPILE_SWEEP_MODES = ("off", "on", *COMPILE_DENOISE_MODES)

_COMPILE_MODE_ALIASES = {
    "on": "max-autotune-no-cudagraphs",
    "max_autotune": "max-autotune",
    "max_autotune_no_cudagraphs": "max-autotune-no-cudagraphs",
}

_COMPILE_MODE_LABELS = {
    "default": "default",
    "reduce-overhead": "reduce_overhead",
    "max-autotune": "max_autotune",
    "max-autotune-no-cudagraphs": "max_autotune_no_cg",
}

_COMPILE_SCOPE_ALIASES = {
    "velocity": "step",
    "model": "step",
    "sampler": "loop",
    "denoise-loop": "loop",
    "denoise_loop": "loop",
}

_COMPILE_SCOPE_LABELS = {
    "step": "step",
    "loop": "loop",
}


def normalize_compile_denoise_mode(value: str | None) -> str:
    """Return the canonical active torch.compile mode."""
    raw = (value or "max-autotune-no-cudagraphs").strip().lower()
    mode = _COMPILE_MODE_ALIASES.get(raw, raw)
    if mode not in COMPILE_DENOISE_MODES:
        allowed = ", ".join(COMPILE_DENOISE_MODES)
        raise ValueError(f"compile denoise mode must be one of: {allowed}")
    return mode


def normalize_compile_sweep_mode(value: str | None) -> str:
    """Return ``off`` or a canonical active compile mode for sweep configs."""
    raw = (value or "off").strip().lower()
    if raw == "off":
        return "off"
    return normalize_compile_denoise_mode(raw)


def normalize_compile_denoise_scope(value: str | None) -> str:
    """Return the canonical scope wrapped by ``torch.compile``."""
    raw = (value or "step").strip().lower()
    scope = _COMPILE_SCOPE_ALIASES.get(raw, raw)
    if scope not in COMPILE_DENOISE_SCOPES:
        allowed = ", ".join(COMPILE_DENOISE_SCOPES)
        raise ValueError(f"compile denoise scope must be one of: {allowed}")
    return scope


def compile_mode_label(mode: str) -> str:
    """Short label fragment for output directory names."""
    return _COMPILE_MODE_LABELS[normalize_compile_denoise_mode(mode)]


def compile_scope_label(scope: str) -> str:
    """Short label fragment for output directory names."""
    return _COMPILE_SCOPE_LABELS[normalize_compile_denoise_scope(scope)]


def torch_compile_mode_arg(mode: str) -> str | None:
    """Map our explicit mode names to the ``torch.compile`` API."""
    mode = normalize_compile_denoise_mode(mode)
    return None if mode == "default" else mode
