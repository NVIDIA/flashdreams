# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate an exported latent-caption model and describe it to the client.

Parallel to :mod:`omnidreams.webrtc.vae_artifacts`, for the live‑captioning path:
a small latent-native classifier (single forward, consuming the same latents the
VAE decoder does — no pixels) exported to ONNX, plus a ``.spec.json`` carrying
its label schema / caption bank. When no artifact has been exported the
descriptor is ``None``, and the client falls back to its stub classifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Precisions the export can produce and the client can consume.
SUPPORTED_PRECISIONS = ("fp32", "fp16")

#: Route base the server serves the ONNX from and the client fetches it at.
URL_PREFIX = "/api/token-stream/caption-model"

_DEFAULT_PRECISION = "fp32"
_MODEL_STEM = "caption_model"


def cache_dir() -> Path:
    """Directory holding the exported caption-model artifacts."""
    root = Path(
        os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams"))
    )
    return root / "omnidreams-caption-models"


def onnx_path(precision: str) -> Path:
    """Path to the exported caption ONNX for ``precision``."""
    return cache_dir() / f"{_MODEL_STEM}.{precision}.onnx"


def _spec_path(precision: str) -> Path:
    return cache_dir() / f"{_MODEL_STEM}.{precision}.spec.json"


def available_precisions() -> list[str]:
    """Precisions with both an ONNX and its spec present, in preference order."""
    return [
        p
        for p in SUPPORTED_PRECISIONS
        if onnx_path(p).exists() and _spec_path(p).exists()
    ]


def build_descriptor() -> dict[str, Any] | None:
    """Describe the exported caption model for the token-stream session header.

    Returns ``None`` when no artifact has been exported (the client then uses its
    stub classifier). The shape/schema spec is identical across precisions, so
    one sidecar supplies the shared fields.
    """
    precisions = available_precisions()
    if not precisions:
        return None
    spec = json.loads(_spec_path(precisions[0]).read_text())
    default = _DEFAULT_PRECISION if _DEFAULT_PRECISION in precisions else precisions[0]
    return {
        "kind": spec.get("kind", "latent-caption-onnx"),
        "version": spec["version"],
        "default_precision": default,
        "precisions": {p: f"{URL_PREFIX}/{p}.onnx" for p in precisions},
        # How many latent chunks the model expects as its temporal window.
        "input_window_chunks": spec.get("input_window_chunks"),
        "latent_shape": spec.get("latent_shape"),
        # Exactly one of these drives the client's caption rendering.
        "labels": spec.get("labels"),
        "caption_bank": spec.get("caption_bank"),
    }
