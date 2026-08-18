# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate exported TAEHV VAE-decoder artifacts and describe them to the client.

The pre-export step (:mod:`omnidreams.webrtc.export_vae`) writes
``taehv_decoder.<precision>.onnx`` plus a ``.spec.json`` sidecar into a cache
directory. The WebRTC server serves those files over an HTTP route and
advertises them in the token-stream session header so the browser can fetch and
initialize the decoder. This module is the single source of truth for that
layout, shared by the export CLI, the serving route, and the session header.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Precisions the export CLI can produce and the client can consume. ``fp32`` is
#: the portable default; ``fp16`` requires the WebGPU ``shader-f16`` feature and
#: is a client-side opt-in.
SUPPORTED_PRECISIONS = ("fp32", "fp16")

#: Route base the server serves the ONNX from and the client fetches it at.
URL_PREFIX = "/api/token-stream/vae-model"

_DEFAULT_PRECISION = "fp32"
_MODEL_STEM = "taehv_decoder"


def cache_dir() -> Path:
    """Directory holding the exported decoder artifacts."""
    root = Path(
        os.path.expanduser(os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams"))
    )
    return root / "omnidreams-vae-decoders"


def onnx_path(precision: str) -> Path:
    """Path to the exported ONNX for ``precision``."""
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
    """Describe the exported decoder for the token-stream session header.

    Returns ``None`` when no artifact has been exported, in which case the token
    stream carries no decoder descriptor and the client cannot decode. The
    cache/shape spec is identical across precisions (only the tensor dtype
    differs), so one sidecar supplies the shared fields.
    """
    precisions = available_precisions()
    if not precisions:
        return None
    spec = json.loads(_spec_path(precisions[0]).read_text())
    default = _DEFAULT_PRECISION if _DEFAULT_PRECISION in precisions else precisions[0]
    return {
        "kind": "taehv-cache-io-onnx",
        "version": spec["version"],
        "default_precision": default,
        "precisions": {p: f"{URL_PREFIX}/{p}.onnx" for p in precisions},
        "latent_shape": spec["latent_shape"],
        "output_shape": spec["output_shape"],
        "input_names": spec["input_names"],
        "output_names": spec["output_names"],
        "cache": spec["cache"],
    }
