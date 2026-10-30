# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate exported TAEHV VAE-decoder artifacts and describe them to the client.

The pre-export step (:mod:`omnidreams.token_stream.vae_export`) writes
``taehv_decoder.<precision>.onnx`` plus a ``.spec.json`` sidecar into a cache
directory. The token-stream WebRTC server
(:mod:`flashdreams.runtime_v2.serving.webrtc_server`) serves those files over an
HTTP route and advertises them in the token-stream session header so the browser
can fetch and initialize the decoder. This module is the single source of truth
for that layout, shared by the export CLI, the serving route, and the session
header.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.runtime_v2.serving.token_stream.config import ASSET_URL_PREFIX

#: Precisions the export CLI can produce and the client can consume. ``fp32`` is
#: the portable default; ``fp16`` requires the WebGPU ``shader-f16`` feature and
#: is a client-side opt-in.
SUPPORTED_PRECISIONS = ("fp32", "fp16")

_DEFAULT_PRECISION = "fp32"
_MODEL_STEM = "taehv_decoder"


def _asset_served_path(version: str, precision: str) -> str:
    """Served path (under ``ASSET_URL_PREFIX``) for a precision's ONNX.

    The single source of the per-precision path, so :func:`build_descriptor`'s
    advertised URLs and :func:`build_asset_routes`' served keys cannot drift. The
    ``version`` segment makes a re-exported artifact a new URL, so the browser
    can never serve a stale decoder from cache.
    """
    return f"vae-model/{version}/{precision}.onnx"


def cache_dir() -> Path:
    """Directory holding the exported decoder artifacts."""
    return default_flashdreams_cache_dir() / "omnidreams-vae-decoders"


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


class TokenStreamUnsupportedDecoder(RuntimeError):
    """A launched decoder has no token-stream (cache-IO) export.

    Token streaming decodes on the *client* from an exported ONNX, which exists
    only for the TAEHV cache-IO decoder. Any other decoder family (e.g. the full
    Wan VAE) cannot serve token streaming, so callers refuse loudly rather than
    advertising a decoder the browser cannot run.
    """


def _checkpoint_id(checkpoint_path: str) -> str:
    """Filename stem of a checkpoint path or URL (e.g. ``lighttaew2_1``).

    The checkpoint pins the decoder's weights and architecture, so its stem
    identifies them without hashing the weights.
    """
    stem = Path(urlparse(str(checkpoint_path)).path or str(checkpoint_path)).stem
    return stem or str(checkpoint_path)


def decoder_identity(
    decoder_class: str, checkpoint_id: str, len_t: int, video_w: int, video_h: int
) -> dict[str, Any]:
    """The fields that must match between the launched decoder and the exported ONNX.

    A token client decodes correctly only if its decoder is the same family and
    weights the server generates with (``decoder_class`` + ``checkpoint_id``),
    sized for the same latent frames per chunk (``len_t``) and output resolution
    (``video_dims``). Latent channels / spatial ratio are *consequences* of the
    checkpoint, so they live in the spec for the client but are deliberately not
    in the identity -- which keeps this a pure, instantiation-free function.
    """
    return {
        "decoder_class": decoder_class,
        "checkpoint_id": checkpoint_id,
        "len_t": int(len_t),
        "video_dims": [int(video_w), int(video_h)],
    }


def identity_hash(identity: dict[str, Any]) -> str:
    """Short, stable hash of a decoder identity.

    Used as the artifact ``version`` and as a path segment in the client fetch
    URL, so a re-export under a changed identity yields a new URL the browser
    cannot satisfy from a stale cache.
    """
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def decoder_family(decoder_cfg: Any) -> str:
    """Token-stream family tag for a pipeline's decoder config; raises if unsupported."""
    # Lazy import so this layout module stays free of the recipes/torch import at
    # load time; the config class is only needed when a caller is actually
    # resolving a token-stream decoder.
    from flashdreams.recipes.taehv import TaehvVAEDecoderConfig

    if isinstance(decoder_cfg, TaehvVAEDecoderConfig):
        return "taehv"
    raise TokenStreamUnsupportedDecoder(
        f"token streaming needs a TAEHV decoder; launched config uses "
        f"{type(decoder_cfg).__name__}"
    )


def identity_for_config(
    decoder_cfg: Any, len_t: int, video_w: int, video_h: int
) -> dict[str, Any]:
    """Decoder identity for a launched pipeline's decoder config.

    Combines the supported-family check with the pure identity, so the server
    handshake and the exporter derive the identity the same way. Raises
    :class:`TokenStreamUnsupportedDecoder` for non-TAEHV decoders.
    """
    family = decoder_family(decoder_cfg)
    checkpoint_id = _checkpoint_id(getattr(decoder_cfg, "checkpoint_path", ""))
    return decoder_identity(family, checkpoint_id, len_t, video_w, video_h)


def artifact_identity(precision: str) -> dict[str, Any] | None:
    """The identity stamped in a precision's spec, or ``None`` if the artifact is
    absent, unreadable, or predates the identity stamp.
    """
    path = _spec_path(precision)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("identity")
    except (json.JSONDecodeError, OSError):
        return None


def build_descriptor(precisions: Sequence[str] | None = None) -> dict[str, Any] | None:
    """Describe the exported decoder for the token-stream session header.

    Returns ``None`` when no artifact has been exported, in which case the token
    stream carries no decoder descriptor and the client cannot decode. The
    cache/shape spec is identical across precisions (only the tensor dtype
    differs), so one sidecar supplies the shared fields.

    Only ``precisions`` are advertised (default: every available one). The caller
    passes the subset whose stamped identity it verified against the launched
    model, so a stale artifact left at a different identity is never advertised.
    """
    precisions = list(available_precisions() if precisions is None else precisions)
    if not precisions:
        return None
    spec = json.loads(_spec_path(precisions[0]).read_text())
    default = _DEFAULT_PRECISION if _DEFAULT_PRECISION in precisions else precisions[0]
    version = spec["version"]
    descriptor = {
        "kind": "taehv-cache-io-onnx",
        "version": version,
        "default_precision": default,
        "precisions": {
            p: f"{ASSET_URL_PREFIX}/{_asset_served_path(version, p)}"
            for p in precisions
        },
        "latent_shape": spec["latent_shape"],
        "output_shape": spec["output_shape"],
        "input_names": spec["input_names"],
        "output_names": spec["output_names"],
        "cache": spec["cache"],
    }
    # Forwarded when present in the spec; omitted for older artifacts so the
    # client sees each field only when it is actually there (it defaults
    # cold_start to "zeros" for pre-field artifacts).
    if "identity" in spec:
        descriptor["identity"] = spec["identity"]
    if "cold_start" in spec:
        descriptor["cold_start"] = spec["cold_start"]
    return descriptor


def build_asset_routes(precisions: Sequence[str] | None = None) -> dict[str, str]:
    """Map each precision's served path to its local ONNX file.

    The application publishes this under ``SessionDesc.metadata`` so the server's
    asset route serves each ONNX file. The keys match the URLs
    :func:`build_descriptor` advertises (both derive from
    :func:`_asset_served_path`), so an advertised URL always resolves to a served
    file. ``precisions`` defaults to every available one; pass the same verified
    subset handed to :func:`build_descriptor` so only advertised files are served.
    Empty when nothing has been exported.
    """
    precisions = list(available_precisions() if precisions is None else precisions)
    if not precisions:
        return {}
    version = json.loads(_spec_path(precisions[0]).read_text())["version"]
    return {_asset_served_path(version, p): str(onnx_path(p)) for p in precisions}
