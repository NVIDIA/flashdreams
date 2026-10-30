# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the video token stream."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.runtime_v2.serving.token_stream.codec import (
    RawFloat16TokenCodecConfig,
    TokenCodecConfig,
)

#: Route prefix the server serves per-session token-stream assets from (the
#: client-side VAE-decoder ONNX, and later the onnxruntime-web dist). An
#: application puts a ``{served_path: local_file}`` map under the
#: ``token_stream_assets`` key of ``SessionDesc.metadata``; the server serves
#: each file at ``<ASSET_URL_PREFIX>/<served_path>`` and nowhere else. Integration
#: code (e.g. omnidreams) imports this so the advertised URLs and the served
#: paths cannot drift.
ASSET_URL_PREFIX = "/api/token-stream/assets"

#: Key under ``SessionDesc.metadata`` holding the ``{served_path: local_file}``
#: map the asset route serves from.
ASSET_METADATA_KEY = "token_stream_assets"

#: Key under ``SessionDesc.metadata`` holding the extra session-header fields
#: (e.g. the ``vae_model`` descriptor) the emitter advertises to the client.
EXTRA_HEADER_METADATA_KEY = "token_stream_extra_header"

#: Content types the asset route sets explicitly by extension. A ``.wasm`` served
#: as octet-stream breaks ``WebAssembly.instantiateStreaming`` and a ``.mjs``
#: without a JavaScript type breaks module loading, and the stdlib ``mimetypes``
#: table that ``FileResponse`` consults does not reliably carry either; anything
#: not listed here keeps ``FileResponse``'s own guess.
ASSET_CONTENT_TYPES = {".wasm": "application/wasm", ".mjs": "text/javascript"}


@dataclass
class TokenStreamConfig:
    """Settings for streaming latent tokens alongside the rendered video."""

    enabled: bool = False
    """Whether the token stream is emitted for a session."""

    codec: TokenCodecConfig = field(default_factory=RawFloat16TokenCodecConfig)
    """Codec used to serialize each latent frame."""

    flow_window_size: int = 4
    """Maximum number of chunks in flight before waiting on client acks."""
