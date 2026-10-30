# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefetch the onnxruntime-web (WebGPU) runtime and serve it same-origin.

Token streaming decodes the latents in the browser with onnxruntime-web's WebGPU
backend. Rather than have the page fetch that runtime from a public CDN, a rank-0
preparation step downloads the pinned dist into the FlashDreams cache (verifying
each file against a pinned sha256), and the server advertises + serves it under
the token-stream asset route. Serving it same-origin keeps the page free of a
third-party runtime dependency and sidesteps the COEP/CORP cross-origin friction
WebGPU + SharedArrayBuffer impose.

onnxruntime-web is MIT-licensed (https://github.com/microsoft/onnxruntime); the
pinned version, upstream URL, and redistribution belong in THIRD-PARTY-NOTICES
(tracked with the ``onnx`` dependency declaration in the dependency workstream).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from flashdreams.core.io.disk import default_flashdreams_cache_dir
from flashdreams.core.io.download import download_to_cache
from flashdreams.runtime_v2.serving.token_stream.config import ASSET_URL_PREFIX

#: Pinned onnxruntime-web release. The client's decoder is validated against this
#: exact runtime, so a bump is a deliberate, re-tested change.
ORT_VERSION = "1.20.1"

_ORT_DIST_URL = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist"

#: The WebGPU dist file set and its pinned sha256. ``ort.webgpu.min.js`` is the
#: ``<script>`` entry point; it loads the JSEP WebAssembly (the ``.wasm`` binary
#: and its ``.mjs`` glue) from ``wasm_base`` at decode time.
_ORT_FILES: dict[str, str] = {
    "ort.webgpu.min.js": (
        "8443389bf14b34582ea40c590b5007bb652d6ac07dd76829a5bd82d2372e6895"
    ),
    "ort-wasm-simd-threaded.jsep.wasm": (
        "185b0861a6cd6cbdfb057289338090436483cc59e10a7bc83bd167b15531a51b"
    ),
    "ort-wasm-simd-threaded.jsep.mjs": (
        "2fb14505d39926e4c544de8c57fe59a2f2bc74caba6f71890be9afecc5046758"
    ),
}

_ORT_LOADER = "ort.webgpu.min.js"

# NOTE: the minified loader also references ``ort-wasm-proxy-worker``, fetched
# only when ``ort.env.wasm.proxy`` is enabled (it is not, matching the source
# client). If the browser client turns proxy mode on, pin that file here too.


def _served_path(filename: str) -> str:
    """Asset-route key (and URL tail) a dist file is published at."""
    return f"ort/{ORT_VERSION}/{filename}"


def _sha256_validator(expected: str) -> Callable[[Path], None]:
    """Return a :func:`download_to_cache` validator enforcing ``expected``."""

    def _validate(path: Path) -> None:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"onnxruntime-web {path.name} failed its integrity check: "
                f"sha256 {actual} != pinned {expected}"
            )

    return _validate


def ensure_ort_runtime() -> tuple[dict[str, Any], dict[str, str]]:
    """Prefetch the pinned onnxruntime-web dist and describe it for the client.

    Downloads each dist file into the cache once (verified against its pinned
    sha256; a cached file is reused and never re-fetched), and returns:

    * a descriptor for the token-stream session header -- ``version``, the
      ``wasm_base`` the client points ``ort.env.wasm.wasmPaths`` at, and the
      ``loader`` script URL -- so the browser loads the runtime same-origin; and
    * the ``{served_path: local_file}`` asset routes the server serves it from.

    Runs in the preparation phase on rank zero. Raises ``RuntimeError`` (from
    :func:`download_to_cache`) if a file cannot be fetched or fails its integrity
    check, aborting the launch rather than advertising a runtime that is absent
    or corrupt.
    """
    cache = default_flashdreams_cache_dir() / "token-stream-ort" / ORT_VERSION
    routes: dict[str, str] = {}
    for filename, expected_sha in _ORT_FILES.items():
        local = download_to_cache(
            f"{_ORT_DIST_URL}/{filename}",
            cache_dir=cache,
            filename=filename,
            validator=_sha256_validator(expected_sha),
            timeout=120.0,
        )
        routes[_served_path(filename)] = str(local)

    descriptor = {
        "version": ORT_VERSION,
        "wasm_base": f"{ASSET_URL_PREFIX}/ort/{ORT_VERSION}/",
        "loader": f"{ASSET_URL_PREFIX}/{_served_path(_ORT_LOADER)}",
    }
    return descriptor, routes
