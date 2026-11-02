# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the onnxruntime-web prefetch.

The real download is network-bound; these cover the integrity validator and the
descriptor/route construction (with the download stubbed), so a drift in the
served URLs or the pinned file set is caught without fetching ~22 MB.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from flashdreams.runtime_v2.serving.token_stream import ort_runtime

pytestmark = pytest.mark.ci_cpu


def test_sha256_validator_accepts_matching_and_rejects_mismatch(tmp_path: Path) -> None:
    content = b"onnxruntime-web bytes"
    target = tmp_path / "dist.bin"
    target.write_bytes(content)

    ort_runtime._sha256_validator(hashlib.sha256(content).hexdigest())(target)  # ok
    with pytest.raises(RuntimeError, match="integrity"):
        ort_runtime._sha256_validator("0" * 64)(target)


def test_ensure_ort_runtime_builds_descriptor_and_routes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))

    def _fake_download(url, *, cache_dir, filename, validator, timeout):
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / filename
        local.write_bytes(b"stub")  # stand in for the real dist file
        return local

    monkeypatch.setattr(ort_runtime, "download_to_cache", _fake_download)

    descriptor, routes = ort_runtime.ensure_ort_runtime()

    base = f"/api/token-stream/assets/ort/{ort_runtime.ORT_VERSION}/"
    assert descriptor["version"] == ort_runtime.ORT_VERSION
    assert descriptor["wasm_base"] == base
    assert descriptor["loader"] == base + "ort.webgpu.min.js"

    # Every pinned file is published at ort/<version>/<filename>, resolving to a
    # local file, and the loader URL's tail is one of those served keys.
    for filename in ort_runtime._ORT_FILES:
        served = f"ort/{ort_runtime.ORT_VERSION}/{filename}"
        assert served in routes
        assert routes[served].endswith(filename)
    assert descriptor["loader"].endswith(
        f"ort/{ort_runtime.ORT_VERSION}/ort.webgpu.min.js"
    )
