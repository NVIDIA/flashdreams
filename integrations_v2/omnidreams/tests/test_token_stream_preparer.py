# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test for the token-stream preparer hook.

``prepare_token_stream`` is what the interactive-drive adapter wires in as the
application's ``token_stream_preparer``. The GPU export is stubbed here; these
cover the glue: it advertises the exported decoder only when its stamped
identity matches the launched model, and the asset map it returns points the
server at the real ONNX file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
from omnidreams.token_stream import vae_artifacts, vae_export

pytestmark = pytest.mark.ci_cpu

_PRECISION = "fp32"

_ORT_DESCRIPTOR = {
    "version": "1.20.1",
    "wasm_base": "/api/token-stream/assets/ort/1.20.1/",
    "loader": "/api/token-stream/assets/ort/1.20.1/ort.webgpu.min.js",
}
_ORT_ROUTES = {"ort/1.20.1/ort.webgpu.min.js": "/cache/ort/ort.webgpu.min.js"}


def _patch_ort(monkeypatch):
    """Stub the onnxruntime-web prefetch (no network) in the preparer's module."""
    monkeypatch.setattr(
        vae_export, "ensure_ort_runtime", lambda: (_ORT_DESCRIPTOR, _ORT_ROUTES)
    )


def _write_artifact(
    cache_root: Path, identity: dict, precision: str = _PRECISION
) -> str:
    """Write a fake artifact for ``precision`` stamped with ``identity``."""
    out = cache_root / "omnidreams-vae-decoders"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"taehv_decoder.{precision}.onnx").write_bytes(b"onnx-bytes")
    version = vae_artifacts.identity_hash(identity)
    spec = {
        "version": version,
        "latent_shape": [2, 16, 88, 160],
        "output_shape": [8, 3, 704, 1280],
        "input_names": ["latent"],
        "output_names": ["rgb"],
        "cache": [],
        "identity": identity,
        "cold_start": "zeros",
    }
    (out / f"taehv_decoder.{precision}.spec.json").write_text(json.dumps(spec))
    return version


def test_prepare_token_stream_returns_matching_descriptor_and_assets(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    # Stand in for the GPU export; the artifact is supplied by the fixture.
    monkeypatch.setattr(vae_export, "ensure_vae_artifacts", lambda *a, **k: None)
    _patch_ort(monkeypatch)

    launched = vae_artifacts.identity_for_config(
        OMNIDREAMS_PIPELINE_CONFIG.decoder, 2, 1280, 704
    )
    version = _write_artifact(tmp_path, launched)

    metadata = vae_export.prepare_token_stream(
        OMNIDREAMS_PIPELINE_CONFIG, video_width=1280, video_height=704, device="cpu"
    )

    extra = metadata["token_stream_extra_header"]
    descriptor = extra["vae_model"]
    assert descriptor["identity"] == launched
    served = f"vae-model/{version}/{_PRECISION}.onnx"
    assert metadata["token_stream_assets"][served] == str(
        vae_artifacts.onnx_path(_PRECISION)
    )
    # The served key is exactly the tail of the advertised fetch URL.
    assert descriptor["precisions"][_PRECISION].endswith(served)
    # The onnxruntime-web runtime is advertised and its assets merged in too.
    assert extra["ort_runtime"] == _ORT_DESCRIPTOR
    assert (
        metadata["token_stream_assets"]["ort/1.20.1/ort.webgpu.min.js"]
        == "/cache/ort/ort.webgpu.min.js"
    )


def test_prepare_token_stream_aborts_when_artifact_identity_mismatches(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(vae_export, "ensure_vae_artifacts", lambda *a, **k: None)
    _patch_ort(monkeypatch)
    # Artifact stamped for a different output size than the launched model.
    stale = vae_artifacts.decoder_identity("taehv", "lighttaew2_1", 2, 640, 384)
    _write_artifact(tmp_path, stale)

    with pytest.raises(vae_export.TokenStreamExportError):
        vae_export.prepare_token_stream(
            OMNIDREAMS_PIPELINE_CONFIG,
            video_width=1280,
            video_height=704,
            device="cpu",
        )


def test_prepare_token_stream_excludes_a_stale_other_precision(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(vae_export, "ensure_vae_artifacts", lambda *a, **k: None)
    _patch_ort(monkeypatch)

    launched = vae_artifacts.identity_for_config(
        OMNIDREAMS_PIPELINE_CONFIG.decoder, 2, 1280, 704
    )
    _write_artifact(tmp_path, launched, "fp32")  # matches the launched model
    # A leftover fp16 at a DIFFERENT identity, as a prior v1 export could leave.
    stale = vae_artifacts.decoder_identity("taehv", "lighttaew2_1", 2, 640, 384)
    _write_artifact(tmp_path, stale, "fp16")

    metadata = vae_export.prepare_token_stream(
        OMNIDREAMS_PIPELINE_CONFIG,
        video_width=1280,
        video_height=704,
        device="cpu",
        precisions=("fp32", "fp16"),  # request both; only fp32 matches
    )

    descriptor = metadata["token_stream_extra_header"]["vae_model"]
    assert set(descriptor["precisions"]) == {"fp32"}  # the stale fp16 is not advertised
    assert all("fp16" not in served for served in metadata["token_stream_assets"])
