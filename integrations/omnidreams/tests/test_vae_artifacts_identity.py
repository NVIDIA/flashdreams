# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the token-stream VAE-decoder identity contract (Phase 0).

The identity is what the server's handshake compares and what the client fetch
URL is versioned by, so these cover stability, per-field sensitivity, the hash,
and the supported-family gate (TAEHV accepted, every other decoder refused).
"""

from __future__ import annotations

import json

import pytest
from omnidreams.webrtc import vae_artifacts

pytestmark = pytest.mark.ci_cpu


def _identity(
    decoder_class: str = "taehv",
    checkpoint_id: str = "lighttaew2_1",
    len_t: int = 2,
    video_w: int = 1280,
    video_h: int = 704,
):
    return vae_artifacts.decoder_identity(
        decoder_class, checkpoint_id, len_t, video_w, video_h
    )


def test_identity_is_stable_for_equal_inputs():
    assert _identity() == _identity()
    assert vae_artifacts.identity_hash(_identity()) == vae_artifacts.identity_hash(
        _identity()
    )


@pytest.mark.parametrize(
    "changed",
    [
        _identity(decoder_class="wan"),
        _identity(checkpoint_id="someother"),
        _identity(len_t=3),
        _identity(video_w=1920),
        _identity(video_h=1080),
    ],
)
def test_identity_and_hash_change_when_any_field_changes(changed):
    base = _identity()
    assert changed != base
    assert vae_artifacts.identity_hash(changed) != vae_artifacts.identity_hash(base)


def test_identity_hash_is_short_lowercase_hex():
    digest = vae_artifacts.identity_hash(_identity())
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("https://huggingface.co/x/resolve/main/lighttaew2_1.pth", "lighttaew2_1"),
        ("/local/checkpoints/foo.pt", "foo"),
        ("bare-name", "bare-name"),
    ],
)
def test_checkpoint_id_extracts_stem(path, expected):
    assert vae_artifacts._checkpoint_id(path) == expected


# --- family gate (imports the real recipe config classes) ---


def test_taehv_config_is_supported():
    from flashdreams.recipes.taehv import TeahvVAEDecoderConfig

    cfg = TeahvVAEDecoderConfig()  # defaults to the lighttae checkpoint
    assert vae_artifacts.decoder_family(cfg) == "taehv"
    # identity_for_config must agree with the pure identity for the same inputs.
    identity = vae_artifacts.identity_for_config(
        cfg, len_t=2, video_w=1280, video_h=704
    )
    assert identity == vae_artifacts.decoder_identity(
        "taehv", "lighttaew2_1", 2, 1280, 704
    )


def test_wan_decoder_is_refused():
    from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoderConfig

    with pytest.raises(vae_artifacts.TokenStreamUnsupportedDecoder):
        vae_artifacts.decoder_family(WanVAEDecoderConfig())


def test_non_config_object_is_refused():
    class _NotADecoder:
        pass

    with pytest.raises(vae_artifacts.TokenStreamUnsupportedDecoder):
        vae_artifacts.identity_for_config(
            _NotADecoder(), len_t=2, video_w=1280, video_h=704
        )


# --- session-header descriptor (filesystem; no GPU) ---


def _write_artifact(cache_root, precision, version, identity):
    out = cache_root / "omnidreams-vae-decoders"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"taehv_decoder.{precision}.onnx").write_bytes(b"onnx-bytes")
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


def test_build_descriptor_versions_urls_and_forwards_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    identity = _identity()
    version = vae_artifacts.identity_hash(identity)
    _write_artifact(tmp_path, "fp32", version, identity)

    descriptor = vae_artifacts.build_descriptor()
    assert descriptor is not None
    assert descriptor["version"] == version
    assert descriptor["identity"] == identity
    assert descriptor["cold_start"] == "zeros"
    # The version segment in the URL is what busts a stale browser cache.
    assert (
        descriptor["precisions"]["fp32"]
        == f"{vae_artifacts.URL_PREFIX}/{version}/fp32.onnx"
    )


def test_build_descriptor_is_none_without_an_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    assert vae_artifacts.build_descriptor() is None


def test_artifact_identity_reads_stamped_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    identity = _identity()
    _write_artifact(tmp_path, "fp32", vae_artifacts.identity_hash(identity), identity)
    assert vae_artifacts.artifact_identity("fp32") == identity


def test_artifact_identity_is_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    assert vae_artifacts.artifact_identity("fp32") is None
