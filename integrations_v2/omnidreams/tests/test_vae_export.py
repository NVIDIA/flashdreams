# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for config-driven VAE export.

The actual ONNX export needs a GPU (it instantiates the decoder weights); these
cover the parts that run before any weights load: config resolution, the
non-TAEHV refuse, the eager-config derivation, and the pure export/reuse plan.
"""

from __future__ import annotations

import dataclasses

import pytest
from omnidreams.config import OMNIDREAMS_PIPELINE_CONFIG
from omnidreams.token_stream import vae_artifacts, vae_export

pytestmark = pytest.mark.ci_cpu

_DEFAULT_NAME = OMNIDREAMS_PIPELINE_CONFIG.name


def _wan_pipeline_config():
    """A pipeline config whose decoder is the Wan VAE (no cache-IO export)."""
    from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoderConfig

    return dataclasses.replace(
        OMNIDREAMS_PIPELINE_CONFIG, decoder=WanVAEDecoderConfig()
    )


def test_resolve_default_config_uses_a_taehv_decoder():
    config = vae_export.resolve_pipeline_config(_DEFAULT_NAME)
    assert config.name == _DEFAULT_NAME
    assert vae_artifacts.decoder_family(config.decoder) == "taehv"
    # len_t is read from the config, not defaulted in the exporter.
    assert config.diffusion_model.transformer.len_t == 2


def test_resolve_unknown_config_lists_available():
    with pytest.raises(ValueError, match="available:"):
        vae_export.resolve_pipeline_config("nope-not-a-config")


def test_export_refuses_wan_decoder_before_loading_weights():
    # Refusal must come from the identity/family check, i.e. before any
    # GPU-bound instantiation -- hence this runs on CPU.
    with pytest.raises(vae_artifacts.TokenStreamUnsupportedDecoder):
        vae_export.export_decoder_from_config(
            _wan_pipeline_config(),
            video_width=1280,
            video_height=704,
            precision="fp32",
            device="cpu",
        )


def test_eager_replace_disables_compile_and_cuda_graph():
    eager = dataclasses.replace(
        OMNIDREAMS_PIPELINE_CONFIG.decoder, use_compile=False, use_cuda_graph=False
    )
    assert eager.use_compile is False
    assert eager.use_cuda_graph is False


# --- plan_vae_exports: the auto-export decision matrix (pure, no GPU) ---

_LAUNCHED = vae_artifacts.decoder_identity("taehv", "lighttaew2_1", 2, 1280, 704)
_STALE = vae_artifacts.decoder_identity("taehv", "lighttaew2_1", 2, 640, 384)


def test_plan_exports_all_on_empty_cache():
    plan = vae_export.plan_vae_exports(
        _LAUNCHED, ["fp32", "fp16"], {"fp32": None, "fp16": None}
    )
    assert plan == {"fp32": vae_export.EXPORT, "fp16": vae_export.EXPORT}


def test_plan_reuses_all_on_full_match():
    plan = vae_export.plan_vae_exports(
        _LAUNCHED, ["fp32", "fp16"], {"fp32": _LAUNCHED, "fp16": _LAUNCHED}
    )
    assert plan == {"fp32": vae_export.REUSE, "fp16": vae_export.REUSE}


def test_plan_reuse_match_export_stale_and_missing():
    # fp32 matches -> reuse; fp16 is a stale identity -> export; fp8 absent -> export.
    plan = vae_export.plan_vae_exports(
        _LAUNCHED,
        ["fp32", "fp16", "fp8"],
        {"fp32": _LAUNCHED, "fp16": _STALE, "fp8": None},
    )
    assert plan == {
        "fp32": vae_export.REUSE,
        "fp16": vae_export.EXPORT,
        "fp8": vae_export.EXPORT,
    }


# --- ensure_vae_artifacts: the launch-time refuse paths (reached before any GPU) ---


def test_ensure_refuses_unsupported_decoder(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    with pytest.raises(vae_artifacts.TokenStreamUnsupportedDecoder):
        vae_export.ensure_vae_artifacts(
            _wan_pipeline_config(),
            video_width=1280,
            video_height=704,
            precisions=["fp32"],
            device="cpu",
        )


def test_ensure_aborts_when_missing_and_auto_export_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))  # empty -> export needed
    with pytest.raises(vae_export.TokenStreamExportError):
        vae_export.ensure_vae_artifacts(
            OMNIDREAMS_PIPELINE_CONFIG,
            video_width=1280,
            video_height=704,
            precisions=["fp32"],
            device="cpu",
            auto_export=False,
        )
