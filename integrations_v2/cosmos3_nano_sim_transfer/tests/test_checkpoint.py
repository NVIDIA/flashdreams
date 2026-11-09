# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native export preparation without loading model tensors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmos3_nano_sim_transfer.config import TransferModelConfig, fetch_release
from cosmos3_nano_sim_transfer.probes.checkpoint import (
    prepare_export,
    transfer_metadata,
)

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def native_config() -> dict:
    # Resolved native export field names at Framework 5950e7b4. Deliberately
    # distinct numerical values detect accidental use of release defaults.
    return {
        "model": {
            "config": {
                "video_temporal_causal": True,
                "vision_gen": True,
                "action_gen": False,
                "sound_gen": False,
                "precision": "bfloat16",
                "kv_cache_dtype": None,
                "teacher_forcing_frames_per_chunk": 1,
                "teacher_forcing_kv_implementation": "singleview_threeway_kv",
                "teacher_forcing_replay_policy": {
                    "control_visibility": "causal",
                    "controls_read_strict_past_clean_rgb": True,
                    "clean_pass_causality": "frame",
                    "decomposed_temporal_window_seconds": None,
                },
                "tokenizer": {
                    "_target": "wan2pt2_vae_interface",
                    "causal": True,
                    "temporal_compression_factor": 4,
                    "spatial_compression_factor": 16,
                },
                "diffusion_expert_config": {
                    "patch_spatial": 2,
                    "vision_temporal_position_mode": "latent_index",
                    "base_fps": 25,
                    "enable_fps_modulation": True,
                    "unified_3d_mrope_reset_spatial_ids": True,
                    "unified_3d_mrope_temporal_modality_margin": 14000,
                },
                "fixed_step_sampler_config": {
                    "sample_type": "sde",
                    "t_list": [1.0, 0.8, 0.6],
                },
                "rectified_flow_inference_config": {"num_train_timesteps": 900},
                "ar_static_und_cache_max_len": 480,
                "kv_cache_inference_size": 12,
                "attention_sink_size": 2,
            }
        }
    }


def test_native_settings_are_translated_without_release_defaults(
    native_config: dict,
) -> None:
    result = transfer_metadata(
        native_config, checkpoint_id="trained@iter500", hints=["seg"]
    )
    model = TransferModelConfig.from_metadata(result)
    assert model.checkpoint_id == "trained@iter500"
    assert (model.base_fps, model.temporal_modality_margin) == (25, 14000)
    assert (model.window_frames, model.sink_frames, model.text_cache_max_len) == (
        12,
        2,
        480,
    )
    assert (model.t_list, model.num_train_timesteps) == ((1.0, 0.8, 0.6), 900)
    assert model.hints == ("seg",)
    assert result["conditioning"]["emphasize_control_in_prompt"] is False
    del native_config["model"]["config"]["ar_static_und_cache_max_len"]
    with pytest.raises(KeyError, match="ar_static_und_cache_max_len"):
        transfer_metadata(native_config, checkpoint_id="trained", hints=["seg"])


def test_original_hydra_vae_target_is_supported(native_config: dict) -> None:
    tokenizer = native_config["model"]["config"]["tokenizer"]
    del tokenizer["_target"]
    tokenizer["_target_"] = (
        "projects.cosmos3.cosmos3.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface"
    )
    result = transfer_metadata(native_config, checkpoint_id="trained", hints=["seg"])
    assert result["vae_spatial_compression_factor"] == 16


@pytest.mark.parametrize("change", ["global", "chunk", "precision", "vae"])
def test_incompatible_native_execution_contract_is_rejected(
    native_config: dict, change: str
) -> None:
    config = native_config["model"]["config"]
    if change == "global":
        config["teacher_forcing_replay_policy"]["control_visibility"] = "global"
    elif change == "chunk":
        config["teacher_forcing_frames_per_chunk"] = 2
    elif change == "precision":
        config["precision"] = "float16"
    else:
        config["tokenizer"]["_target"] = "another_vae"
    with pytest.raises(ValueError):
        transfer_metadata(native_config, checkpoint_id="trained", hints=["seg"])


@pytest.fixture
def native_export(tmp_path: Path, native_config: dict) -> Path:
    source = tmp_path / "native"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(native_config))
    (source / "checkpoint.json").write_text(
        '{"source_weights": "ema", "student_only": true}'
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.net.vae2llm.weight": "model-00001.safetensors",
                }
            }
        )
    )
    (source / "model-00001.safetensors").write_bytes(b"unchanged test weight bytes")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        (source / filename).write_text("{}")
    return source


def test_preparation_links_original_weights_and_preserves_source(
    native_export: Path, tmp_path: Path
) -> None:
    original = {path.name: path.read_bytes() for path in native_export.iterdir()}
    output = prepare_export(
        native_export,
        tmp_path / "prepared",
        tokenizer=native_export,
        checkpoint_id="trained@500",
        hints=["seg"],
    )
    assert fetch_release(output) == output
    assert (
        output / "model-00001.safetensors"
    ).resolve() == native_export / "model-00001.safetensors"
    assert (output / "text_tokenizer").resolve() == native_export
    assert {
        path.name: path.read_bytes() for path in native_export.iterdir()
    } == original
    report = json.loads((output / "preparation.json").read_text())
    assert report["key_coverage"]["ok"] and report["gpu_parity_verified"] is False
    assert report["export_provenance"]["source_weights"] == "ema"
    with pytest.raises(FileExistsError, match="fresh output"):
        prepare_export(
            native_export,
            output,
            tokenizer=native_export,
            checkpoint_id="trained",
            hints=["seg"],
        )


@pytest.mark.parametrize(
    "invalid", ["regular", "unknown_key", "missing_shard", "missing_tokenizer", "dcp"]
)
def test_invalid_export_never_creates_output(
    native_export: Path, tmp_path: Path, invalid: str
) -> None:
    if invalid == "regular":
        (native_export / "checkpoint.json").write_text('{"source_weights": "regular"}')
    elif invalid == "unknown_key":
        (native_export / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.teacher.unexpected.weight": "model-00001.safetensors",
                    }
                }
            )
        )
    elif invalid == "missing_shard":
        (native_export / "model-00001.safetensors").unlink()
    elif invalid == "missing_tokenizer":
        (native_export / "tokenizer.json").unlink()
    else:
        (native_export / ".metadata").touch()
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_export(
            native_export,
            tmp_path / "prepared",
            tokenizer=native_export,
            checkpoint_id="trained",
            hints=["seg"],
        )
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize(
    "shard",
    [
        "checkpoint.json",
        "preparation.json",
        "../outside.safetensors",
        "/tmp/outside.safetensors",
    ],
)
def test_unsafe_shard_paths_leave_source_untouched(
    native_export: Path,
    tmp_path: Path,
    shard: str,
) -> None:
    (native_export / "preparation.json").write_text("source metadata to preserve")
    (native_export / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.net.vae2llm.weight": shard}})
    )
    original = {path.name: path.read_bytes() for path in native_export.iterdir()}
    with pytest.raises(ValueError, match="Weight shard"):
        prepare_export(
            native_export,
            tmp_path / "prepared",
            tokenizer=native_export,
            checkpoint_id="trained",
            hints=["seg"],
        )
    assert {
        path.name: path.read_bytes() for path in native_export.iterdir()
    } == original
    assert not (tmp_path / "prepared").exists()
    assert not list(tmp_path.glob(".prepared-*"))
