# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export metadata parsing, history presets and the chunk partition."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from cosmos3_nano_sim_transfer.config import (
    EXPORT_METADATA_KEY,
    LONG_ROLLOUT_HISTORY,
    SYSTEM_PROMPT_ID,
    HistoryProfile,
    TransferModelConfig,
    check_t_list,
    fetch_release,
    skip_schedule,
    steady_from,
)

pytestmark = pytest.mark.ci_cpu

# The settings block of the release's checkpoint.json (revision 42da1b57), verbatim.
CHECKPOINT_JSON_BLOCK = json.loads(
    (Path(__file__).parent / "data" / "checkpoint_metadata.json").read_text()
)[EXPORT_METADATA_KEY]

# The same settings as transformer/config.json states them (flat mRoPE keys, no partition).
TRANSFORMER_CONFIG_BLOCK = {
    key: value
    for key, value in CHECKPOINT_JSON_BLOCK.items()
    if key not in ("mrope", "chunk_partition")
} | {
    "base_fps": 24,
    "enable_fps_modulation": True,
    "temporal_modality_margin": 15000,
    "unified_3d_mrope_reset_spatial_ids": True,
}


def test_release_settings_come_from_checkpoint_json() -> None:
    config = TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK)
    assert config.base_fps == 24.0
    assert config.enable_fps_modulation and config.reset_spatial_ids
    assert config.temporal_modality_margin == 15000.0
    assert config.t_list == (1.0, 0.9375, 0.8333333333333334, 0.625)
    assert config.num_train_timesteps == 1000.0 and config.sample_type == "sde"
    assert config.chunk_partition == (1, 1)
    assert config.export_history == HistoryProfile(51, 1)
    assert config.text_cache_max_len == 512
    assert config.temporal_compression_factor == 4
    assert config.spatial_compression_factor == 16
    assert config.latent_patch_size == 2
    assert config.hints == ("edge", "blur", "depth", "seg")
    assert config.system_prompt_id == SYSTEM_PROMPT_ID == "cosmos3_transfer_v1"


def test_transformer_config_block_gives_the_same_settings() -> None:
    assert TransferModelConfig.from_metadata(
        TRANSFORMER_CONFIG_BLOCK
    ) == TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK)


def test_from_export_prefers_checkpoint_json_and_falls_back(tmp_path: Path) -> None:
    (tmp_path / "transformer").mkdir()
    flat = copy.deepcopy(TRANSFORMER_CONFIG_BLOCK) | {"window_frames": 40}
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({EXPORT_METADATA_KEY: flat})
    )
    assert TransferModelConfig.from_export(tmp_path).window_frames == 40
    (tmp_path / "checkpoint.json").write_text(
        json.dumps({EXPORT_METADATA_KEY: CHECKPOINT_JSON_BLOCK})
    )
    assert TransferModelConfig.from_export(tmp_path).window_frames == 51


def test_from_export_without_metadata_raises(tmp_path: Path) -> None:
    (tmp_path / "checkpoint.json").write_text(json.dumps({"cosmos_dreams": {}}))
    with pytest.raises(FileNotFoundError):
        TransferModelConfig.from_export(tmp_path)


@pytest.mark.parametrize(
    "metadata_path", ["checkpoint.json", "transformer/config.json"]
)
def test_local_export_validates_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_path: str
) -> None:
    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("an explicit export must never download release weights")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)
    metadata = tmp_path / metadata_path
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({EXPORT_METADATA_KEY: CHECKPOINT_JSON_BLOCK}))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "model.safetensors"}}))
    shard = tmp_path / "model.safetensors"
    shard.touch()
    assert fetch_release(tmp_path) == tmp_path

    shard.unlink()
    with pytest.raises(FileNotFoundError, match="weight shard is missing"):
        fetch_release(tmp_path)
    index.unlink()
    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        fetch_release(tmp_path)


@pytest.mark.parametrize("kind", ["missing", "empty", "native-dcp", "incompatible"])
def test_invalid_local_source_never_falls_back_to_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    def unexpected_download(*args: object, **kwargs: object) -> None:
        pytest.fail("an invalid local checkpoint must never download release weights")

    monkeypatch.setattr("huggingface_hub.snapshot_download", unexpected_download)
    root = tmp_path / "trained"
    if kind != "missing":
        root.mkdir()
    if kind == "native-dcp":
        (root / "model").mkdir()
        (root / "model" / ".metadata").touch()
    if kind == "incompatible":
        (root / "checkpoint.json").write_text('{"student_only": true}')
    message = {
        "missing": "directory does not exist",
        "empty": "compatible Transfer export",
        "native-dcp": "native DCP.*student EMA",
        "incompatible": "compatible Transfer export",
    }[kind]
    with pytest.raises((FileNotFoundError, ValueError), match=message):
        fetch_release(root)


def test_history_presets() -> None:
    config = TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK)
    assert config.history("export") == HistoryProfile(window_frames=51, sink_frames=1)
    assert config.history("long-rollout") == LONG_ROLLOUT_HISTORY
    assert LONG_ROLLOUT_HISTORY == HistoryProfile(window_frames=30, sink_frames=3)
    with pytest.raises(ValueError):
        config.history("full")


def test_history_as_window_and_sinks() -> None:
    config = TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK)
    assert config.history("30/3") == LONG_ROLLOUT_HISTORY
    assert config.history(" 12/0 ") == HistoryProfile(window_frames=12, sink_frames=0)
    assert config.history(HistoryProfile(8, 2)) == HistoryProfile(8, 2)
    assert HistoryProfile.parse("51/1") == config.history("export")
    assert str(HistoryProfile(30, 3)) == "30/3"
    for text in ("30", "30-3", "a/b", "3/3", "0/0", "30/3/1"):
        with pytest.raises(ValueError):
            config.history(text)


@pytest.mark.parametrize(("window", "sinks"), [(0, 0), (3, 3), (3, -1)])
def test_history_profile_rejects_impossible_windows(window: int, sinks: int) -> None:
    with pytest.raises(ValueError):
        HistoryProfile(window, sinks)


@pytest.mark.parametrize(
    "partition", [[1, 4], [4, 4], [1, 1, 2]], ids=["1-then-4", "4", "mixed"]
)
def test_only_one_latent_per_step_is_accepted(partition: list[int]) -> None:
    block = copy.deepcopy(CHECKPOINT_JSON_BLOCK) | {"chunk_partition": partition}
    with pytest.raises(ValueError, match="one latent per step"):
        TransferModelConfig.from_metadata(block)


def test_other_control_attention_modes_are_rejected() -> None:
    block = copy.deepcopy(CHECKPOINT_JSON_BLOCK)
    block["conditioning"]["transfer_control_attention_mode"] = "prefill_controls"
    with pytest.raises(ValueError, match="control attention mode"):
        TransferModelConfig.from_metadata(block)


@pytest.mark.parametrize("system_prompt_id", ["cosmos3_transfer_v2", "", None])
def test_other_system_prompts_are_rejected(system_prompt_id: str | None) -> None:
    block = copy.deepcopy(CHECKPOINT_JSON_BLOCK)
    if system_prompt_id is None:
        del block["conditioning"]["system_prompt_id"]
    else:
        block["conditioning"]["system_prompt_id"] = system_prompt_id
    with pytest.raises(ValueError, match="system prompt"):
        TransferModelConfig.from_metadata(block)


def test_export_hints_are_read_and_unknown_kinds_rejected() -> None:
    block = copy.deepcopy(CHECKPOINT_JSON_BLOCK)
    block["conditioning"]["hints"] = ["seg", "depth"]
    assert TransferModelConfig.from_metadata(block).hints == ("seg", "depth")
    for hints in (["seg", "normals"], []):
        block["conditioning"]["hints"] = hints
        with pytest.raises(ValueError, match="control kinds"):
            TransferModelConfig.from_metadata(block)


def test_skip_schedule_drops_the_intermediate_sigmas_after_the_first() -> None:
    release = TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK).t_list
    assert release == (1.0, 0.9375, 0.8333333333333334, 0.625)
    assert skip_schedule(release, 4) == release
    assert skip_schedule(release, 3) == (1.0, 0.8333333333333334, 0.625)
    assert skip_schedule(release, 2) == (1.0, 0.625)
    assert skip_schedule(release, 1) == (1.0,)
    for steps in (0, 5, -1):
        with pytest.raises(ValueError, match="steps must be in"):
            skip_schedule(release, steps)


@pytest.mark.parametrize(
    "bad",
    [(), (1.0, 1.0), (1.0, 0.5, 0.625), (1.0, 0.0), (1.5, 0.625), "1,0.625", None],
    ids=["empty", "repeat", "rising", "zero", "above-one", "string", "none"],
)
def test_check_t_list_refuses_invalid_schedules(bad: object) -> None:
    with pytest.raises(ValueError):
        check_t_list(bad)  # ty: ignore[invalid-argument-type]


def test_with_schedule_replaces_the_sampler_schedule_only() -> None:
    model = TransferModelConfig.from_metadata(CHECKPOINT_JSON_BLOCK)
    assert model.with_schedule() is model
    two = model.with_schedule(steps=2)
    assert two.t_list == (1.0, 0.625)
    assert two == model.with_schedule(t_list=[1, 0.625])
    assert {f: getattr(two, f) for f in vars(two) if f != "t_list"} == {
        f: getattr(model, f) for f in vars(model) if f != "t_list"
    }
    assert model.t_list == (1.0, 0.9375, 0.8333333333333334, 0.625)
    with pytest.raises(ValueError, match="not both"):
        model.with_schedule(steps=2, t_list=(1.0, 0.625))
    with pytest.raises(ValueError, match="must start at 1.0"):
        model.with_schedule(t_list=(0.9375, 0.625))


def test_steady_state_starts_after_the_window_fills() -> None:
    assert steady_from(HistoryProfile(30, 3), 151) == 33
    assert steady_from(HistoryProfile(51, 1), 51) == 3  # the window never fills
    assert steady_from(HistoryProfile(30, 3), 2) == 1


def test_release_snapshot_when_available() -> None:
    """Reads a real release snapshot named by ``COSMOS3_TRANSFER_RELEASE``."""
    root = os.environ.get("COSMOS3_TRANSFER_RELEASE")
    if not root or not (Path(root) / "checkpoint.json").is_file():
        pytest.skip("set COSMOS3_TRANSFER_RELEASE to a release snapshot to check it")
    assert TransferModelConfig.from_export(root) == TransferModelConfig.from_metadata(
        CHECKPOINT_JSON_BLOCK
    )
