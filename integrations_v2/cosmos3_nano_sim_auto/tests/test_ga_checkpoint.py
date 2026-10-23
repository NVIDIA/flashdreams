# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GA checkpoint compatibility without downloads or tensor storage."""

import json
from pathlib import Path

import pytest
import torch
from cosmos3_nano_sim_auto.config import DEFAULT_CHECKPOINT
from cosmos_interactive import checkpoint_load

pytestmark = pytest.mark.ci_cpu


def test_ga_tensor_inventory_strict_loads(monkeypatch) -> None:
    inventory = json.loads(
        (Path(__file__).parent / "fixtures/ga_tensor_shapes.json").read_text()
    )
    raw = {}
    for key, shape in inventory["shapes"].items():
        names = (
            [
                key.replace("layers.0.", f"layers.{layer}.")
                for layer in range(inventory["num_layers"])
            ]
            if key.startswith("layers.0.")
            else [key]
        )
        raw.update(
            {
                name: torch.empty(shape, device="meta", dtype=torch.bfloat16)
                for name in names
            }
        )
    monkeypatch.setattr(checkpoint_load, "_load_raw_state_dict", lambda *_: raw)
    network = checkpoint_load.load_exported_checkpoint(
        DEFAULT_CHECKPOINT, device="meta"
    )
    assert len(network.state_dict()) == len(raw) == 809
    assert network.rig_view_embed.weight.shape == (12, 4096)
    assert network.lidar2llm.weight.shape == (4096, 128)


def test_ga_default_preserves_cli_and_environment_overrides(monkeypatch) -> None:
    monkeypatch.delenv(checkpoint_load.CHECKPOINT_ENV_VAR, raising=False)
    assert (
        checkpoint_load.resolve_checkpoint_source(None, default=DEFAULT_CHECKPOINT)
        == DEFAULT_CHECKPOINT
    )
    monkeypatch.setenv(checkpoint_load.CHECKPOINT_ENV_VAR, "/env/checkpoint")
    assert (
        checkpoint_load.resolve_checkpoint_source(None, default=DEFAULT_CHECKPOINT)
        == "/env/checkpoint"
    )
    assert (
        checkpoint_load.resolve_checkpoint_source(
            "/cli/checkpoint", default=DEFAULT_CHECKPOINT
        )
        == "/cli/checkpoint"
    )


def test_unknown_ga_weight_is_rejected() -> None:
    from cosmos_interactive.checkpoint_remap import state_dict_transform

    with pytest.raises(KeyError, match="no target mapping"):
        state_dict_transform({"layers.0.self_attn.unknown.weight": object()})


@pytest.mark.parametrize("route", ["resolve", "blob"])
@pytest.mark.parametrize(
    "revision", ["main", "ccde59bc545540cbbb38389a9c18e2347ba679bf", "release%2Fqa"]
)
def test_hf_transformer_download_stages_matching_lidar_first(
    monkeypatch, route, revision
):
    from cosmos3_nano_sim_auto import checkpoint

    calls = []
    network = object()

    def download(**kwargs):
        calls.append(kwargs)
        return "/cache/" + kwargs["filename"]

    monkeypatch.setattr(checkpoint, "hf_hub_download", download)

    def load(source, *, device):
        calls.append((source, device))
        return network

    monkeypatch.setattr(checkpoint_load, "load_exported_checkpoint", load)
    source = f"https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/{route}/{revision}/model.safetensors.index.json"
    assert checkpoint.load_exported_checkpoint(source, device="cpu") is network
    assert calls == [
        {
            "repo_id": "nvidia/Cosmos3-Nano-Sim-Auto",
            "filename": filename,
            "revision": revision.replace("%2F", "/"),
        }
        for filename in (
            "lidar_vae/config.json",
            "lidar_vae/diffusion_pytorch_model.safetensors",
        )
    ] + [(source, "cpu")]


@pytest.mark.parametrize(
    "source",
    [
        "/local/model.safetensors.index.json",
        "s3://checkpoints/student.safetensors",
        "https://huggingface.co/nvidia/another-model/resolve/main/model.safetensors.index.json",
        "https://huggingface.co.example/nvidia/Cosmos3-Nano-Sim-Auto/resolve/main/model.safetensors.index.json",
    ],
)
def test_other_checkpoint_sources_do_not_download_sim_auto_assets(monkeypatch, source):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint, "hf_hub_download", lambda **_: pytest.fail("Unexpected HF download")
    )
    network = object()
    monkeypatch.setattr(
        checkpoint_load, "load_exported_checkpoint", lambda *a, **kw: network
    )
    assert checkpoint.load_exported_checkpoint(source, device="cpu") is network


def test_failed_lidar_download_stops_model_preparation(monkeypatch):
    from cosmos3_nano_sim_auto import checkpoint

    def denied(**kwargs):
        raise PermissionError("HF repository access required")

    monkeypatch.setattr(checkpoint, "hf_hub_download", denied)
    monkeypatch.setattr(
        checkpoint_load,
        "load_exported_checkpoint",
        lambda *a, **kw: pytest.fail("Loaded transformer with missing companion"),
    )
    with pytest.raises(PermissionError, match="repository access"):
        checkpoint.load_exported_checkpoint(DEFAULT_CHECKPOINT, device="cpu")


@pytest.mark.parametrize("host", ["huggingface.co", "www.huggingface.co"])
def test_auto_lidar_uses_returned_hf_snapshot_path(monkeypatch, tmp_path, host):
    from cosmos3_nano_sim_auto import checkpoint

    blob = tmp_path / "blob-without-extension"
    blob.touch()
    cached = (
        tmp_path / "snapshots/release/lidar_vae/diffusion_pytorch_model.safetensors"
    )
    cached.parent.mkdir(parents=True)
    cached.symlink_to(blob)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return (
            str(cached)
            if kwargs["filename"].endswith(".safetensors")
            else str(tmp_path / "config.json")
        )

    monkeypatch.setattr(checkpoint, "hf_hub_download", download)
    result = checkpoint.resolve_lidar_checkpoint(
        f"https://{host}/nvidia/Cosmos3-Nano-Sim-Auto/resolve/release%2Fqa/model.safetensors.index.json"
    )
    assert result == cached and result.suffix == ".safetensors"
    assert result != blob
    assert [call["revision"] for call in calls] == ["release/qa", "release/qa"]


@pytest.mark.parametrize("kind", ["directory", "index"])
def test_auto_lidar_uses_local_export_companion(monkeypatch, tmp_path, kind):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint, "hf_hub_download", lambda **_: pytest.fail("Unexpected HF download")
    )
    codec = tmp_path / "lidar_vae/diffusion_pytorch_model.safetensors"
    codec.parent.mkdir()
    codec.touch()
    source = (
        tmp_path if kind == "directory" else tmp_path / "model.safetensors.index.json"
    )
    assert checkpoint.resolve_lidar_checkpoint(str(source)) == codec


def test_lidar_override_preserves_missing_path_and_v1p1_without_download(
    monkeypatch, tmp_path
):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint, "hf_hub_download", lambda **_: pytest.fail("Unexpected HF download")
    )
    explicit = tmp_path / "custom.pt"
    assert (
        checkpoint.resolve_lidar_checkpoint(
            DEFAULT_CHECKPOINT, checkpoint=explicit, version="v1p1"
        )
        == explicit
    )
    with pytest.raises(ValueError, match="v1p1 needs --lidar-checkpoint"):
        checkpoint.resolve_lidar_checkpoint(DEFAULT_CHECKPOINT, version="v1p1")


@pytest.mark.parametrize(
    "source",
    [
        "/missing/model.safetensors.index.json",
        "s3://custom/model.safetensors",
        "https://huggingface.co/nvidia/other-model/resolve/main/model.safetensors.index.json",
    ],
)
def test_auto_lidar_rejects_sources_without_matching_companion(monkeypatch, source):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint, "hf_hub_download", lambda **_: pytest.fail("Unexpected HF download")
    )
    with pytest.raises(ValueError, match="No matching LiDAR companion"):
        checkpoint.resolve_lidar_checkpoint(source)
