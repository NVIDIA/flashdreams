# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading behavior tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("sharded", [False, True])
def test_scoped_model_load_is_strict_and_ignores_unselected_shards(tmp_path, sharded):
    """Select codec components without opening unrelated weight shards."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2, bias=False)
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    shard = tmp_path / "decoder.safetensors"
    if sharded:
        save_safetensors_file({"decoder.weight": expected}, shard)
        checkpoint = tmp_path / "model.safetensors.index.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "weight_map": {
                        "decoder.weight": shard.name,
                        "encoder.weight": "absent-unused-shard.safetensors",
                    }
                }
            )
        )
    else:
        checkpoint = shard
        save_safetensors_file(
            {"decoder.weight": expected, "encoder.weight": torch.zeros(1)}, shard
        )
    checkpoint_load.load_checkpoint(
        str(checkpoint), model=model, include_prefixes=("decoder.",)
    )
    torch.testing.assert_close(model.decoder.weight, expected)
    with pytest.raises(RuntimeError, match="match"):
        checkpoint_load.load_checkpoint(
            str(checkpoint), model=model, include_prefixes=("encoder.",)
        )
    with pytest.raises(RuntimeError, match="match"):
        checkpoint_load.load_checkpoint(str(checkpoint), model=model)


def test_scoped_remote_load_filters_before_downloading(monkeypatch, tmp_path):
    """Download only selected shards from a Hub index."""
    module = importlib.import_module("flashdreams.core.checkpoint.load")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "decoder.weight": "wanted.safetensors",
                    "encoder.weight": "unwanted.safetensors",
                }
            }
        )
    )
    shard = tmp_path / "wanted.safetensors"
    save_safetensors_file({"decoder.weight": torch.ones(2, 2)}, shard)
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(index))
    monkeypatch.setattr(
        module, "_preflight_checkpoint_cache_requirement", lambda **kwargs: None
    )
    monkeypatch.setattr(module, "_preflight_hf_cache", lambda **kwargs: 0)
    downloaded = []

    def fetch(**kwargs):
        downloaded.extend(kwargs["shard_files"])
        return {"wanted.safetensors": str(shard)}

    monkeypatch.setattr(module, "_parallel_hf_hub_download_shards", fetch)
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2, bias=False)
    module.load_checkpoint(
        "https://huggingface.co/example/model/blob/abc/model.safetensors.index.json",
        model=model,
        include_prefixes=("decoder.",),
    )
    assert downloaded == ["wanted.safetensors"]


@pytest.mark.parametrize("prefixes", [(), ("",), ("decoder",)])
def test_scoped_load_rejects_ambiguous_prefixes(prefixes):
    from flashdreams.core.checkpoint.load import load_checkpoint

    with pytest.raises(ValueError, match="prefixes"):
        load_checkpoint(
            "unused.safetensors", model=torch.nn.Linear(2, 2), include_prefixes=prefixes
        )


def test_local_safetensors_uses_file_backed_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Load local safetensors without materializing the file as bytes."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    checkpoint_path = tmp_path / "weights.safetensors"
    expected = {"weight": torch.ones(2)}
    calls: list[tuple[str, str]] = []

    def fake_load_file(path: str, *, device: str) -> dict[str, torch.Tensor]:
        calls.append((path, device))
        return expected

    def reject_bytes_load(_data: bytes) -> dict[str, torch.Tensor]:
        pytest.fail("safetensors checkpoints must use the file-backed loader")

    monkeypatch.setattr(checkpoint_load, "load_safetensors_file", fake_load_file)
    monkeypatch.setattr(checkpoint_load, "load_safetensors", reject_bytes_load)

    actual = checkpoint_load.load_single_checkpoint(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
    )

    assert actual is expected
    assert calls == [(str(checkpoint_path), "cpu")]


def test_safetensors_model_load_streams_without_full_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stream safetensors tensors directly into a materialized model."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    checkpoint_path = tmp_path / "weights.safetensors"
    expected = torch.arange(6, dtype=torch.float32).view(2, 3)
    save_safetensors_file({"weight": expected}, checkpoint_path)
    model = torch.nn.Linear(3, 2, bias=False)

    def reject_full_load(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("model loads must not materialize the complete state dict")

    monkeypatch.setattr(checkpoint_load, "load_safetensors_file", reject_full_load)

    actual = checkpoint_load.load_checkpoint(str(checkpoint_path), model=model)

    assert actual is model
    torch.testing.assert_close(model.weight, expected)


def test_sharded_safetensors_model_load_streams_without_merged_state_dict(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Stream indexed safetensors shards into a model without merging first."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    index_path = tmp_path / "model.safetensors.index.json"
    expected_weight = torch.arange(6, dtype=torch.float32).view(2, 3)
    expected_bias = torch.tensor([3.0, 4.0], dtype=torch.float32)
    save_safetensors_file({"weight": expected_weight}, shard_a)
    save_safetensors_file({"bias": expected_bias}, shard_b)
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "weight": shard_a.name,
                    "bias": shard_b.name,
                },
            }
        ),
        encoding="utf-8",
    )
    model = torch.nn.Linear(3, 2)

    def reject_merge(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("sharded model loads must not materialize a merged state dict")

    monkeypatch.setattr(
        checkpoint_load,
        "_load_sharded_safetensors_index_checkpoint",
        reject_merge,
    )

    actual = checkpoint_load.load_checkpoint(str(index_path), model=model)

    assert actual is model
    torch.testing.assert_close(model.weight, expected_weight)
    torch.testing.assert_close(model.bias, expected_bias)
