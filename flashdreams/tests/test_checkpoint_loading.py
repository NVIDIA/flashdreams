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

from flashdreams.core.checkpoint.remap import unwrap_generator_state_dict

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("container", ["generator_ema", "generator"])
def test_unwrap_generator_state_dict_strips_training_prefixes(
    container: str,
) -> None:
    """Unwrap generator containers and strip their root training prefixes."""
    model_weight = torch.tensor(1.0)
    net_bias = torch.tensor(2.0)
    fsdp_scale = torch.tensor(3.0)
    untouched = torch.tensor(4.0)

    actual = unwrap_generator_state_dict(
        {
            container: {
                "model.weight": model_weight,
                "net.bias": net_bias,
                "_fsdp_wrapped_module.scale": fsdp_scale,
                "untouched": untouched,
            }
        }
    )

    assert actual == {
        "weight": model_weight,
        "bias": net_bias,
        "scale": fsdp_scale,
        "untouched": untouched,
    }


def test_unwrap_generator_state_dict_prefers_ema_container() -> None:
    """Prefer EMA parameters when both generator containers are present."""
    generator = torch.tensor(1.0)
    generator_ema = torch.tensor(2.0)

    actual = unwrap_generator_state_dict(
        {
            "generator": {"model.weight": generator},
            "generator_ema": {"model._fsdp_wrapped_module.weight": generator_ema},
        }
    )

    assert actual == {"weight": generator_ema}


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
