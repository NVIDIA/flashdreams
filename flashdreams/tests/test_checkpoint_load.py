# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from flashdreams.core.checkpoint.load import (
    load_checkpoint,
    load_sharded_safetensors_checkpoint_into_model,
)

pytestmark = pytest.mark.ci_cpu


class _TinyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, 3)
        self.norm = torch.nn.LayerNorm(3)


def _state_for(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for index, (key, tensor) in enumerate(module.state_dict().items(), start=1):
        values = torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
        state[key] = values + index
    return state


def _write_sharded_checkpoint(
    tmp_path: Path,
    state: dict[str, torch.Tensor],
) -> Path:
    shard_1_keys = ["proj.weight", "proj.bias"]
    shard_2_keys = [key for key in state if key not in shard_1_keys]
    shard_1 = tmp_path / "model-00001-of-00002.safetensors"
    shard_2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file({key: state[key] for key in shard_1_keys}, shard_1)
    save_file({key: state[key] for key in shard_2_keys}, shard_2)

    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = {
        **{key: shard_1.name for key in shard_1_keys},
        **{key: shard_2.name for key in shard_2_keys},
    }
    index_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(tensor.nbytes for tensor in state.values())
                },
                "weight_map": weight_map,
            }
        ),
        encoding="utf-8",
    )
    return index_path


def test_stream_sharded_safetensors_checkpoint_loads_meta_module(tmp_path: Path) -> None:
    expected = _state_for(_TinyModule())
    index_path = _write_sharded_checkpoint(tmp_path, expected)

    with torch.device("meta"):
        target = _TinyModule()

    assert all(tensor.is_meta for tensor in target.state_dict().values())

    load_sharded_safetensors_checkpoint_into_model(
        target,
        str(index_path),
        assign=True,
    )

    for key, tensor in target.state_dict().items():
        assert not tensor.is_meta
        torch.testing.assert_close(tensor, expected[key])


def test_stream_sharded_safetensors_strict_detects_missing_keys(
    tmp_path: Path,
) -> None:
    expected = _state_for(_TinyModule())
    expected.pop("norm.bias")
    index_path = _write_sharded_checkpoint(tmp_path, expected)

    with torch.device("meta"):
        target = _TinyModule()

    with pytest.raises(RuntimeError, match="Missing key"):
        load_sharded_safetensors_checkpoint_into_model(
            target,
            str(index_path),
            assign=True,
        )


def test_local_safetensors_checkpoint_loads_from_file(tmp_path: Path) -> None:
    expected = _state_for(_TinyModule())
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(expected, checkpoint_path)

    loaded = load_checkpoint(str(checkpoint_path))

    assert set(loaded) == set(expected)
    for key, tensor in loaded.items():
        torch.testing.assert_close(tensor, expected[key])
