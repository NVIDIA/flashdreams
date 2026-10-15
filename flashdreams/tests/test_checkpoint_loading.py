# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint loading behavior tests."""

from __future__ import annotations

import importlib
import io
import json
import os
import pickle
import stat
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file as save_safetensors_file

pytestmark = pytest.mark.ci_cpu


def _record_object_load(marker_path: str) -> dict[str, torch.Tensor]:
    Path(marker_path).write_text("loaded", encoding="utf-8")
    return {"weight": torch.ones(1, 1)}


class _NonWeightCheckpointObject:
    def __init__(self, marker_path: str) -> None:
        self.marker_path = marker_path

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return _record_object_load, (self.marker_path,)


@pytest.mark.parametrize("source", ["huggingface", "s3", "distributed-cache"])
def test_checkpoint_loads_reject_non_weight_objects(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use weights-only mode for every core checkpoint source."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    marker_path = tmp_path / "loaded"
    serialized = io.BytesIO()
    torch.save(_NonWeightCheckpointObject(str(marker_path)), serialized)
    checkpoint_bytes = serialized.getvalue()

    if source == "huggingface":
        checkpoint_path = tmp_path / "weights.pt"
        checkpoint_path.write_bytes(checkpoint_bytes)
        monkeypatch.setattr(
            checkpoint_load,
            "_download_checkpoint_from_huggingface_url",
            lambda *_args, **_kwargs: str(checkpoint_path),
        )

        def load() -> object:
            return checkpoint_load.load_single_checkpoint(
                "https://huggingface.co/org/model/resolve/main/weights.pt"
            )

    elif source == "s3":

        class FakeS3FileSystem:
            def __init__(self, credential_path: str) -> None:
                assert credential_path == "credentials"

            def create_stream(self, path: str, mode: str) -> io.BytesIO:
                assert (path, mode) == ("s3://bucket/weights.pt", "rb")
                return io.BytesIO(checkpoint_bytes)

        monkeypatch.setattr(checkpoint_load, "S3FileSystem", FakeS3FileSystem)

        def load() -> object:
            return checkpoint_load._load_checkpoint_from_s3(
                "s3://bucket/weights.pt", ".pt", "credentials"
            )

    else:
        checkpoint_path = Path(
            checkpoint_load._s3_cache_path(
                str(tmp_path), "s3://bucket/model", suffix=".pt"
            )
        )
        checkpoint_path.write_bytes(checkpoint_bytes)
        model = torch.nn.Linear(1, 1, bias=False)

        def load() -> object:
            return checkpoint_load.load_distributed_checkpoint(
                model,
                "s3://bucket/model",
                local_cache_dir=str(tmp_path),
            )

    with pytest.raises(pickle.UnpicklingError):
        load()

    assert not marker_path.exists()


@pytest.mark.parametrize("extension", [".pt", ".pth", ".ckpt"])
def test_pickle_checkpoint_formats_still_load_tensor_state_dicts(
    extension: str,
    tmp_path: Path,
) -> None:
    """Keep tensor-only legacy checkpoint formats working in safe mode."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    checkpoint_path = tmp_path / f"weights{extension}"
    expected = {"weight": torch.arange(3)}
    torch.save(expected, checkpoint_path)

    actual = checkpoint_load.load_single_checkpoint(str(checkpoint_path))

    torch.testing.assert_close(actual["weight"], expected["weight"])


@pytest.mark.parametrize("extension", [".pt", ".safetensors"])
@pytest.mark.parametrize("existing_cache", [False, True])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_cache_write_failure_preserves_published_cache(
    extension: str,
    existing_cache: bool,
    error_type: type[BaseException],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep partial writes out of the cache and clean up after interruptions."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    checkpoint_path = tmp_path / f"weights{extension}"
    original_bytes = b"previous cache"
    if existing_cache:
        checkpoint_path.write_bytes(original_bytes)

    def interrupted_save(_state_dict: object, path: str) -> None:
        temporary_path = Path(path)
        assert (
            temporary_path.parent.stat().st_dev == checkpoint_path.parent.stat().st_dev
        )
        assert temporary_path != checkpoint_path
        temporary_path.write_bytes(b"partial checkpoint")
        if existing_cache:
            assert checkpoint_path.read_bytes() == original_bytes
        else:
            assert not checkpoint_path.exists()
        raise error_type("interrupted serialization")

    if extension == ".safetensors":
        monkeypatch.setattr(checkpoint_load, "save_safetensors", interrupted_save)
    else:
        monkeypatch.setattr(torch, "save", interrupted_save)

    with pytest.raises(error_type, match="interrupted serialization"):
        checkpoint_load._save_to_local_cache(
            {"weight": torch.ones(2)}, str(checkpoint_path), extension
        )

    assert list(tmp_path.iterdir()) == ([checkpoint_path] if existing_cache else [])
    if existing_cache:
        assert checkpoint_path.read_bytes() == original_bytes


@pytest.mark.parametrize("extension", [".pt", ".pth", ".ckpt", ".safetensors"])
@pytest.mark.parametrize("existing_cache", [False, True])
def test_cache_write_publishes_complete_checkpoint(
    extension: str,
    existing_cache: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publish reloadable caches for every supported checkpoint format."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    checkpoint_path = tmp_path / f"weights{extension}"
    if existing_cache:
        checkpoint_path.write_bytes(b"previous cache")
    expected = {"weight": torch.arange(3)}

    checkpoint_load._save_to_local_cache(expected, str(checkpoint_path), extension)
    actual = checkpoint_load.load_single_checkpoint(str(checkpoint_path))

    torch.testing.assert_close(actual["weight"], expected["weight"])
    assert list(tmp_path.iterdir()) == [checkpoint_path]


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX file permissions.")
@pytest.mark.parametrize("extension", [".pt", ".safetensors"])
@pytest.mark.parametrize("existing_mode", [None, 0o600, 0o664])
@pytest.mark.parametrize("umask", [0o022, 0o002, 0o077])
def test_cache_write_preserves_permissions(
    extension: str,
    existing_mode: int | None,
    umask: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Honor creation permissions and retain permissions on cache replacement."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    checkpoint_path = tmp_path / f"weights{extension}"
    expected_mode = 0o600 if extension == ".safetensors" else 0o666 & ~umask
    if existing_mode is not None:
        checkpoint_path.write_bytes(b"previous cache")
        checkpoint_path.chmod(existing_mode)
        expected_mode = existing_mode

    previous_umask = os.umask(umask)
    try:
        checkpoint_load._save_to_local_cache(
            {"weight": torch.ones(2)}, str(checkpoint_path), extension
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(checkpoint_path.stat().st_mode) == expected_mode
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_cache_publish_failure_preserves_existing_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Remove a fully serialized temporary file when publication fails."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")
    monkeypatch.setenv("FLASHDREAMS_MIN_CACHE_FREE_GB", "0")
    checkpoint_path = tmp_path / "weights.safetensors"
    original_bytes = b"previous cache"
    checkpoint_path.write_bytes(original_bytes)

    def reject_replace(source: str, destination: str) -> None:
        assert Path(source).is_file()
        assert destination == str(checkpoint_path)
        raise OSError("publication failed")

    monkeypatch.setattr(checkpoint_load.os, "replace", reject_replace)

    with pytest.raises(OSError, match="publication failed"):
        checkpoint_load._save_to_local_cache(
            {"weight": torch.ones(2)}, str(checkpoint_path), ".safetensors"
        )

    assert checkpoint_path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [checkpoint_path]


def test_s3_cache_path_preserves_distinct_opaque_s3_keys(tmp_path: Path) -> None:
    """Keep different S3 object keys in separate local cache entries."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")

    nested_key_path = checkpoint_load._s3_cache_path(
        str(tmp_path), "s3://bucket/a/../model.pt"
    )
    direct_key_path = checkpoint_load._s3_cache_path(
        str(tmp_path), "s3://bucket/model.pt"
    )

    assert nested_key_path != direct_key_path
    assert Path(nested_key_path).parent == tmp_path
    assert Path(direct_key_path).parent == tmp_path


def test_s3_cache_path_rejects_uri_without_a_bucket(tmp_path: Path) -> None:
    """Reject malformed S3 URIs before creating a cache entry."""
    checkpoint_load = importlib.import_module("flashdreams.core.checkpoint.load")

    with pytest.raises(ValueError, match="must include a bucket"):
        checkpoint_load._s3_cache_path(str(tmp_path), "s3:///tmp/escape.pt")


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
