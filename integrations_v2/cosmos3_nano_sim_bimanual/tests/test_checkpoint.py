# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Checkpoint aliases, HF loading, and legacy DCP conversion without network I/O."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cosmos3_nano_sim_bimanual.fetch_checkpoint as fetch_checkpoint_module
import pytest
import torch
from cosmos3_nano_sim_bimanual.fetch_checkpoint import (
    DEFAULT_CHECKPOINT,
    HAND_CHECKPOINT_URL,
    KNOWN_CHECKPOINTS,
    _default_out_dir,
    _normalize_raw_key,
    _select_source_prefix,
    checkpoint_denoise_steps,
    ensure_checkpoint,
    fetch_and_convert,
    read_dcp_key_metadata,
    resolve_checkpoint_arg,
)
from cosmos3_nano_sim_bimanual.transformer import HandTransformerConfig
from cosmos3_nano_sim_bimanual.transformer.network import CosmosMoTNetworkConfig

from flashdreams.infra.diffusion.scheduler import FlowMatchSchedulerConfig

pytestmark = pytest.mark.ci_cpu


def test_checkpoint_credentials_prefer_explicit_then_environment_then_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cosmos3_nano_sim_bimanual import checkpoint

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(checkpoint, "__file__", str(tmp_path / "src/checkpoint.py"))
    monkeypatch.delenv(checkpoint.CREDENTIAL_ENV_VAR, raising=False)
    assert checkpoint.resolve_credential_path() is None

    credentials = tmp_path / "credentials"
    credentials.mkdir()
    local = credentials / "s3_checkpoint.secret"
    local.write_text("{}")
    assert checkpoint.resolve_credential_path() == str(local)

    monkeypatch.setenv(checkpoint.CREDENTIAL_ENV_VAR, "/custom/environment.secret")
    assert checkpoint.resolve_credential_path() == "/custom/environment.secret"
    explicit = tmp_path / "explicit.secret"
    assert checkpoint.resolve_credential_path(explicit) == str(explicit)


def test_ga_checkpoint_alias() -> None:
    uri = (
        "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/"
        "304bf3dc45901793f0782254a5270cc087b9a157/"
        "transformer/diffusion_pytorch_model.safetensors.index.json"
    )
    assert HAND_CHECKPOINT_URL == uri
    assert KNOWN_CHECKPOINTS == {DEFAULT_CHECKPOINT: uri}
    assert resolve_checkpoint_arg(DEFAULT_CHECKPOINT) == uri
    assert checkpoint_denoise_steps(DEFAULT_CHECKPOINT) == 4
    assert checkpoint_denoise_steps(uri + "/") == 4
    assert checkpoint_denoise_steps("some/local/checkpoint") is None


def test_hf_alias_uses_core_loader_without_s3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict

    import flashdreams.core.checkpoint.load as core_load

    def unexpected_s3(*args: object, **kwargs: object) -> None:
        pytest.fail("The HF default must not fetch DCPs or resolve S3 credentials")

    monkeypatch.setattr(fetch_checkpoint_module, "fetch_and_convert", unexpected_s3)
    monkeypatch.setattr(
        fetch_checkpoint_module, "resolve_credential_path", unexpected_s3
    )
    raw = {"action_modality_embed": torch.zeros(4)}
    calls: list[tuple[str, str]] = []

    def load(source: str, *, map_location: str) -> dict[str, torch.Tensor]:
        calls.append((source, map_location))
        return raw

    monkeypatch.setattr(core_load, "load_single_checkpoint", load)
    source = ensure_checkpoint(DEFAULT_CHECKPOINT)
    assert _load_raw_state_dict(source, "cpu", None) is raw
    assert calls == [(HAND_CHECKPOINT_URL, "cpu")]


def test_hf_default_matches_ga_sampling_and_action_contract() -> None:
    import hashlib

    from cosmos3_nano_sim_bimanual.action_format import (
        AGIBOT_DOMAIN_ID,
        NORMALIZER_PATH,
        RAW_ACTION_DIM,
    )
    from cosmos3_nano_sim_bimanual.apps.handdreams.config import (
        DEFAULT_HAND_PIPELINE_CONFIG,
    )

    config = DEFAULT_HAND_PIPELINE_CONFIG
    assert isinstance(config.diffusion_model.transformer, HandTransformerConfig)
    assert config.diffusion_model.transformer.checkpoint is not None
    assert (
        resolve_checkpoint_arg(config.diffusion_model.transformer.checkpoint)
        == HAND_CHECKPOINT_URL
    )
    assert isinstance(config.diffusion_model.scheduler, FlowMatchSchedulerConfig)
    assert config.diffusion_model.scheduler.num_inference_steps == 4
    assert config.diffusion_model.scheduler.denoising_timesteps == [1000, 750, 500, 250]
    assert config.diffusion_model.scheduler.shift == 5.0
    assert config.diffusion_model.transformer.chunk_size == 2
    assert config.diffusion_model.transformer.window_frames == 64
    assert AGIBOT_DOMAIN_ID == 15
    assert RAW_ACTION_DIM == 59
    assert hashlib.sha256(NORMALIZER_PATH.read_bytes()).hexdigest() == (
        "ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0"
    )


@pytest.mark.parametrize(
    "value",
    [
        "agibot_tf_dcm",
        "agibot_chunk2_tf_dcm",
        "agibot_chunk2_tf_dcm_2step",
        "agibot_chunk2_tf_dcm_4step",
        "/cache/stage3_sfdmd_chunk2_mecka_agibot_480p_step42__iter_000002000",
        HAND_CHECKPOINT_URL.replace(
            "304bf3dc45901793f0782254a5270cc087b9a157",
            "1d95a0b5b19d49a24aceebf578cf5e85db310ac0",
        ),
    ],
)
def test_pre_ga_checkpoint_fails_before_loading(value: str) -> None:
    with pytest.raises(ValueError, match="retired 29D"):
        ensure_checkpoint(value)
    with pytest.raises(ValueError, match="retired 29D"):
        checkpoint_denoise_steps(value)


def test_vision_tower_toggle_key_counts() -> None:
    # The GCS-fetched student net (``net.*``, no ViT) is 804 keys; the full
    # exported checkpoint (with ViT) is 1155. load_exported_checkpoint auto-picks
    # by presence of ``visual.*`` keys, so both must strict-load exactly.
    with torch.device("meta"):
        no_vit = CosmosMoTNetworkConfig(include_vision_tower=False).setup()
        with_vit = CosmosMoTNetworkConfig(include_vision_tower=True).setup()
    assert len(list(no_vit.state_dict())) == 804
    assert len(list(with_vit.state_dict())) == 1155


def test_action_gen_adds_five_keys() -> None:
    # Camera checkpoint = noaction net (804) + 5 action keys = 809 (matches probe).
    with torch.device("meta"):
        no_action = CosmosMoTNetworkConfig(
            include_vision_tower=False, action_gen=False
        ).setup()
        with_action = CosmosMoTNetworkConfig(
            include_vision_tower=False, action_gen=True
        ).setup()
    assert (
        len(list(with_action.state_dict()))
        == len(list(no_action.state_dict())) + 5
        == 809
    )


def test_default_out_dir_is_collision_free() -> None:
    sources = (
        "s3://bucket-a/project-a/experiment/checkpoints/iter_000004000",
        "s3://bucket-b/project-a/experiment/checkpoints/iter_000004000",
        "s3://bucket-a/project-b/experiment/checkpoints/iter_000004000",
        "s3://bucket-a/project-a/other-experiment/checkpoints/iter_000004000",
    )
    paths = {
        _default_out_dir(source, prefer_ema=prefer_ema, dtype=dtype)
        for source in sources
        for prefer_ema in (True, False)
        for dtype in (torch.bfloat16, torch.float32)
    }
    assert len(paths) == len(sources) * 4
    assert _default_out_dir(sources[0]) == _default_out_dir(sources[0] + "/")


def test_ensure_checkpoint_uses_conversion_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        fetch_checkpoint_module, "resolve_credential_path", lambda *args, **kwargs: None
    )
    legacy = (
        tmp_path
        / "cosmos_interactive"
        / "interact_8b_tfdcm_chunk4_agibot__iter_000001600"
    )
    legacy.mkdir(parents=True)
    legacy_index = legacy / "model.safetensors.index.json"
    legacy_index.write_text("legacy export without provenance")
    calls = []

    def convert(source, credential, out, *, prefer_ema):
        calls.append((source, credential, out, prefer_ema))
        return out

    monkeypatch.setattr(fetch_checkpoint_module, "fetch_and_convert", convert)
    for prefer_ema in (True, False):
        source = "s3://bucket/ga/checkpoints/iter_000001380"
        expected = _default_out_dir(source, prefer_ema=prefer_ema)
        assert ensure_checkpoint(source, prefer_ema=prefer_ema) == str(expected)
        assert calls[-1] == (source, None, expected, prefer_ema)
    assert calls[0][2] != calls[1][2]
    assert all(call[2] != legacy for call in calls)
    with pytest.raises(ValueError, match="retired 29D"):
        ensure_checkpoint(str(legacy))
    assert legacy_index.read_text() == "legacy export without provenance"


def test_default_out_dir_normalizes_local_source(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert _default_out_dir("checkpoint") == _default_out_dir(
        str(tmp_path / "checkpoint")
    )


def test_pathlib_compat_shim_resolves_313_path_class() -> None:
    # A 3.13-pickled DCP .metadata references pathlib._local.PosixPath; the shim
    # must make the unpickler's class lookup resolve it under Python 3.12.
    import io
    import pathlib
    import pickle

    from cosmos3_nano_sim_bimanual.fetch_checkpoint import (
        _install_pathlib_compat_shim,
    )

    _install_pathlib_compat_shim()
    cls = pickle.Unpickler(io.BytesIO(b"")).find_class("pathlib._local", "PosixPath")
    assert cls is pathlib.PosixPath


def test_pathlib_compat_shim_installs_fallback(monkeypatch) -> None:
    import importlib
    import pathlib
    import sys

    monkeypatch.delitem(sys.modules, "pathlib._local", raising=False)

    def missing_pathlib_local(name: str):
        assert name == "pathlib._local"
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(importlib, "import_module", missing_pathlib_local)
    fetch_checkpoint_module._install_pathlib_compat_shim()
    assert sys.modules["pathlib._local"] is pathlib


def _stub_dcp(monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    import torch.distributed.checkpoint as dcp

    entry = SimpleNamespace(size=(2,), properties=SimpleNamespace(dtype=torch.float32))
    monkeypatch.setattr(
        fetch_checkpoint_module,
        "read_dcp_key_metadata",
        lambda *_: (
            source,
            {"net.vae2llm.weight": entry, "net_ema.vae2llm.weight": entry},
        ),
    )
    monkeypatch.setattr(fetch_checkpoint_module, "_storage_reader", lambda *_: object())

    def load(state_dict, **_):
        for key, tensor in state_dict.items():
            tensor.fill_(3 if key.startswith("net_ema.") else 2)

    monkeypatch.setattr(dcp, "load", load)


@pytest.mark.parametrize("prefer_ema", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fetch_and_convert_reuses_only_matching_conversion(
    tmp_path, monkeypatch, prefer_ema, dtype
) -> None:
    from safetensors.torch import load_file

    source = "s3://bucket/experiment/checkpoints/iter_0"
    _stub_dcp(monkeypatch, source)
    assert (
        fetch_and_convert(source, None, tmp_path, prefer_ema=prefer_ema, dtype=dtype)
        == tmp_path
    )
    weights = load_file(tmp_path / "model.safetensors")
    torch.testing.assert_close(
        weights["model.net.vae2llm.weight"],
        torch.full((2,), 3 if prefer_ema else 2, dtype=dtype),
    )
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert json.loads(index["metadata"]["flashdreams_conversion"]) == {
        "source": source,
        "prefer_ema": prefer_ema,
        "dtype": str(dtype),
        "version": 1,
    }

    def unexpected_fetch(*_):
        pytest.fail("cache validation must finish before accessing DCP storage")

    monkeypatch.setattr(
        fetch_checkpoint_module, "read_dcp_key_metadata", unexpected_fetch
    )
    assert (
        fetch_and_convert(
            source + "/", None, tmp_path, prefer_ema=prefer_ema, dtype=dtype
        )
        == tmp_path
    )
    originals = {path: path.read_bytes() for path in tmp_path.iterdir()}
    for changed_source, changed_ema, changed_dtype in (
        (source.replace("bucket", "other-bucket"), prefer_ema, dtype),
        (source, not prefer_ema, dtype),
        (
            source,
            prefer_ema,
            torch.float32 if dtype == torch.bfloat16 else torch.bfloat16,
        ),
    ):
        with pytest.raises(ValueError, match="source or conversion settings differ"):
            fetch_and_convert(
                changed_source,
                None,
                tmp_path,
                prefer_ema=changed_ema,
                dtype=changed_dtype,
            )
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == originals


@pytest.mark.parametrize("failure", ["save", "publish_index"])
def test_ensure_checkpoint_retries_interrupted_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import safetensors.torch

    source = "s3://bucket/experiment/checkpoints/iter_0"
    _stub_dcp(monkeypatch, source)
    monkeypatch.setattr(fetch_checkpoint_module, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(
        fetch_checkpoint_module, "resolve_credential_path", lambda *_, **__: None
    )
    out_dir = _default_out_dir(source)
    weights_path = out_dir / "model.safetensors"
    index_path = out_dir / "model.safetensors.index.json"
    with monkeypatch.context() as interrupted:
        if failure == "save":

            def fail_save(tensors, filename, **kwargs):
                Path(filename).write_bytes(b"incomplete")
                raise OSError("injected interrupted save")

            interrupted.setattr(safetensors.torch, "save_file", fail_save)
        else:
            replace = Path.replace

            def fail_publish(path, target):
                if Path(target) == index_path:
                    raise OSError("injected interrupted publish")
                return replace(path, target)

            interrupted.setattr(Path, "replace", fail_publish)
        with pytest.raises(OSError, match="injected interrupted"):
            ensure_checkpoint(source)

    assert not index_path.exists()
    assert weights_path.exists() == (failure == "publish_index")
    assert not list(tmp_path.glob(".dcp-*"))
    assert ensure_checkpoint(source) == str(out_dir)
    weights = safetensors.torch.load_file(weights_path)
    torch.testing.assert_close(
        weights["model.net.vae2llm.weight"],
        torch.full((2,), 3, dtype=torch.bfloat16),
    )

    def unexpected_fetch(*_):
        pytest.fail("A completed retry must be reused without accessing storage")

    monkeypatch.setattr(
        fetch_checkpoint_module, "read_dcp_key_metadata", unexpected_fetch
    )
    assert ensure_checkpoint(source) == str(out_dir)


def test_fetch_and_convert_reuses_symlinked_cache(tmp_path, monkeypatch) -> None:
    source = "s3://bucket/experiment/checkpoints/iter_0"
    _stub_dcp(monkeypatch, source)
    monkeypatch.setattr(fetch_checkpoint_module, "_cache_dir", lambda: tmp_path)
    out_dir = _default_out_dir(source)
    out_dir.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(out_dir, target_is_directory=True)
    assert fetch_and_convert(source, None, alias) == alias

    def unexpected_fetch(*_):
        pytest.fail("A symlinked cache must reuse the same conversion")

    monkeypatch.setattr(
        fetch_checkpoint_module, "read_dcp_key_metadata", unexpected_fetch
    )
    assert fetch_and_convert(source, None, out_dir) == out_dir


def test_local_checkpoint_bypasses_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_fetch(*args: object, **kwargs: object) -> None:
        pytest.fail("A local checkpoint must not trigger a remote fetch")

    monkeypatch.setattr(fetch_checkpoint_module, "fetch_and_convert", unexpected_fetch)
    assert ensure_checkpoint(str(tmp_path)) == str(tmp_path)


@pytest.mark.parametrize(
    "index_name",
    ["model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"],
)
def test_local_checkpoint_directory_loads_without_download_or_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, index_name: str
) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict
    from safetensors.torch import save_file

    import flashdreams.core.checkpoint.load as core_load

    tensors = {"left": torch.ones(2), "right": torch.zeros(3)}
    for name, tensor in tensors.items():
        save_file({name: tensor}, str(tmp_path / f"{name}.safetensors"))
    (tmp_path / index_name).write_text(
        json.dumps({"weight_map": {name: f"{name}.safetensors" for name in tensors}})
    )

    def unexpected_fetch(*args: object, **kwargs: object) -> None:
        pytest.fail("A local directory must not download or create a merged cache")

    monkeypatch.setattr(core_load, "load_single_checkpoint", unexpected_fetch)
    restored = _load_raw_state_dict(str(tmp_path), "cpu", None)
    assert restored.keys() == tensors.keys()
    for name, tensor in tensors.items():
        torch.testing.assert_close(restored[name], tensor, rtol=0, atol=0)


def test_local_checkpoint_directory_preserves_hf_cache_symlinks(tmp_path: Path) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict
    from safetensors.torch import save_file

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    tensor = torch.ones(2)
    save_file({"key": tensor}, str(tmp_path / "blob"))
    (snapshot / "shard.safetensors").symlink_to("../blob")
    (snapshot / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"key": "shard.safetensors"}})
    )
    restored = _load_raw_state_dict(str(snapshot), "cpu", None)
    torch.testing.assert_close(restored["key"], tensor, rtol=0, atol=0)


def test_local_checkpoint_directory_requires_unambiguous_index(tmp_path: Path) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict

    with pytest.raises(FileNotFoundError, match="No checkpoint index"):
        _load_raw_state_dict(str(tmp_path), "cpu", None)
    for name in (
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        (tmp_path / name).write_text("{}")
    with pytest.raises(ValueError, match="pass an explicit index path"):
        _load_raw_state_dict(str(tmp_path), "cpu", None)


@pytest.mark.parametrize(
    "weight_map",
    [None, {}, [], {"key": None}, {"key": ""}, {"": "shard.safetensors"}],
)
def test_local_checkpoint_directory_rejects_invalid_index(
    tmp_path: Path, weight_map: object
) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict

    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with pytest.raises(ValueError, match="weight_map"):
        _load_raw_state_dict(str(tmp_path), "cpu", None)


@pytest.mark.parametrize(
    "shard", ["/tmp/outside.safetensors", "../outside.safetensors"]
)
def test_local_checkpoint_directory_rejects_shard_path_escape(
    tmp_path: Path, shard: str
) -> None:
    from cosmos3_nano_sim_bimanual.checkpoint import _load_raw_state_dict

    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"key": shard}})
    )
    with pytest.raises(ValueError, match="relative path"):
        _load_raw_state_dict(str(tmp_path), "cpu", None)


def test_dcp_credential_permissions_preserve_actionable_error(tmp_path: Path) -> None:
    credential = tmp_path / "checkpoint.secret"
    credential.write_text("{}")
    credential.chmod(0o644)

    with pytest.raises(PermissionError, match="chmod 600") as failure:
        read_dcp_key_metadata("s3://test-bucket/model", str(credential))
    assert str(credential) in str(failure.value)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize(
    "damage",
    [
        "old_export",
        "json",
        "index_type",
        "null_index",
        "metadata_type",
        "weight_map_type",
        "empty_weight_map",
        "escaped_shard",
        "keys",
        "missing_shard",
        "invalid_shard",
        "weights_provenance",
        "index_provenance",
        "weights_no_provenance",
        "weights_only_provenance",
        "missing_index",
        "truncated_weights_only",
    ],
)
def test_fetch_and_convert_validates_existing_cache(
    tmp_path, monkeypatch, damage, managed
) -> None:
    from safetensors.torch import load_file, save_file

    source = "s3://bucket/experiment/checkpoints/iter_0"
    monkeypatch.setattr(fetch_checkpoint_module, "_cache_dir", lambda: tmp_path)
    out_dir = _default_out_dir(source) if managed else tmp_path / "custom"
    out_dir.mkdir()
    identity = json.dumps(
        {"source": source, "prefer_ema": True, "dtype": "torch.bfloat16", "version": 1},
        sort_keys=True,
    )
    key = "model.net.vae2llm.weight"
    index: dict[str, Any] = {
        "metadata": {"flashdreams_conversion": identity},
        "weight_map": {key: "model.safetensors"},
    }
    if damage == "old_export":
        index["metadata"] = {}
    elif damage == "index_provenance":
        index["metadata"]["flashdreams_conversion"] = "different"
    elif damage == "metadata_type":
        index["metadata"] = None
    elif damage == "weight_map_type":
        index["weight_map"] = []
    elif damage == "empty_weight_map":
        index["weight_map"] = {}
    elif damage == "escaped_shard":
        index["weight_map"][key] = "../outside.safetensors"
    elif damage == "keys":
        index["weight_map"] = {"other-key": "model.safetensors"}
    if damage not in (
        "missing_index",
        "truncated_weights_only",
        "weights_only_provenance",
    ):
        (out_dir / "model.safetensors.index.json").write_text(
            "{"
            if damage == "json"
            else json.dumps(
                None
                if damage == "null_index"
                else []
                if damage == "index_type"
                else index
            )
        )
    if damage in ("invalid_shard", "truncated_weights_only"):
        (out_dir / "model.safetensors").write_bytes(b"incomplete")
    elif damage != "missing_shard":
        save_file(
            {key: torch.zeros(2, dtype=torch.bfloat16)},
            out_dir / "model.safetensors",
            metadata=None
            if damage == "weights_no_provenance"
            else {
                "flashdreams_conversion": "different"
                if damage in ("weights_provenance", "weights_only_provenance")
                else identity
            },
        )
    if managed and damage in (
        "json",
        "missing_shard",
        "invalid_shard",
        "missing_index",
        "truncated_weights_only",
    ):
        _stub_dcp(monkeypatch, source)
        assert fetch_and_convert(source, None, out_dir) == out_dir
        torch.testing.assert_close(
            load_file(out_dir / "model.safetensors")[key],
            torch.full((2,), 3, dtype=torch.bfloat16),
        )
        restored = json.loads((out_dir / "model.safetensors.index.json").read_text())
        assert restored["metadata"]["flashdreams_conversion"] == identity
        assert restored["weight_map"] == {key: "model.safetensors"}
        return
    originals = {path: path.read_bytes() for path in out_dir.iterdir()}

    def unexpected_fetch(*_):
        pytest.fail("an invalid existing cache must fail before accessing storage")

    monkeypatch.setattr(
        fetch_checkpoint_module, "read_dcp_key_metadata", unexpected_fetch
    )
    with pytest.raises(ValueError, match="Cannot reuse DCP cache.*force=True"):
        fetch_and_convert(source, None, out_dir)
    assert {path: path.read_bytes() for path in out_dir.iterdir()} == originals


def test_normalize_raw_key() -> None:
    # DCP keys relative to OmniMoTModel (``net.``) -> exported ``model.net.`` form.
    assert _normalize_raw_key("net.vae2llm.weight") == "model.net.vae2llm.weight"
    # torch.compile shim is stripped.
    assert _normalize_raw_key("_orig_mod.net.llm2vae.bias") == "model.net.llm2vae.bias"
    # Already-exported keys pass through unchanged.
    assert _normalize_raw_key("model.net.time_embedder.mlp.0.weight") == (
        "model.net.time_embedder.mlp.0.weight"
    )
    # EMA-only inference cache: net_ema.* rewrites to the same model.net.* target.
    assert _normalize_raw_key("net_ema.action2llm.fc.weight", "net_ema.") == (
        "model.net.action2llm.fc.weight"
    )


def test_select_source_prefix_prefers_ema_to_match_i4() -> None:
    training_dcp = {"net.x": 0, "net_ema.x": 0, "net_fake_score.x": 0}
    # Default prefers EMA (matches i4 use_ema_weights=True) when both are present.
    assert _select_source_prefix(training_dcp) == "net_ema."
    # prefer_ema=False forces the student net copy.
    assert _select_source_prefix(training_dcp, prefer_ema=False) == "net."
    # net-only -> student regardless of preference.
    assert _select_source_prefix({"net.x": 0}) == "net."
    assert _select_source_prefix({"net.x": 0}, prefer_ema=True) == "net."
    # EMA-only inference cache -> net_ema either way.
    assert _select_source_prefix({"net_ema.x": 0}) == "net_ema."
    assert _select_source_prefix({"net_ema.x": 0}, prefer_ema=False) == "net_ema."
