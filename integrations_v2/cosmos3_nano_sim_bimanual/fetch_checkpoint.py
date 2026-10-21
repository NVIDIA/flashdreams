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

"""AgiBot checkpoint aliases and optional DCP conversion via FlashDreams I/O."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import torch
from filelock import FileLock

from flashdreams.core.checkpoint.load import get_storage_reader
from flashdreams.core.io.disk import default_flashdreams_cache_dir

from .checkpoint import resolve_credential_path
from .checkpoint_remap import remap_key


def _cache_dir() -> Path:
    """Return the shared cache root while retaining the legacy namespace."""
    return default_flashdreams_cache_dir() / "cosmos_interactive"


HAND_CHECKPOINT_URL = (
    "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/resolve/"
    "304bf3dc45901793f0782254a5270cc087b9a157/"
    "transformer/diffusion_pytorch_model.safetensors.index.json"
)
"""Pinned GA iteration-1380 EMA export with the unified_v1 action contract."""

DEFAULT_CHECKPOINT = "cosmos3_nano_sim_bimanual_ga"

# Do not repoint legacy aliases: their weights expect a different action codec.
KNOWN_CHECKPOINTS: dict[str, str] = {
    DEFAULT_CHECKPOINT: HAND_CHECKPOINT_URL,
}

CHECKPOINT_DENOISE_STEPS: dict[str, int] = {
    DEFAULT_CHECKPOINT: 4,
}


def checkpoint_denoise_steps(value: str | None) -> int | None:
    """Return the demo's sampling-step preset for a known alias or URI."""
    if not value:
        return None
    resolve_checkpoint_arg(value)
    if value in CHECKPOINT_DENOISE_STEPS:
        return CHECKPOINT_DENOISE_STEPS[value]
    normalized = value.rstrip("/")
    for alias, steps in CHECKPOINT_DENOISE_STEPS.items():
        if KNOWN_CHECKPOINTS[alias].rstrip("/") == normalized:
            return steps
    return None


def resolve_checkpoint_arg(value: str) -> str:
    """Resolve a registered alias; leave explicit URLs and local paths unchanged."""
    if (
        value
        in {
            "agibot_tf_dcm",
            "agibot_chunk2_tf_dcm",
            "agibot_chunk2_tf_dcm_2step",
            "agibot_chunk2_tf_dcm_4step",
        }
        or "1d95a0b5b19d49a24aceebf578cf5e85db310ac0" in value
        or "stage3_sfdmd_chunk2_mecka_agibot_480p" in value
        or "interact_8b_tfdcm_chunk4_agibot" in value
    ):
        raise ValueError(
            "This pre-GA checkpoint requires the retired 29D action codec. "
            f"Remove --checkpoint to use {DEFAULT_CHECKPOINT}, or provide a GA export."
        )
    return KNOWN_CHECKPOINTS.get(value, value)


def ensure_checkpoint(
    spec: str, *, credential_path: str | None = None, prefer_ema: bool = True
) -> str:
    """Resolve a checkpoint alias, URL, or local export for loading.

    Hugging Face URLs pass to FlashDreams' checkpoint loader, which uses the
    standard HF cache and token. Legacy raw S3 DCPs are converted into a local
    export; S3 credentials use the explicit path, environment, or local files.
    """
    resolved = resolve_checkpoint_arg(str(spec))
    if not resolved.startswith("s3://"):
        return resolved  # local dir / index.json / HF URL -> load_exported_checkpoint handles it
    if resolved.rstrip("/").endswith(".safetensors"):
        return resolved  # already an exported single-file on s3
    cred = resolve_credential_path(credential_path)
    out = _default_out_dir(resolved, prefer_ema=prefer_ema)
    print(
        f"[fetch] ensuring checkpoint is local: {resolved}\n        -> {out}",
        flush=True,
    )
    return str(fetch_and_convert(resolved, cred, out, prefer_ema=prefer_ema))


def _select_source_prefix(raw_meta: dict[str, Any], prefer_ema: bool = True) -> str:
    """Pick which model copy to convert from a DCP that holds several.

    A training DCP holds ``net`` (student), ``net_ema`` (EMA), and
    ``net_fake_score`` (critic). Prefer ``net_ema.`` when present
    (``prefer_ema=True``, default);
    fall back to the student ``net.`` otherwise. Pass ``prefer_ema=False`` to force
    the student copy. An EMA-only cache resolves to the EMA weights either way.
    """
    top = {k.split(".", 1)[0] for k in raw_meta}
    if prefer_ema and "net_ema" in top:
        return "net_ema."
    if "net" in top:
        return "net."
    if "net_ema" in top:
        return "net_ema."
    return "net."  # default; nothing will map if the layout is unexpected


def _normalize_raw_key(key: str, source_prefix: str = "net.") -> str:
    """Normalize a raw DCP key toward the exported ``model.net.*`` convention.

    The exported checkpoint keys start with ``model.net.``; the raw DCP (relative
    to ``OmniMoTModel``) starts with ``net.`` (student) or ``net_ema.`` (EMA-only
    inference cache). ``source_prefix`` selects which copy; it is rewritten to
    ``model.net.`` so both flow through the same remap. Also strips a
    ``torch.compile`` ``_orig_mod.`` shim.
    """
    key = key.replace("_orig_mod.", "")
    if source_prefix and key.startswith(source_prefix):
        return "model.net." + key[len(source_prefix) :]
    return key


def _storage_reader(path: str, credential_path: str | None) -> Any:
    if credential_path:
        return get_storage_reader(path, credential_path=credential_path)
    return get_storage_reader(path)


def _install_pathlib_compat_shim() -> None:
    """Let a Python-3.13-pickled DCP ``.metadata`` unpickle under Python < 3.13.

    Python 3.13 split ``pathlib`` into a package with ``pathlib._local`` (where
    ``PosixPath`` etc. now live), so a ``.metadata`` pickled by a 3.13 env
    references ``pathlib._local.PosixPath``. Under 3.12 ``pathlib`` is a single
    module and that import fails (``'pathlib' is not a package``). Aliasing the
    submodule name to the ``pathlib`` module resolves the reference (3.12's
    ``pathlib.PosixPath`` is the same class).
    """
    import importlib
    import pathlib
    import sys

    try:
        importlib.import_module("pathlib._local")
    except ModuleNotFoundError as exc:
        if exc.name != "pathlib._local":
            raise
        sys.modules["pathlib._local"] = pathlib


def _resolve_dcp_dir(
    checkpoint_path: str, credential_path: str | None
) -> tuple[str, Any]:
    """Find the DCP dir that actually holds ``.metadata``.

    Checkpoints may store model weights in a ``model`` subdirectory. Try
    ``<path>`` then ``<path>/model`` and return whichever reads.
    """
    _install_pathlib_compat_shim()  # tolerate 3.13-pickled caches under 3.12
    base = checkpoint_path.rstrip("/")
    tried: list[str] = []
    for candidate in (base, base + "/model"):
        try:
            metadata = _storage_reader(candidate, credential_path).read_metadata()
            return candidate, metadata
        except PermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 - probe which layout the bucket uses
            tried.append(f"{candidate} ({type(exc).__name__})")
    raise RuntimeError("could not read a DCP .metadata; tried: " + "; ".join(tried))


def read_dcp_key_metadata(
    checkpoint_path: str, credential_path: str | None
) -> tuple[str, dict[str, Any]]:
    """Return ``(resolved_dcp_dir, {raw_key: TensorStorageMetadata})`` (no tensor reads)."""
    resolved, metadata = _resolve_dcp_dir(checkpoint_path, credential_path)
    return resolved, dict(metadata.state_dict_metadata)


def _tensor_shape_dtype(entry: Any) -> tuple[tuple[int, ...], Any] | None:
    """Extract ``(shape, dtype)`` from a DCP metadata entry, or None if not a tensor."""
    size = getattr(entry, "size", None)
    props = getattr(entry, "properties", None)
    dtype = getattr(props, "dtype", None)
    if size is None or dtype is None:
        return None
    return tuple(int(s) for s in size), dtype


def _conversion_identity(
    checkpoint_path: str, *, dtype: torch.dtype, prefer_ema: bool
) -> str:
    """Return the full source and conversion settings recorded with cached weights."""
    source = (
        checkpoint_path.rstrip("/")
        if checkpoint_path.startswith("s3://")
        else str(Path(checkpoint_path).resolve())
    )
    return json.dumps(
        {"source": source, "prefer_ema": prefer_ema, "dtype": str(dtype), "version": 1},
        sort_keys=True,
    )


def _default_out_dir(
    checkpoint_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    prefer_ema: bool = True,
) -> Path:
    """Return a cache directory keyed by the full source and conversion settings."""
    identity = _conversion_identity(checkpoint_path, dtype=dtype, prefer_ema=prefer_ema)
    return _cache_dir() / ("dcp-" + hashlib.sha256(identity.encode()).hexdigest())


def fetch_and_convert(
    checkpoint_path: str,
    credential_path: str | None,
    out_dir: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
    prefer_ema: bool = True,
) -> Path:
    """Load the mapped weights from the DCP checkpoint, remap, and cache as safetensors.

    Reuse ``out_dir`` only when its index and weights match the requested source
    and conversion settings. Incomplete managed caches are rebuilt; unidentified
    or mismatched caches and invalid custom output directories require
    ``force=True``. Strict model key/shape validation happens at network load.
    """
    target = out_dir.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        _fetch_and_convert(
            checkpoint_path,
            credential_path,
            target,
            dtype=dtype,
            force=force,
            prefer_ema=prefer_ema,
        )
    return out_dir


def _fetch_and_convert(
    checkpoint_path: str,
    credential_path: str | None,
    out_dir: Path,
    *,
    dtype: torch.dtype,
    force: bool,
    prefer_ema: bool,
) -> Path:
    from safetensors import SafetensorError, safe_open
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    index_path = out_dir / "model.safetensors.index.json"
    weights_path = out_dir / "model.safetensors"
    identity = _conversion_identity(checkpoint_path, dtype=dtype, prefer_ema=prefer_ema)
    managed = (
        out_dir.resolve()
        == _default_out_dir(
            checkpoint_path, dtype=dtype, prefer_ema=prefer_ema
        ).resolve()
    )
    if not force and (index_path.exists() or weights_path.exists()):
        try:
            try:
                index = json.loads(index_path.read_text())
                if not isinstance(index, dict):
                    raise ValueError("invalid converted-checkpoint index")
            except (FileNotFoundError, json.JSONDecodeError):
                if not managed:
                    raise
                index = None
            if index is not None:
                if index["metadata"]["flashdreams_conversion"] != identity:
                    raise ValueError("checkpoint source or conversion settings differ")
                weight_map = index["weight_map"]
                if (
                    not isinstance(weight_map, dict)
                    or not weight_map
                    or set(weight_map.values()) != {weights_path.name}
                ):
                    raise ValueError("invalid converted-checkpoint weight_map")
            weights_complete = False
            try:
                with safe_open(weights_path, framework="pt", device="cpu") as weights:
                    if (weights.metadata() or {}).get(
                        "flashdreams_conversion"
                    ) != identity:
                        raise ValueError("checkpoint weights have different provenance")
                    if index is not None and set(weights.keys()) != set(weight_map):
                        raise ValueError("checkpoint weights do not match the index")
                    weights_complete = True
            except (FileNotFoundError, SafetensorError):
                if not managed:
                    raise
        except (OSError, ValueError, KeyError, TypeError, SafetensorError) as exc:
            raise ValueError(
                f"Cannot reuse DCP cache at {out_dir}: {exc}. "
                "Choose another output directory or use force=True to rebuild."
            ) from exc
        if index is not None and weights_complete:
            print(f"already cached: {len(weight_map)} tensors -> {out_dir}")
            return out_dir
        print(f"rebuilding interrupted DCP conversion -> {out_dir}", flush=True)

    resolved, raw_meta = read_dcp_key_metadata(checkpoint_path, credential_path)
    # Select the weights our remap accepts (``net_ema.*`` by default,
    # else student ``net.*``); skip the
    # ``net_fake_score`` critic. Load keyed by the raw DCP key (DCP matches by
    # name), then re-key to the exported ``model.net.*`` form so the cached file
    # loads through the standard ``load_exported_checkpoint`` remap.
    source_prefix = _select_source_prefix(raw_meta, prefer_ema)
    load_sd: dict[str, torch.Tensor] = {}
    raw_to_exported: dict[str, str] = {}
    for raw_key, entry in raw_meta.items():
        exported = _normalize_raw_key(raw_key, source_prefix)
        if remap_key(exported) is None:
            continue
        sd = _tensor_shape_dtype(entry)
        if sd is None:
            continue
        shape, src_dtype = sd
        load_sd[raw_key] = torch.empty(shape, dtype=src_dtype)
        raw_to_exported[raw_key] = exported
    if not load_sd:
        raise RuntimeError("no DCP keys mapped to the AgiBot MoT network")
    print(
        f"source copy: {source_prefix.rstrip('.')} "
        f"({'EMA' if source_prefix == 'net_ema.' else 'student net'})"
    )

    approx_gb = sum(t.numel() * t.element_size() for t in load_sd.values()) / 1e9
    n_action = sum(1 for k in raw_to_exported.values() if "action" in k)
    print(
        f"reading {len(load_sd)} tensors (~{approx_gb:.1f} GB, {n_action} action) from {resolved} ...\n"
        "  (this is the slow, network-bound DCP read; it is silent until done)",
        flush=True,
    )
    reader = _storage_reader(resolved, credential_path)
    dcp_load(
        load_sd,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    exported = {
        raw_to_exported[k]: v.to(dtype).contiguous() for k, v in load_sd.items()
    }
    index = {
        "metadata": {
            "total_size": sum(v.numel() * v.element_size() for v in exported.values()),
            "flashdreams_conversion": identity,
        },
        "weight_map": {k: "model.safetensors" for k in exported},
    }
    # Stage both writes before publishing; the index is the completion marker.
    with tempfile.TemporaryDirectory(prefix=".dcp-", dir=out_dir.parent) as temporary:
        stage = Path(temporary)
        save_file(
            exported,
            str(stage / weights_path.name),
            metadata={"flashdreams_conversion": identity},
        )
        (stage / index_path.name).write_text(json.dumps(index))
        out_dir.mkdir(parents=True, exist_ok=True)
        (stage / weights_path.name).replace(weights_path)
        (stage / index_path.name).replace(index_path)
    print(
        f"fetched + converted {len(exported)} tensors ({source_prefix.rstrip('.')}, no ViT) -> {out_dir}"
    )
    return out_dir
