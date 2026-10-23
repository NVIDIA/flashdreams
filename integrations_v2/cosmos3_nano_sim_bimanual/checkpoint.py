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

"""Strict AgiBot safetensors loading through FlashDreams checkpoint I/O."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from .checkpoint_remap import state_dict_transform

if TYPE_CHECKING:
    from .transformer.network import CosmosMoTNetwork

CREDENTIAL_ENV_VAR = "COSMOS_CHECKPOINT_CREDENTIAL"
"""Env var giving the S3/GCS credential file path for remote ``s3://`` sources."""

# Credential files are searched under ``credentials/`` (relative to CWD, then the
# flashdreams repo root) when none is given explicitly / via the env var.
_DEFAULT_CREDENTIAL_FILES = ("gcp_checkpoint.secret", "s3_checkpoint.secret")


def resolve_credential_path(explicit: str | os.PathLike | None = None) -> str | None:
    """Resolve the S3/GCS credential file: explicit → env var → standard locations.

    Use an explicit path or ``COSMOS_CHECKPOINT_CREDENTIAL`` to select credentials
    when several storage accounts are available.

    Returns ``None`` if nothing is found (fine for local / HF sources, which need
    no credential). Lets the user drop the secret under ``<repo>/credentials/`` and
    never pass ``--credential-path``.
    """
    if explicit:
        return str(explicit)
    env = os.environ.get(CREDENTIAL_ENV_VAR)
    if env:
        return env
    roots: list[Path] = [Path.cwd()]
    roots.extend(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "credentials").is_dir()
    )
    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for fname in _DEFAULT_CREDENTIAL_FILES:
            path = root / "credentials" / fname
            if path.is_file():
                return str(path)
    return None


def _shard_files(exported_dir: Path) -> list[str]:
    indexes = [
        exported_dir / name
        for name in (
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        )
        if (exported_dir / name).is_file()
    ]
    if not indexes:
        raise FileNotFoundError(
            f"No checkpoint index in {exported_dir}; expected "
            "model.safetensors.index.json or diffusion_pytorch_model.safetensors.index.json"
        )
    if len(indexes) != 1:
        raise ValueError(
            f"Multiple checkpoint indexes in {exported_dir}; pass an explicit index path"
        )
    index = json.loads(indexes[0].read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {indexes[0]}")
    shards: set[str] = set()
    for key, shard in weight_map.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(shard, str)
            or not shard
        ):
            raise ValueError(f"Invalid weight_map entry in {indexes[0]}")
        path = Path(shard)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Checkpoint shard must be a relative path without '..': {shard!r}"
            )
        shards.add(shard)
    return sorted(shards)


def _load_raw_state_dict(
    source: str, device: str, credential_path: str | None
) -> dict[str, torch.Tensor]:
    """Read the raw (pre-remap) checkpoint tensors from a local dir or a remote source.

    A native export or Hugging Face transformer directory is read shard-by-shard
    directly (no extra merge/cache).
    Everything else — a local ``*.index.json``, an HF index URL, or an ``s3://``
    single file — is delegated to FlashDreams' ``load_single_checkpoint``, which
    handles HF parallel-shard download + merge, s3 credentials, and local caching.
    """
    if Path(source).is_dir():
        exported_dir = Path(source)
        raw: dict[str, torch.Tensor] = {}
        for shard in _shard_files(exported_dir):
            with safe_open(
                str(exported_dir / shard), framework="pt", device=device
            ) as f:
                for key in f.keys():
                    raw[key] = f.get_tensor(key)
        return raw

    from flashdreams.core.checkpoint.load import load_single_checkpoint

    if credential_path:
        return load_single_checkpoint(
            source, map_location=device, credential_path=credential_path
        )
    return load_single_checkpoint(source, map_location=device)


def load_exported_checkpoint(
    source: str | os.PathLike,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    credential_path: str | None = None,
) -> CosmosMoTNetwork:
    """Build the network and strict-load the remapped exported checkpoint.

    Args:
        source: Native export or HF transformer directory, local index, HF index URL,
            or ``s3://`` single-file ``.safetensors`` (see module docstring).
        device: Where to place the loaded weights (``cuda`` fits a 48 GB card).
        dtype: Parameter dtype (checkpoint is bf16).
        credential_path: S3/GCS credential file for remote ``s3://`` sources.

    Returns:
        The loaded :class:`CosmosMoTNetwork`.

    Raises:
        RuntimeError: if the strict load reports any missing / unexpected key.
    """
    from .transformer.network import CosmosMoTNetwork, CosmosMoTNetworkConfig

    started = time.perf_counter()
    raw = _load_raw_state_dict(str(source), device, credential_path)
    remapped = state_dict_transform(raw)

    # Auto-detect optional pathways from the remapped keys so the tree matches the
    # checkpoint for a strict load: the full export carries ``visual.*`` (ViT); the
    # camera checkpoint carries ``action2llm.*`` (action conditioning) while the
    # no-action checkpoint carries neither.
    has_vision = any(k.startswith("visual.") for k in remapped)
    has_action = any(k.startswith("action2llm.") for k in remapped)
    with torch.device("meta"):
        net = CosmosMoTNetworkConfig(
            dtype=dtype, include_vision_tower=has_vision, action_gen=has_action
        ).setup()
    assert isinstance(net, CosmosMoTNetwork)
    if not has_vision:
        print("no visual.* keys -> building without the ViT tower (text2video).")
    if has_action:
        print(
            "action2llm.* keys -> building the action-conditioning pathway (AgiBot checkpoint)."
        )

    net.load_state_dict(remapped, strict=True, assign=True)

    elapsed = time.perf_counter() - started
    n_params = sum(p.numel() for p in net.parameters())
    param_gb = sum(p.numel() * p.element_size() for p in net.parameters()) / 1e9
    print(
        f"strict load OK: {n_params:,} params ({param_gb:.1f} GB weights) "
        f"onto {device} in {elapsed:.1f}s"
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return net
