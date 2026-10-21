# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict loading of local and remote Cosmos Interactive checkpoints."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos_interactive._network import CosmosMoTNetwork, CosmosMoTNetworkConfig
from cosmos_interactive.checkpoint_remap import remap_key

CHECKPOINT_ENV_VAR = "COSMOS_INTERACTIVE_CHECKPOINT"
"""Env var giving the default checkpoint source (dir / index.json / HF URL / s3 URI)."""


def resolve_checkpoint_source(
    cli_source: str | os.PathLike | None, *, default: str | None = None
) -> str:
    """Resolve CLI, environment, then an optional integration default."""
    source = (
        str(cli_source) if cli_source else os.environ.get(CHECKPOINT_ENV_VAR) or default
    )
    if not source:
        raise SystemExit(
            "No checkpoint source. Pass it as an argument or set "
            f"${CHECKPOINT_ENV_VAR} (local dir, local model.safetensors.index.json, "
            "a Hugging Face index URL, or an s3:// single-file .safetensors)."
        )
    return source


def _is_remote(source: str) -> bool:
    return (
        source.startswith(("s3://", "http://", "https://"))
        or "huggingface.co" in source
    )


def _shard_files(exported_dir: Path) -> list[str]:
    index = json.loads((exported_dir / "model.safetensors.index.json").read_text())
    return sorted(set(index["weight_map"].values()))


def _load_raw_state_dict(
    source: str, device: str, credential_path: str | None
) -> dict[str, torch.Tensor]:
    """Read the raw (pre-remap) checkpoint tensors from a local dir or a remote source.

    A local directory is read shard-by-shard directly (no extra merge/cache).
    Everything else — a local ``*.index.json``, an HF index URL, or an ``s3://``
    single file — is delegated to FlashDreams' ``load_single_checkpoint``, which
    handles HF parallel-shard download + merge, s3 credentials, and local caching.
    """
    if not _is_remote(source) and Path(source).is_dir():
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
        source: local dir, local ``model.safetensors.index.json``, HF index URL,
            or ``s3://`` single-file ``.safetensors`` (see module docstring).
        device: Where to place the loaded weights (``cuda`` fits a 48 GB card).
        dtype: Parameter dtype (checkpoint is bf16).
        credential_path: S3/GCS credential file for remote ``s3://`` sources.

    Returns:
        The loaded :class:`CosmosMoTNetwork`.

    Raises:
        RuntimeError: if the strict load reports any missing / unexpected key.
    """
    started = time.perf_counter()
    raw = _load_raw_state_dict(str(source), device, credential_path)
    remapped: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        mapped = remap_key(key)
        if mapped is None:
            raise RuntimeError("some source keys did not remap")
        remapped[mapped] = value

    # Auto-detect optional pathways from the remapped keys so the tree matches the
    # checkpoint for a strict load: the full export carries ``visual.*`` (ViT); the
    # camera checkpoint carries ``action2llm.*`` (action conditioning) while the
    # no-action checkpoint carries neither.
    has_vision = any(k.startswith("visual.") for k in remapped)
    has_action = any(k.startswith("action2llm.") for k in remapped)
    lidar_weight = remapped.get("lidar2llm.weight")
    lidar_dim = None if lidar_weight is None else int(lidar_weight.shape[1])
    if (lidar_weight is None) != ("llm2lidar.weight" not in remapped):
        raise ValueError("Joint checkpoints require both LiDAR adapter weights")
    rig_weight = remapped.get("rig_view_embed.weight")
    with torch.device("meta"):
        net = CosmosMoTNetworkConfig(
            dtype=dtype,
            include_vision_tower=has_vision,
            action_gen=has_action,
            lidar_patch_dim=lidar_dim,
            lidar_input_bias="lidar2llm.bias" in remapped,
            lidar_output_bias="llm2lidar.bias" in remapped,
            num_view_embeddings=0 if rig_weight is None else rig_weight.shape[0],
        ).setup()
    assert isinstance(net, CosmosMoTNetwork)
    if not has_vision:
        print("no visual.* keys -> building without the ViT tower (text2video).")
    if has_action:
        print(
            "action2llm.* keys -> building the action-conditioning pathway (camera checkpoint)."
        )

    missing, unexpected = net.load_state_dict(remapped, strict=True, assign=True)  # type: ignore[arg-type]
    if missing or unexpected:
        raise RuntimeError(
            f"non-strict load: missing={missing[:5]} unexpected={unexpected[:5]}"
        )

    elapsed = time.perf_counter() - started
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"strict load OK: {n_params:,} params ({n_params * 2 / 1e9:.1f} GB bf16) "
        f"onto {device} in {elapsed:.1f}s"
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return net
