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

"""Waypoint checkpoint metadata and key-layout validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn

from waypoint.spec import WAYPOINT_1_5, WaypointModelSpec


@dataclass(frozen=True, kw_only=True)
class CheckpointInventory:
    """Small, CPU-only description of a safetensors checkpoint."""

    tensor_count: int
    """Number of tensors in the safetensors artifact."""
    dtypes: dict[str, int]
    """Count of tensors by safetensors dtype label."""
    total_elements: int
    """Total scalar element count across all tensors."""


def expected_waypoint_1_5_checkpoint_shapes(
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> dict[str, tuple[int, ...]]:
    """Return raw checkpoint tensor shapes for the published 1.5 artifact.

    Args:
        spec: Static checkpoint contract that defines the block layout.

    Returns:
        Raw safetensors names mapped to their expected shapes.
    """
    d_model = spec.d_model
    hidden_dim = spec.mlp_ratio * d_model
    kv_dim = spec.n_kv_heads * spec.head_dim
    shapes = {
        "ctrl_cfg.null_emb": (1, 1, d_model),
        "ctrl_emb.mlp.fc1.weight": (hidden_dim, spec.n_buttons + 3),
        "ctrl_emb.mlp.fc2.weight": (d_model, hidden_dim),
        "denoise_step_emb.mlp.fc1.weight": (hidden_dim, 512),
        "denoise_step_emb.mlp.fc2.weight": (d_model, hidden_dim),
        "out_norm.fc.weight": (2 * d_model, d_model),
        "patchify.weight": (
            d_model,
            spec.channels,
            spec.patch_height,
            spec.patch_width,
        ),
        "unpatchify.bias": (spec.channels,),
        "unpatchify.weight": (
            d_model,
            spec.channels,
            spec.patch_height,
            spec.patch_width,
        ),
    }
    for layer_index in range(spec.n_layers):
        prefix = f"transformer.blocks.{layer_index}."
        shapes.update(
            {
                prefix + "attn.k_proj.weight": (kv_dim, d_model),
                prefix + "attn.out_proj.weight": (d_model, d_model),
                prefix + "attn.q_proj.weight": (d_model, d_model),
                prefix + "attn.v_lamb": (),
                prefix + "attn.v_proj.weight": (kv_dim, d_model),
                prefix + "attn_cond_head.bias_in": (d_model,),
                **{
                    prefix + f"attn_cond_head.cond_proj.{index}.weight": (
                        d_model,
                        d_model,
                    )
                    for index in range(3)
                },
                prefix + "dit_mlp.fc1.weight": (hidden_dim, d_model),
                prefix + "dit_mlp.fc2.weight": (d_model, hidden_dim),
                prefix + "mlp_cond_head.bias_in": (d_model,),
                **{
                    prefix + f"mlp_cond_head.cond_proj.{index}.weight": (
                        d_model,
                        d_model,
                    )
                    for index in range(3)
                },
            }
        )
        if layer_index % spec.controller_conditioning_period == 0:
            shapes.update(
                {
                    prefix + "ctrl_mlpfusion.fc1_c.weight": (d_model, d_model),
                    prefix + "ctrl_mlpfusion.fc1_x.weight": (d_model, d_model),
                    prefix + "ctrl_mlpfusion.fc2.weight": (d_model, d_model),
                }
            )
    return shapes


def expected_waypoint_1_5_checkpoint_keys(
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> frozenset[str]:
    """Return raw checkpoint names for the published 1.5 artifact.

    Args:
        spec: Static checkpoint contract that defines the block layout.

    Returns:
        Every raw safetensors key expected from the target checkpoint.
    """
    return frozenset(expected_waypoint_1_5_checkpoint_shapes(spec))


def validate_waypoint_1_5_checkpoint_keys(
    keys: set[str] | frozenset[str] | tuple[str, ...] | list[str],
    *,
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> None:
    """Reject a checkpoint whose raw tensor-key layout differs from Waypoint 1.5.

    Args:
        keys: Raw key names read from a safetensors checkpoint.
        spec: Static checkpoint contract that defines the expected layout.

    Raises:
        ValueError: The artifact contains missing or unexpected tensor keys.
    """
    actual = set(keys)
    expected = expected_waypoint_1_5_checkpoint_keys(spec)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            "Waypoint 1.5 checkpoint key layout mismatch: "
            f"missing={missing[:5]} ({len(missing)} total), "
            f"extra={extra[:5]} ({len(extra)} total)"
        )


def validate_waypoint_1_5_checkpoint_shapes(
    shapes: Mapping[str, tuple[int, ...]],
    *,
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> None:
    """Reject a checkpoint whose tensor shapes differ from Waypoint 1.5.

    Args:
        shapes: Raw tensor names mapped to shapes read from safetensors headers.
        spec: Static checkpoint contract that defines expected tensor shapes.

    Raises:
        ValueError: The artifact contains missing, unexpected, or mismatched tensors.
    """
    expected = expected_waypoint_1_5_checkpoint_shapes(spec)
    validate_waypoint_1_5_checkpoint_keys(tuple(shapes), spec=spec)
    mismatched = sorted(
        key
        for key, expected_shape in expected.items()
        if tuple(shapes[key]) != expected_shape
    )
    if mismatched:
        key = mismatched[0]
        raise ValueError(
            "Waypoint 1.5 checkpoint shape mismatch: "
            f"{key} expected={expected[key]}, actual={tuple(shapes[key])}; "
            f"{len(mismatched)} tensors differ"
        )


def load_waypoint_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    spec: WaypointModelSpec = WAYPOINT_1_5,
) -> None:
    """Validate and strictly load a raw Waypoint checkpoint into ``module``.

    Args:
        module: Native module with the published raw state-dict namespace.
        state_dict: Checkpoint tensors already materialized on the target device.
        spec: Architecture contract used to validate the raw tensor layout.

    Raises:
        ValueError: The checkpoint layout or tensor shapes differ from ``spec``.
        RuntimeError: The module does not expose exactly the validated namespace.
    """
    validate_waypoint_1_5_checkpoint_shapes(
        {key: tuple(tensor.shape) for key, tensor in state_dict.items()}, spec=spec
    )
    module.load_state_dict(state_dict, strict=True)


def inspect_safetensors_checkpoint(path: Path) -> CheckpointInventory:
    """Read safetensors headers without materializing tensor payloads.

    Args:
        path: Local safetensors artifact to inspect.

    Returns:
        Tensor-count, dtype, and element-count metadata.

    Raises:
        ImportError: An importable ``safetensors`` build is unavailable.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - import environment dependent.
        raise ImportError(
            "Inspecting a Waypoint checkpoint requires an importable safetensors build. "
            "Install the FlashDreams checkpoint dependencies first."
        ) from error

    dtypes: Counter[str] = Counter()
    total_elements = 0
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        for key in keys:
            tensor_slice = checkpoint.get_slice(key)
            dtypes[str(tensor_slice.get_dtype())] += 1
            shape = tensor_slice.get_shape()
            elements = 1
            for dimension in shape:
                elements *= dimension
            total_elements += elements
    return CheckpointInventory(
        tensor_count=len(keys), dtypes=dict(dtypes), total_elements=total_elements
    )


def validate_waypoint_1_5_checkpoint(path: Path) -> CheckpointInventory:
    """Validate published Waypoint 1.5 safetensors metadata before native loading.

    This validates raw artifact identity only. The caller must still load it
    into a checkpoint-compatible module.

    Args:
        path: Local Waypoint 1.5 safetensors artifact.

    Returns:
        Validated artifact metadata.

    Raises:
        ImportError: An importable ``safetensors`` build is unavailable.
        ValueError: The raw key layout or metadata differs from Waypoint 1.5.
    """
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - import environment dependent.
        raise ImportError(
            "Validating a Waypoint checkpoint requires an importable safetensors build."
        ) from error

    dtypes: Counter[str] = Counter()
    total_elements = 0
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor_slice = checkpoint.get_slice(key)
            shape = tuple(tensor_slice.get_shape())
            shapes[key] = shape
            dtypes[str(tensor_slice.get_dtype())] += 1
            elements = 1
            for dimension in shape:
                elements *= dimension
            total_elements += elements
    validate_waypoint_1_5_checkpoint_shapes(shapes)
    inventory = CheckpointInventory(
        tensor_count=len(shapes), dtypes=dict(dtypes), total_elements=total_elements
    )
    if inventory.dtypes != {"BF16": 393} or inventory.total_elements != 1_860_823_096:
        raise ValueError(f"Waypoint 1.5 checkpoint metadata mismatch: {inventory}")
    return inventory
