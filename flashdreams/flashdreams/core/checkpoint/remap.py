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

"""Checkpoint state-dict unwrapping and key remapping."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from torch import Tensor


def unwrap_generator_state_dict(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    """Unwrap a generator checkpoint and strip root training prefixes.

    ``generator_ema`` takes precedence over ``generator`` when both containers
    are present. A flat state dict passes through without envelope unwrapping.

    Args:
        state_dict: Flat state dict or training checkpoint envelope.

    Returns:
        Flat state dict without one ``model.`` or ``net.`` prefix followed by
        an optional ``_fsdp_wrapped_module.`` prefix.
    """
    if "generator_ema" in state_dict:
        source = state_dict["generator_ema"]
    elif "generator" in state_dict:
        source = state_dict["generator"]
    else:
        source = state_dict
    source = cast(Mapping[str, Tensor], source)

    transformed: dict[str, Tensor] = {}
    for key, value in source.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        elif key.startswith("net."):
            key = key[len("net.") :]
        if key.startswith("_fsdp_wrapped_module."):
            key = key[len("_fsdp_wrapped_module.") :]
        transformed[key] = value
    return transformed


def remap_checkpoint_keys(
    state_dict: dict[str, Tensor], mapping: dict[str, str]
) -> dict[str, Tensor]:
    r"""Rename state-dict keys via regex substitution.

    Each key is matched against ``mapping`` in insertion order; the first
    matching pattern is applied with ``re.sub``. Keys without a match pass
    through unchanged.

    Args:
        state_dict: Source state dict.
        mapping: ``{regex: replacement}`` pairs.

    Returns:
        New state dict with renamed keys; tensors are not copied.

    Examples:

      >>> mapping = {r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2"}
      >>> remapped = remap_checkpoint_keys(state_dict, mapping)
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        matched = False
        for old_key, new_key in mapping.items():
            if re.match(old_key, k):
                new_state_dict[re.sub(old_key, new_key, k)] = v
                matched = True
                break
        if not matched:
            new_state_dict[k] = v
    return new_state_dict
