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

"""Regex-based renaming of checkpoint state-dict keys."""

import re

from torch import Tensor


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

    Typical usage example:

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
