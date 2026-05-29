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

"""Checks for the Wan 2.2 TI2V-5B DiT checkpoint key conventions."""

from __future__ import annotations

import pytest


@pytest.mark.manual
def test_native_dit_checkpoint_needs_no_remap() -> None:
    """The native ``Wan-AI/Wan2.2-TI2V-5B`` DiT keys match ``WanDiTNetwork`` exactly.

    Confirms the finding behind :data:`WAN22_TI2V_5B_DIT_NATIVE_PATH`:
    upstream's *native* (non-diffusers) checkpoint already uses our key
    names, so it loads with ``state_dict_transform=None`` -- no analogue
    of :data:`_WAN22_TI2V_5B_DIT_KEY_REMAP` is needed.

    Marked ``manual``: it fetches the ~250 KB sharded-safetensors index
    from HuggingFace (no weights), so it stays out of the offline CI.
    """
    import json
    from urllib.request import urlopen

    import torch
    from wan22.config import WAN22_TI2V_5B_DIT_NATIVE_PATH

    from flashdreams.recipes.wan.transformer.impl.network import (
        WanDiTNetworkTI2V5BConfig,
    )

    with urlopen(WAN22_TI2V_5B_DIT_NATIVE_PATH) as resp:
        index = json.load(resp)
    native_keys = set(index["weight_map"])

    with torch.device("meta"):
        network = WanDiTNetworkTI2V5BConfig().setup()
    model_keys = set(network.state_dict())

    missing = model_keys - native_keys  # would load onto meta -> .to() raises
    extra = native_keys - model_keys  # unexpected keys
    assert not missing, (
        f"{len(missing)} model params absent from native ckpt: {sorted(missing)[:5]}"
    )
    assert not extra, f"{len(extra)} native keys not in the model: {sorted(extra)[:5]}"
    assert native_keys == model_keys
