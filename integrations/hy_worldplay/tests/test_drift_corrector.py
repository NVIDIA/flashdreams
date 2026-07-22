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

"""CPU-only unit tests for the Clean Forcing drift-corrector deploy hook.

Covers the two behaviours a deployment depends on:

* ``_LoRALinear`` is a strict identity at ``scale == 0`` and at the
  zero-initialized ``B`` (so wrapping the network never changes base
  outputs until a trained checkpoint is loaded and gated on).
* ``is_static_trajectory`` keys the content-based selection: all-idle
  pose strings bypass the corrector, any commanded motion enables it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hy_worldplay._drift_corrector import (
    _apply_lora,
    _LoRALinear,
    _set_scale,
    is_static_trajectory,
)

pytestmark = pytest.mark.ci_cpu


def test_lora_linear_is_identity_at_zero_scale():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)  # non-zero delta path
    x = torch.randn(3, 8)
    lora.scale = 0.0
    assert torch.equal(lora(x), base(x))


def test_lora_linear_is_identity_at_zero_init_b():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)  # B is zero-initialized
    lora.scale = 1.0
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x))


def test_lora_linear_applies_scaled_delta():
    torch.manual_seed(0)
    base = nn.Linear(8, 6, bias=False)
    lora = _LoRALinear(base, rank=2)
    nn.init.normal_(lora.A.weight)
    nn.init.normal_(lora.B.weight)
    x = torch.randn(3, 8)
    lora.scale = 0.5
    delta = lora(x) - base(x)
    lora.scale = 1.0
    assert torch.allclose(2.0 * delta, lora(x) - base(x), atol=1e-5)


def test_apply_lora_wraps_only_attention_targets_and_set_scale_reaches_all():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {n: nn.Linear(4, 4, bias=False) for n in ("q", "k", "v", "o")}
            )
            self.ffn = nn.Linear(4, 4, bias=False)

    toy = Toy()
    params = _apply_lora(toy)
    wrapped = [m for m in toy.modules() if isinstance(m, _LoRALinear)]
    assert len(wrapped) == 4  # q/k/v/o but not ffn
    assert len(params) == 8  # A + B per wrapped linear
    assert not isinstance(toy.ffn, _LoRALinear)
    _set_scale(toy, 0.25)
    assert all(m.scale == 0.25 for m in wrapped)


def test_static_pose_json_bypasses_the_corrector(tmp_path):
    # The upstream grammar has no explicit "stay" token; a locked-off
    # camera is an all-identity trajectory JSON (see demo_static.py).
    import json

    import numpy as np

    eye = np.eye(4).tolist()
    intrinsic = [
        [1000.0, 0.0, 960.0],
        [0.0, 1000.0, 540.0],
        [0.0, 0.0, 1.0],
    ]
    n_latents = 16
    poses = {str(i): {"extrinsic": eye, "K": intrinsic} for i in range(n_latents + 1)}
    pose_json = tmp_path / "static_pose.json"
    pose_json.write_text(json.dumps(poses))
    assert is_static_trajectory(pose_json, n_latents=n_latents) is True


def test_commanded_motion_pose_enables_the_corrector():
    assert is_static_trajectory("w-8, s-8", n_latents=16) is False
