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

Covers the behaviours a deployment depends on:

* ``_LoRALinear`` is a strict identity at ``scale == 0`` and at the
  zero-initialized ``B`` (so wrapping the network never changes base
  outputs until a trained checkpoint is loaded and gated on).
* ``_apply_lora`` wraps exactly the self-attention projections the
  training-side module wraps (same match rule -> same checkpoint order),
  and ``_set_scale`` reaches every wrapped linear.
* The ``alpha*(t)`` gate profile resolves by nearest-t lookup, including
  the context-noise forward.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omnidreams._drift_corrector import (
    _LORA_RANK,
    GATE_ALPHA,
    _apply_lora,
    _LoRALinear,
    _nearest_alpha,
    _premerge_weight_sets,
    _set_scale,
    _target_linears,
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
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.mlp = nn.Linear(4, 4, bias=False)

    toy = Toy()
    params = _apply_lora(toy)
    wrapped = [m for m in toy.modules() if isinstance(m, _LoRALinear)]
    assert len(wrapped) == 4  # q/k/v/output projections but not the mlp
    assert len(params) == 8  # A + B per wrapped linear
    assert not isinstance(toy.mlp, _LoRALinear)
    _set_scale(toy, 0.25)
    assert all(m.scale == 0.25 for m in wrapped)


def test_gate_profile_nearest_t_lookup():
    assert _nearest_alpha(1000.0) == GATE_ALPHA[1000.0]
    assert _nearest_alpha(803.0) == GATE_ALPHA[803.0]
    # The context-noise forward (t=128) resolves to the low-t entry,
    # matching the evaluated deploy configs.
    assert _nearest_alpha(128.0) == GATE_ALPHA[803.0]
    # The deployed 2-step solver schedule (warped [1000, 350] -> ~[1000,
    # 803]) resolves to the two gate entries in order.
    assert [_nearest_alpha(t) for t in (1000.0, 802.9)] == [
        GATE_ALPHA[1000.0],
        GATE_ALPHA[803.0],
    ]


def test_gate_profile_is_a_strict_attenuation():
    assert all(0.0 < a <= 1.0 for a in GATE_ALPHA.values())


## Pre-merged weight path


def _toy_network() -> nn.Module:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.ModuleDict(
                {
                    n: nn.Linear(4, 4, bias=False)
                    for n in ("q_proj", "k_proj", "v_proj", "output_proj")
                }
            )
            self.mlp = nn.Linear(4, 4, bias=False)

    return nn.Sequential(Block(), Block())


def _random_checkpoint(linears) -> dict[int, torch.Tensor]:
    sd = {}
    for i, lin in enumerate(linears):
        sd[2 * i] = 0.1 * torch.randn(_LORA_RANK, lin.in_features)
        sd[2 * i + 1] = 0.1 * torch.randn(lin.out_features, _LORA_RANK)
    return sd


def test_target_linears_match_apply_lora_load_order():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    wrapped_net = copy.deepcopy(net)
    params = _apply_lora(wrapped_net)
    wrapped = [m for m in wrapped_net.modules() if isinstance(m, _LoRALinear)]
    assert len(params) == 2 * len(linears) == 2 * len(wrapped)
    for lin, w in zip(linears, wrapped):
        assert torch.equal(lin.weight, w.base.weight)  # same walk order


def test_premerged_weights_match_the_unfused_delta():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    sd = _random_checkpoint(linears)
    gain = 0.25
    sets, added_bytes = _premerge_weight_sets(linears, sd, gain)
    assert set(sets) == set(GATE_ALPHA.values())
    x = torch.randn(3, 4)
    for alpha, merged in sets.items():
        for i, lin in enumerate(linears):
            unfused = lin(x) + gain * alpha * (x @ sd[2 * i].T @ sd[2 * i + 1].T)
            premerged = torch.nn.functional.linear(x, merged[i])
            assert torch.allclose(premerged, unfused, atol=1e-5)
    n_weights = sum(lin.weight.numel() for lin in linears)
    assert added_bytes == len(sets) * n_weights * 4  # fp32 toy weights


def test_premerge_does_not_mutate_base_weights():
    torch.manual_seed(0)
    net = _toy_network()
    linears = _target_linears(net)
    before = [lin.weight.detach().clone() for lin in linears]
    _premerge_weight_sets(linears, _random_checkpoint(linears), gain=1.0)
    for lin, w in zip(linears, before):
        assert torch.equal(lin.weight, w)
