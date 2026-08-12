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

"""CPU-only unit tests for the pre-merged text-edit LoRA deploy hook.

Covers the deploy invariants:

* ``TextEditLoRA`` merges ``W + B @ A`` correctly, toggles by in-place
  ``copy_`` (stable storage addresses), restores the base bit-exactly,
  and is idempotent.
* With the hook attached, ``replace_text_embeddings`` builds a
  ``use_lora`` window (no KV snapshots), ``predict_flow`` runs a single
  branch on merged weights, the window expiry restores base weights, and
  a fresh rollout resets the hook.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from omnidreams._edit_lora import TextEditLoRA, _target_linears
from omnidreams.transformer import CosmosTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_text_edit import _init_cache, _tiny_transformer  # noqa: E402

pytestmark = pytest.mark.ci_cpu


def _fake_checkpoint(network, *, rank: int = 4, path: Path):
    torch.manual_seed(3)
    linears = _target_linears(network)
    sd = {}
    for i, lin in enumerate(linears):
        sd[2 * i] = torch.randn(rank, lin.in_features) * 0.02  # A
        sd[2 * i + 1] = torch.randn(lin.out_features, rank) * 0.02  # B
    torch.save({"lora": sd}, path)
    return linears, sd


def _make_hooked_transformer(tmp_path) -> tuple[CosmosTransformer, TextEditLoRA]:
    transformer = _tiny_transformer()
    ckpt = tmp_path / "edit_lora.pt"
    _fake_checkpoint(transformer.network, path=ckpt)
    edit_lora = TextEditLoRA(transformer.network, ckpt)
    transformer.set_text_edit_lora(edit_lora)
    return transformer, edit_lora


def test_merge_toggle_and_bit_exact_restore(tmp_path):
    transformer = _tiny_transformer()
    ckpt = tmp_path / "edit_lora.pt"
    linears, sd = _fake_checkpoint(transformer.network, path=ckpt)
    base = [lin.weight.detach().clone() for lin in linears]
    ptrs = [lin.weight.data_ptr() for lin in linears]

    edit_lora = TextEditLoRA(transformer.network, ckpt)
    assert edit_lora.rank == 4
    assert len(linears) == 2 * 8  # 2 tiny blocks x 8 projections

    edit_lora.set_active(True)
    for i, lin in enumerate(linears):
        expected = (
            base[i].to(torch.float32) + sd[2 * i + 1].float() @ sd[2 * i].float()
        ).to(base[i].dtype)
        assert torch.equal(lin.weight, expected)
        assert lin.weight.data_ptr() == ptrs[i]  # in place: CUDA-graph safe
    edit_lora.set_active(True)  # idempotent

    edit_lora.set_active(False)
    for i, lin in enumerate(linears):
        assert torch.equal(lin.weight, base[i])
        assert lin.weight.data_ptr() == ptrs[i]


def test_checkpoint_shape_mismatch_rejected(tmp_path):
    transformer = _tiny_transformer()
    ckpt = tmp_path / "bad.pt"
    torch.save({"lora": {0: torch.zeros(4, 8), 1: torch.zeros(8, 4)}}, ckpt)
    with pytest.raises(AssertionError, match="target-list mismatch"):
        TextEditLoRA(transformer.network, ckpt)


def test_replace_builds_lora_window_and_expiry_restores(tmp_path):
    transformer, edit_lora = _make_hooked_transformer(tmp_path)
    cache, _ = _init_cache(transformer)

    transformer.replace_text_embeddings(
        cache, torch.randn(1, 1, 10, 32), guidance_scale=3.0, guidance_chunks=2
    )
    guidance = cache.text_edit_guidance
    assert guidance is not None and guidance.use_lora
    assert guidance.kv_old == [] and guidance.kv_new == []  # no snapshots
    assert edit_lora.active

    # predict_flow runs a single branch (the stub counts calls).
    calls = []

    def fake_branch(**kwargs):
        calls.append(kwargs["network_cache"])
        return torch.zeros(4)

    transformer._predict_branch = fake_branch  # ty: ignore[invalid-assignment]
    cache.start(0)
    transformer.predict_flow(
        noisy_latent=torch.zeros(4), timestep=torch.tensor(1000.0), cache=cache
    )
    assert len(calls) == 1  # no double branch
    assert edit_lora.active
    cache.finalize(0)

    cache.start(1)  # second (last) guided chunk
    assert cache.text_edit_guidance is not None
    cache.finalize(1)

    cache.start(2)  # countdown expired -> cleared by the cache...
    assert cache.text_edit_guidance is None
    transformer.predict_flow(
        noisy_latent=torch.zeros(4), timestep=torch.tensor(1000.0), cache=cache
    )
    assert not edit_lora.active  # ...and the first forward restores base
    cache.finalize(2)


def test_plain_swap_and_new_rollout_deactivate(tmp_path):
    transformer, edit_lora = _make_hooked_transformer(tmp_path)
    cache, _ = _init_cache(transformer)

    transformer.replace_text_embeddings(
        cache, torch.randn(1, 1, 10, 32), guidance_scale=3.0, guidance_chunks=4
    )
    assert edit_lora.active

    # A plain swap (no guidance) mid-window supersedes it and restores base.
    transformer.replace_text_embeddings(cache, torch.randn(1, 1, 10, 32))
    assert cache.text_edit_guidance is None
    assert not edit_lora.active

    # Mid-window session teardown: a fresh rollout resets the hook.
    transformer.replace_text_embeddings(
        cache, torch.randn(1, 1, 10, 32), guidance_scale=3.0, guidance_chunks=4
    )
    assert edit_lora.active
    _init_cache(transformer, seed=2)
    assert not edit_lora.active
