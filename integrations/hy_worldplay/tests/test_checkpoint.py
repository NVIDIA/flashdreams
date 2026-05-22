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

"""CPU smoke tests for HY-WorldPlay's distilled state-dict transform.

The phase-2b.5b weight remap is exercised end-to-end against a
synthetic state dict that mirrors upstream's distilled-checkpoint
shape (top-level ``generator`` envelope, ``model.`` and
``_fsdp_wrapped_module.`` prefixes, diffusers
``WanTransformer3DModel`` keys plus the HY-specific
``action_embedder`` / ``to_out_prope`` extras). No GPU and no
checkpoint file required -- the goal is to lock in the rewrite rules
so a regression in the regex remap is caught immediately by
``uv run pytest integrations/hy_worldplay/tests``.

The matching parity check that loads the *real* distilled checkpoint
into a constructed :class:`HyWorldPlayWanDiTNetwork` and asserts
``strict=True`` succeeds is GPU-only (it builds the full 30-block
network) and is documented in the README -- it lives outside this
file so the CPU smoke stays fast.
"""

from __future__ import annotations

import pytest
import torch

from hy_worldplay._checkpoint import hy_worldplay_distilled_state_dict_transform

pytestmark = pytest.mark.ci_cpu


def _make_synthetic_distilled_state_dict() -> dict[str, dict[str, torch.Tensor]]:
    """Build a tiny envelope that exercises every branch of the transform.

    Returns the raw two-level dict ``torch.load`` would produce on
    upstream's ``wan_distilled_model/model.pt``: top-level
    ``generator`` and ``generator_ema`` subkeys, both with the
    ``model.`` prefix; ``generator_ema`` additionally has the
    ``_fsdp_wrapped_module.`` prefix at the block-level. We pick
    one representative key per category so the assertions can pin
    every rewrite path without instantiating the full 889-key tree.
    """
    one = torch.ones(())  # marker tensor; we only check identity / keys
    base_keys = {
        # Standard Wan 2.2 5B diffusers keys (covered by the base remap).
        "model.condition_embedder.text_embedder.linear_1.weight": one,
        "model.condition_embedder.time_embedder.linear_2.bias": one,
        "model.blocks.0.attn1.to_q.weight": one,
        "model.blocks.0.attn1.norm_k.weight": one,
        "model.blocks.0.attn2.to_out.0.bias": one,
        "model.blocks.0.norm2.weight": one,
        "model.blocks.0.scale_shift_table": one,
        "model.scale_shift_table": one,
        "model.proj_out.weight": one,
        # HY-specific extras (covered by the HY remap layer).
        "model.condition_embedder.action_embedder.linear_1.weight": one,
        "model.condition_embedder.action_embedder.linear_2.bias": one,
        "model.blocks.0.attn1.to_out_prope.0.weight": one,
        "model.blocks.5.attn1.to_out_prope.0.bias": one,
    }
    fsdp_keys = {
        # Mirror the same layout but with FSDP-wrapper prefixes inserted
        # at the block level (matches upstream's generator_ema layout).
        k.replace("model.blocks.0.", "model.blocks.0._fsdp_wrapped_module."): v
        for k, v in base_keys.items()
        if k.startswith("model.blocks.0.")
    }
    fsdp_keys.update({
        k: v for k, v in base_keys.items() if not k.startswith("model.blocks.0.")
    })
    return {"generator": base_keys, "generator_ema": fsdp_keys}


def test_distilled_state_dict_transform_unwraps_envelope() -> None:
    """The transform must pick ``generator`` (not ``generator_ema``) and
    discard the rest of the envelope."""
    raw = _make_synthetic_distilled_state_dict()
    out = hy_worldplay_distilled_state_dict_transform(raw)
    # No top-level dict key from the envelope should leak through.
    assert "generator" not in out
    assert "generator_ema" not in out
    # Nothing should still carry the ``model.`` prefix.
    assert not any(k.startswith("model.") for k in out)


def test_distilled_state_dict_transform_strips_fsdp_prefix() -> None:
    """``_fsdp_wrapped_module.`` must be stripped wherever it appears.

    Even though the transform pins to ``generator`` (which has no
    FSDP prefix in upstream's checkpoint), the strip path is
    defensive -- a future upstream refactor that always saves under
    ``generator`` *with* the wrapper prefix should still load.
    """
    one = torch.ones(())
    raw = {
        "generator": {
            "model.blocks.0._fsdp_wrapped_module.attn1.to_q.weight": one,
            "model.blocks.0._fsdp_wrapped_module.attn1.to_out_prope.0.bias": one,
        },
        "generator_ema": {},
    }
    out = hy_worldplay_distilled_state_dict_transform(raw)
    assert "blocks.0.self_attn.q.weight" in out
    assert "blocks.0.self_attn.o_prope.bias" in out
    # No leftover FSDP prefix.
    assert not any("_fsdp_wrapped_module" in k for k in out)


def test_distilled_state_dict_transform_remaps_base_keys() -> None:
    """Base diffusers -> WanDiTNetwork rewrites must still fire.

    The HY transform layers on top of
    :data:`flashdreams.recipes.wan.config.wan22_ti2v_5b_dit_state_dict_transform`,
    so every base 5B rewrite (text/time embedders, attn projections,
    norm tables, head, FFN) must continue to work.
    """
    raw = _make_synthetic_distilled_state_dict()
    out = hy_worldplay_distilled_state_dict_transform(raw)
    # Spot-check one key per base remap rule that the synthetic dict exercises.
    expected_base = {
        "text_embedding.0.weight",
        "time_embedding.2.bias",
        "blocks.0.self_attn.q.weight",
        "blocks.0.self_attn.norm_k.weight",
        "blocks.0.cross_attn.o.bias",
        "blocks.0.norm3.weight",
        "blocks.0.modulation",
        "head.modulation",
        "head.head.weight",
    }
    missing = expected_base - set(out.keys())
    assert not missing, f"base remap dropped keys: {missing}"


def test_distilled_state_dict_transform_remaps_action_embedder() -> None:
    """``action_embedder.linear_{1,2}`` -> ``action_embedding.{0,2}``.

    The Wan-MLP indexing convention is linear / SiLU / linear with
    SiLU as the parameter-less middle layer, so linear_1 ->
    ``action_embedding.0`` and linear_2 -> ``action_embedding.2``.
    """
    raw = _make_synthetic_distilled_state_dict()
    out = hy_worldplay_distilled_state_dict_transform(raw)
    assert "action_embedding.0.weight" in out
    assert "action_embedding.2.bias" in out
    # And no leftover diffusers-style ``action_embedder`` keys.
    assert not any("action_embedder" in k for k in out)


def test_distilled_state_dict_transform_remaps_to_out_prope() -> None:
    """``blocks.{i}.attn1.to_out_prope.0.{...}`` -> ``blocks.{i}.self_attn.o_prope.{...}``.

    Upstream's ``to_out_prope`` is an ``nn.Sequential`` whose first
    element is the linear; our :class:`HyWorldPlayPRoPESelfAttention`
    stores the same linear directly under ``o_prope``, so the remap
    drops the ``.0.`` middle hop. The rule must apply to *every*
    block index (the regex captures ``\\d+``), not just block 0.
    """
    raw = _make_synthetic_distilled_state_dict()
    out = hy_worldplay_distilled_state_dict_transform(raw)
    assert "blocks.0.self_attn.o_prope.weight" in out
    assert "blocks.5.self_attn.o_prope.bias" in out
    assert not any("to_out_prope" in k for k in out)


def test_distilled_state_dict_transform_idempotent_on_pre_stripped() -> None:
    """The transform must accept a state dict whose envelope was already
    unwrapped (e.g. from a cached intermediate) without error.

    Mirrors upstream's defensive ``if state_dict.get('generator') ...``
    pattern in ``wan/generate.py``: callers should be able to feed
    either the raw envelope or just ``state_dict['generator']``.
    """
    raw = _make_synthetic_distilled_state_dict()
    pre_stripped = raw["generator"]
    out_a = hy_worldplay_distilled_state_dict_transform(raw)
    out_b = hy_worldplay_distilled_state_dict_transform(pre_stripped)
    assert set(out_a.keys()) == set(out_b.keys())
