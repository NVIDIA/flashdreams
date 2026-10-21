# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D mRoPE parity + position-ID layout tests (GPU-free).

Validates our interleaved-mRoPE cos/sin against ``transformers`` Qwen3-VL, and
the text / vision position-ID builders against the documented layout.
"""

from __future__ import annotations

import pytest
import torch
from cosmos_interactive._rope import mrope_cos_sin, text_mrope_ids, vae_mrope_ids

pytestmark = pytest.mark.ci_cpu


def test_text_ids_share_all_axes() -> None:
    ids, nxt = text_mrope_ids(4, 0.0)
    assert ids.shape == (3, 4)
    assert torch.equal(ids[0], ids[1]) and torch.equal(ids[1], ids[2])
    assert ids[0].tolist() == [0, 1, 2, 3]
    assert nxt == 4.0


def test_vae_ids_integer_grid_layout() -> None:
    # 2 frames x 2 x 2, integer positions, spatial reset, offset 5.
    ids, _ = vae_mrope_ids(2, 2, 2, 5.0, fps=None)
    assert ids.shape == (3, 8)
    # T-major: frame0 (4 tokens) then frame1 (4 tokens); temporal = offset + frame.
    assert ids[0].tolist() == [5, 5, 5, 5, 6, 6, 6, 6]
    # Height cycles 0,0,1,1 per frame; width cycles 0,1,0,1 per frame.
    assert ids[1].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert ids[2].tolist() == [0, 1, 0, 1, 0, 1, 0, 1]


def test_mrope_cos_sin_matches_hf_qwen3_vl() -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    head_dim = 16
    mrope_section = (4, 2, 2)  # sums to head_dim/2 = 8
    theta = 1_000_000.0

    cfg = Qwen3VLTextConfig.from_dict(
        dict(
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=head_dim,
            num_hidden_layers=1,
            intermediate_size=128,
            rope_theta=theta,
            rope_scaling={"rope_type": "default", "mrope_section": list(mrope_section)},
            max_position_embeddings=128,
        )
    )
    hf_rotary = Qwen3VLTextRotaryEmbedding(cfg)

    torch.manual_seed(0)
    s = 7
    # Mixed 3D positions (distinct t/h/w) to exercise the interleave, not just text.
    position_ids = torch.stack(
        [
            torch.arange(s, dtype=torch.float32),
            torch.randint(0, 5, (s,)).float(),
            torch.randint(0, 5, (s,)).float(),
        ],
        dim=0,
    )

    dummy = torch.zeros(1, s, cfg.hidden_size)
    # HF Qwen3-VL rotary expects position_ids shaped [3, batch, S].
    hf_cos, hf_sin = hf_rotary(dummy, position_ids[:, None, :])
    mine_cos, mine_sin = mrope_cos_sin(position_ids, head_dim, theta, mrope_section)

    cos_err = (mine_cos - hf_cos[0]).abs().max().item()
    sin_err = (mine_sin - hf_sin[0]).abs().max().item()
    assert cos_err < 1e-5 and sin_err < 1e-5, (cos_err, sin_err)
