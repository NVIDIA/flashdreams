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

"""Numerical parity test for the ported PRoPE attention module.

Compares :class:`artifixer.network.prope.PropeDotProductAttention` against
the dreamfix reference at ``model_training/net/prope.py``. When dreamfix is
not importable (the CI image only ships flashdreams), the test is split:

  * ``test_prope_internal_consistency`` runs against an inline copy of the
    upstream math — i.e. tests that ``apply_to_o(apply_to_q(q) ...)`` round
    trips correctly for the identity camera case. Always runs.

  * ``test_prope_matches_dreamfix_reference`` is skipped unless
    ``model_training.net.prope`` can be imported (set
    ``DREAMFIX_REPO_ROOT`` to its repo root to enable).

The intent is to catch any regression in the verbatim port the moment a
GPU + dreamfix env is available, while still exercising the math today.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from artifixer.network.prope import PropeDotProductAttention


@pytest.fixture(autouse=True)
def _deterministic_torch() -> None:
    """Use fp64 by default in this file: PRoPE math is bit-exact at fp64."""
    torch.manual_seed(0)


def _build_inputs(
    batch: int = 1,
    cameras: int = 3,
    patches_x: int = 4,
    patches_y: int = 4,
    num_heads: int = 4,
    head_dim: int = 32,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build a deterministic fixture for PRoPE parity tests."""
    seqlen = cameras * patches_x * patches_y

    # Random SE(3) viewmats: rotation = QR(randn), translation = randn.
    R = torch.linalg.qr(torch.randn(batch, cameras, 3, 3, dtype=dtype, device=device)).Q
    t = torch.randn(batch, cameras, 3, dtype=dtype, device=device) * 0.5
    viewmats = torch.zeros(batch, cameras, 4, 4, dtype=dtype, device=device)
    viewmats[..., :3, :3] = R
    viewmats[..., :3, 3] = t
    viewmats[..., 3, 3] = 1.0

    # Intrinsics in NDC-ish form: fx, fy ~ 1.0; cx, cy ~ 0.5.
    Ks = torch.eye(3, dtype=dtype, device=device).expand(batch, cameras, 3, 3).clone()
    Ks[..., 0, 0] = 1.0 + 0.1 * torch.randn(batch, cameras, dtype=dtype, device=device)
    Ks[..., 1, 1] = 1.0 + 0.1 * torch.randn(batch, cameras, dtype=dtype, device=device)
    Ks[..., 0, 2] = 0.5
    Ks[..., 1, 2] = 0.5

    q = torch.randn(batch, num_heads, seqlen, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, num_heads, seqlen, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, num_heads, seqlen, head_dim, dtype=dtype, device=device)

    return {
        "viewmats": viewmats,
        "Ks": Ks,
        "q": q,
        "k": k,
        "v": v,
        "patches_x": patches_x,
        "patches_y": patches_y,
        "head_dim": head_dim,
    }


def _instantiate(prope_cls, inp: dict) -> object:
    """Build a PRoPE instance and precompute the apply-fns for ``inp``."""
    prope = prope_cls(head_dim=inp["head_dim"])
    prope.update_coeffs(inp["patches_x"], inp["patches_y"], device="cpu")
    prope._precompute_and_cache_apply_fns(inp["viewmats"], inp["Ks"])
    return prope


def test_prope_internal_consistency() -> None:
    """``apply_to_o`` undoes ``apply_to_q`` on identity cameras.

    With viewmats = identity and Ks = identity, ``P = P_T = P_inv = I``
    and the projection-matrix legs of the block-diagonal transform become
    no-ops. The RoPE legs of ``apply_to_o`` are the *inverse* of those in
    ``apply_to_q``, so ``apply_to_o(apply_to_q(x)) == x`` exactly.
    """
    inp = _build_inputs()

    # Replace viewmats / Ks with identity so the projmat legs cancel.
    inp["viewmats"] = (
        torch.eye(4, dtype=inp["viewmats"].dtype).expand_as(inp["viewmats"]).clone()
    )
    inp["Ks"] = (
        torch.eye(3, dtype=inp["Ks"].dtype).expand_as(inp["Ks"]).clone()
    )

    prope = _instantiate(PropeDotProductAttention, inp)
    q = inp["q"]
    round_trip = prope._apply_to_o(prope._apply_to_q(q))
    torch.testing.assert_close(round_trip, q, atol=1e-10, rtol=1e-10)


def test_prope_apply_to_q_changes_input_for_nontrivial_cameras() -> None:
    """Sanity: with real cameras, the PRoPE transform is not a no-op."""
    inp = _build_inputs()
    prope = _instantiate(PropeDotProductAttention, inp)
    transformed = prope._apply_to_q(inp["q"])
    diff = (transformed - inp["q"]).abs().max().item()
    assert diff > 1e-3, f"expected non-trivial PRoPE transform, got max abs diff {diff}"


def _try_import_dreamfix_reference() -> object | None:
    """Return dreamfix's PropeDotProductAttention if importable."""
    repo_root = os.environ.get("DREAMFIX_REPO_ROOT")
    if repo_root and Path(repo_root).is_dir() and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from model_training.net.prope import PropeDotProductAttention as RefPRoPE
    except ImportError:
        return None
    return RefPRoPE


@pytest.mark.skipif(
    _try_import_dreamfix_reference() is None,
    reason=(
        "dreamfix not importable. Set DREAMFIX_REPO_ROOT to the dreamfix "
        "checkout, or run from an env that already has it on sys.path."
    ),
)
def test_prope_matches_dreamfix_reference() -> None:
    """Numerical parity vs dreamfix at fp64.

    Asserts ``apply_to_q``, ``apply_to_kv``, ``apply_to_o`` produce
    bit-identical outputs to the reference within ``atol=1e-10``
    (fp64 round-off floor).
    """
    RefPRoPE = _try_import_dreamfix_reference()
    assert RefPRoPE is not None

    inp = _build_inputs()
    ours = _instantiate(PropeDotProductAttention, inp)
    ref = _instantiate(RefPRoPE, inp)

    for name, x in (("q", inp["q"]), ("k", inp["k"]), ("v", inp["v"])):
        ours_q = ours._apply_to_q(x)
        ref_q = ref._apply_to_q(x)
        torch.testing.assert_close(
            ours_q, ref_q, atol=1e-10, rtol=1e-10, msg=f"apply_to_q on {name}"
        )

        ours_kv = ours._apply_to_kv(x)
        ref_kv = ref._apply_to_kv(x)
        torch.testing.assert_close(
            ours_kv, ref_kv, atol=1e-10, rtol=1e-10, msg=f"apply_to_kv on {name}"
        )

        ours_o = ours._apply_to_o(x)
        ref_o = ref._apply_to_o(x)
        torch.testing.assert_close(
            ours_o, ref_o, atol=1e-10, rtol=1e-10, msg=f"apply_to_o on {name}"
        )
