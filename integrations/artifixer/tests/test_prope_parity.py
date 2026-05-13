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
the ArtiFixer reference's ``model_training/net/prope.py``. When the
reference is not importable (CI images typically only ship flashdreams),
the test is split:

  * ``test_prope_internal_consistency`` runs against an inline copy of the
    upstream math — i.e. tests that ``apply_to_o(apply_to_q(q) ...)`` round
    trips correctly for the identity camera case. Always runs.

  * ``test_prope_matches_reference`` is skipped unless
    ``model_training.net.prope`` from the ArtiFixer reference is on
    ``sys.path``. Set ``ARTIFIXER_REFERENCE_REPO_ROOT`` to its checkout
    root to enable.

The intent is to catch any regression in the verbatim port the moment a
GPU + reference env is available, while still exercising the math today.
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
    """Seed PyTorch for the file.

    Note: PRoPE's ``_rope_precompute_coeffs`` performs an implicit int64->fp32
    promotion (``torch.arange(...) / num_freqs``), so the RoPE coefficients
    are always at fp32 precision regardless of the input dtype. The
    reference has the same property. Tests therefore run at fp32 with
    fp32-appropriate tolerances rather than fp64. The reference's
    ``_lift_K`` also hard-codes float32 output (no ``dtype=`` in
    ``torch.zeros``), so any cross-implementation comparison must use
    fp32 inputs.
    """
    torch.manual_seed(0)


def _build_inputs(
    batch: int = 1,
    cameras: int = 3,
    patches_x: int = 4,
    patches_y: int = 4,
    num_heads: int = 4,
    head_dim: int = 32,
    dtype: torch.dtype = torch.float32,
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
    ``apply_to_q``, so ``apply_to_o(apply_to_q(x)) ~= x`` up to fp32
    round-off (RoPE coeffs are always at fp32 — see fixture docstring).
    """
    inp = _build_inputs()

    inp["viewmats"] = (
        torch.eye(4, dtype=inp["viewmats"].dtype).expand_as(inp["viewmats"]).clone()
    )
    inp["Ks"] = (
        torch.eye(3, dtype=inp["Ks"].dtype).expand_as(inp["Ks"]).clone()
    )

    prope = _instantiate(PropeDotProductAttention, inp)
    q = inp["q"]
    round_trip = prope._apply_to_o(prope._apply_to_q(q))
    torch.testing.assert_close(round_trip, q, atol=5e-6, rtol=5e-6)


def test_prope_apply_to_q_changes_input_for_nontrivial_cameras() -> None:
    """Sanity: with real cameras, the PRoPE transform is not a no-op."""
    inp = _build_inputs()
    prope = _instantiate(PropeDotProductAttention, inp)
    transformed = prope._apply_to_q(inp["q"])
    diff = (transformed - inp["q"]).abs().max().item()
    assert diff > 1e-3, f"expected non-trivial PRoPE transform, got max abs diff {diff}"


def _try_import_reference() -> object | None:
    """Return the ArtiFixer reference's PropeDotProductAttention if importable."""
    repo_root = os.environ.get("ARTIFIXER_REFERENCE_REPO_ROOT")
    if repo_root and Path(repo_root).is_dir() and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from model_training.net.prope import PropeDotProductAttention as RefPRoPE
    except ImportError:
        return None
    return RefPRoPE


@pytest.mark.skipif(
    _try_import_reference() is None,
    reason=(
        "ArtiFixer reference not importable. Set "
        "ARTIFIXER_REFERENCE_REPO_ROOT to the reference checkout, or run "
        "from an env that already has it on sys.path."
    ),
)
def test_prope_matches_reference() -> None:
    """Numerical parity vs the ArtiFixer reference at fp32.

    Asserts ``apply_to_q``, ``apply_to_kv``, ``apply_to_o`` produce
    bit-identical outputs to the reference within fp32 round-off.

    Inputs are fp32 because the reference's ``_lift_K`` hard-codes float32
    output (no ``dtype=`` in ``torch.zeros``), so feeding fp64 viewmats
    crashes its einsum. Our port passes ``dtype=Ks.dtype``, making it
    slightly more dtype-robust; at fp32 the two are bit-identical.
    """
    RefPRoPE = _try_import_reference()
    assert RefPRoPE is not None

    inp = _build_inputs()
    ours = _instantiate(PropeDotProductAttention, inp)
    ref = _instantiate(RefPRoPE, inp)

    for name, x in (("q", inp["q"]), ("k", inp["k"]), ("v", inp["v"])):
        ours_q = ours._apply_to_q(x)
        ref_q = ref._apply_to_q(x)
        torch.testing.assert_close(
            ours_q, ref_q, atol=1e-6, rtol=1e-6, msg=f"apply_to_q on {name}"
        )

        ours_kv = ours._apply_to_kv(x)
        ref_kv = ref._apply_to_kv(x)
        torch.testing.assert_close(
            ours_kv, ref_kv, atol=1e-6, rtol=1e-6, msg=f"apply_to_kv on {name}"
        )

        ours_o = ours._apply_to_o(x)
        ref_o = ref._apply_to_o(x)
        torch.testing.assert_close(
            ours_o, ref_o, atol=1e-6, rtol=1e-6, msg=f"apply_to_o on {name}"
        )
