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

"""CPU-only unit tests for the PRoPE projective positional encoding port.

Verifies :mod:`hy_worldplay._prope` against a numpy reference
that re-implements the same per-camera 4x4 block-diagonal projection
without the torch.einsum / partial-function plumbing. Three groups:

* Tensor-shape contracts: assertions that catch the most common misuse
  (mismatched batch / camera dims, non-multiple-of-4 head_dim).
* Math parity: ``prope_qkv`` matches a numpy reference for the
  ``Ks=None`` (GTA) and ``Ks=...`` (intrinsic) branches; the
  ``apply_fn_o`` returned alongside Q/K/V undoes a round trip on the
  output position.
* Identity round-trip: when ``viewmats`` is a stack of identity matrices
  and ``Ks`` is ``None``, every transform reduces to the identity so
  Q/K/V/O pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.ci_cpu


def _make_random_viewmats(batch: int, cameras: int, *, seed: int = 0) -> torch.Tensor:
    """Build a stack of plausible-looking ``SE(3)`` extrinsics for tests."""
    rng = np.random.default_rng(seed)
    out = np.zeros((batch, cameras, 4, 4), dtype=np.float64)
    for b in range(batch):
        for c in range(cameras):
            # Random small-angle rotation + small translation; keeps the
            # numerical magnitudes comparable to upstream's per-frame
            # viewmats so we exercise the einsum carefully.
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            angle = float(rng.uniform(-0.5, 0.5))
            K = np.array(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ]
            )
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            t = rng.normal(size=3) * 0.1
            out[b, c, :3, :3] = R
            out[b, c, :3, 3] = t
            out[b, c, 3, 3] = 1.0
    return torch.from_numpy(out).to(torch.float64)


def _make_random_intrinsics(batch: int, cameras: int, *, seed: int = 1) -> torch.Tensor:
    """Build a stack of plausible camera intrinsics."""
    rng = np.random.default_rng(seed)
    out = np.zeros((batch, cameras, 3, 3), dtype=np.float64)
    for b in range(batch):
        for c in range(cameras):
            fx = 1.0 + float(rng.uniform(-0.1, 0.1))
            fy = 1.0 + float(rng.uniform(-0.1, 0.1))
            cx = 0.5 + float(rng.uniform(-0.05, 0.05))
            cy = 0.5 + float(rng.uniform(-0.05, 0.05))
            out[b, c] = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    return torch.from_numpy(out).to(torch.float64)


def _numpy_prope_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    viewmats: np.ndarray,
    Ks: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference re-implementation in numpy used to cross-check the torch port.

    Returns the four transformed tensors (Q, K, V, P_for_output) so the
    caller can compare ``apply_fn_o(O)`` against ``einsum(P, O)``.
    """
    batch, num_heads, seqlen, head_dim = q.shape
    cameras = viewmats.shape[1]
    assert head_dim % 4 == 0

    if Ks is not None:
        Ks_norm = np.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 2, 2] = 1.0

        lift = np.zeros(Ks_norm.shape[:-2] + (4, 4))
        lift[..., :3, :3] = Ks_norm
        lift[..., 3, 3] = 1.0
        P = np.einsum("...ij,...jk->...ik", lift, viewmats)

        Kinv = np.zeros_like(Ks_norm)
        Kinv[..., 0, 0] = 1.0 / Ks_norm[..., 0, 0]
        Kinv[..., 1, 1] = 1.0 / Ks_norm[..., 1, 1]
        Kinv[..., 0, 2] = -Ks_norm[..., 0, 2] / Ks_norm[..., 0, 0]
        Kinv[..., 1, 2] = -Ks_norm[..., 1, 2] / Ks_norm[..., 1, 1]
        Kinv[..., 2, 2] = 1.0
        lift_inv = np.zeros(Kinv.shape[:-2] + (4, 4))
        lift_inv[..., :3, :3] = Kinv
        lift_inv[..., 3, 3] = 1.0

        Rinv = viewmats[..., :3, :3].transpose(0, 1, 3, 2)
        SE3_inv = np.zeros_like(viewmats)
        SE3_inv[..., :3, :3] = Rinv
        SE3_inv[..., :3, 3] = -np.einsum("...ij,...j->...i", Rinv, viewmats[..., :3, 3])
        SE3_inv[..., 3, 3] = 1.0
        P_inv = np.einsum("...ij,...jk->...ik", SE3_inv, lift_inv)
    else:
        P = viewmats
        Rinv = viewmats[..., :3, :3].transpose(0, 1, 3, 2)
        P_inv = np.zeros_like(viewmats)
        P_inv[..., :3, :3] = Rinv
        P_inv[..., :3, 3] = -np.einsum("...ij,...j->...i", Rinv, viewmats[..., :3, 3])
        P_inv[..., 3, 3] = 1.0

    P_T = P.transpose(0, 1, 3, 2)

    def apply(feats: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        reshaped = feats.reshape(batch, num_heads, cameras, -1, head_dim // 4, 4)
        out = np.einsum("bcij,bncpkj->bncpki", matrix, reshaped)
        return out.reshape(feats.shape)

    return (
        apply(q, P_T),
        apply(k, P_inv),
        apply(v, P_inv),
        P,
    )


def test_prope_qkv_matches_numpy_reference_with_intrinsics() -> None:
    """``prope_qkv`` Q/K/V/output transforms match the numpy reference."""
    from hy_worldplay._prope import prope_qkv

    batch, num_heads, cameras, patches_per_cam, head_dim = 2, 3, 4, 5, 8
    seqlen = cameras * patches_per_cam

    torch.manual_seed(0)
    q = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    viewmats = _make_random_viewmats(batch, cameras)
    Ks = _make_random_intrinsics(batch, cameras)

    q_p, k_p, v_p, apply_fn_o = prope_qkv(q, k, v, viewmats=viewmats, Ks=Ks)

    q_ref, k_ref, v_ref, P_ref = _numpy_prope_reference(
        q.numpy(), k.numpy(), v.numpy(), viewmats.numpy(), Ks.numpy()
    )
    torch.testing.assert_close(q_p, torch.from_numpy(q_ref), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(k_p, torch.from_numpy(k_ref), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(v_p, torch.from_numpy(v_ref), atol=1e-10, rtol=1e-10)

    # ``apply_fn_o`` is the output-side transform: feed an arbitrary
    # ``[B, H, L, D]`` tensor (any of Q/K/V works) and assert it matches
    # the reference einsum with P.
    o_in = torch.randn_like(q)
    o_out = apply_fn_o(o_in)
    o_ref = np.einsum(
        "bcij,bncpkj->bncpki",
        P_ref,
        o_in.numpy().reshape(batch, num_heads, cameras, -1, head_dim // 4, 4),
    ).reshape(o_in.shape)
    torch.testing.assert_close(o_out, torch.from_numpy(o_ref), atol=1e-10, rtol=1e-10)


def test_prope_qkv_matches_numpy_reference_no_intrinsics() -> None:
    """The ``Ks=None`` (GTA) branch matches the numpy reference."""
    from hy_worldplay._prope import prope_qkv

    batch, num_heads, cameras, patches_per_cam, head_dim = 2, 2, 3, 4, 8
    seqlen = cameras * patches_per_cam

    torch.manual_seed(7)
    q = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    viewmats = _make_random_viewmats(batch, cameras, seed=11)

    q_p, k_p, v_p, _ = prope_qkv(q, k, v, viewmats=viewmats, Ks=None)
    q_ref, k_ref, v_ref, _ = _numpy_prope_reference(
        q.numpy(), k.numpy(), v.numpy(), viewmats.numpy(), Ks=None
    )
    torch.testing.assert_close(q_p, torch.from_numpy(q_ref), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(k_p, torch.from_numpy(k_ref), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(v_p, torch.from_numpy(v_ref), atol=1e-10, rtol=1e-10)


def test_prope_qkv_identity_viewmats_is_pass_through() -> None:
    """Identity extrinsics + no intrinsics is a strict no-op on Q/K/V/O."""
    from hy_worldplay._prope import prope_qkv

    batch, num_heads, cameras, patches_per_cam, head_dim = 1, 2, 3, 2, 8
    seqlen = cameras * patches_per_cam
    torch.manual_seed(13)
    q = torch.randn(batch, num_heads, seqlen, head_dim, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    viewmats = (
        torch.eye(4, dtype=torch.float64).expand(batch, cameras, 4, 4).contiguous()
    )

    q_p, k_p, v_p, apply_fn_o = prope_qkv(q, k, v, viewmats=viewmats, Ks=None)
    torch.testing.assert_close(q_p, q, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(k_p, k, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(v_p, v, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(apply_fn_o(q), q, atol=1e-12, rtol=1e-12)


def test_prope_qkv_rejects_head_dim_not_divisible_by_4() -> None:
    """The 4x4 block-diagonal tiling requires head_dim % 4 == 0."""
    from hy_worldplay._prope import prope_qkv

    batch, num_heads, cameras, patches_per_cam = 1, 1, 2, 2
    seqlen = cameras * patches_per_cam
    bad_head_dim = 6  # not a multiple of 4

    q = torch.zeros(batch, num_heads, seqlen, bad_head_dim, dtype=torch.float64)
    viewmats = _make_random_viewmats(batch, cameras)

    with pytest.raises(AssertionError, match="multiple of 4"):
        prope_qkv(q, q, q, viewmats=viewmats, Ks=None)


def test_prope_qkv_rejects_seqlen_not_divisible_by_cameras() -> None:
    """Token layout requires seqlen divisible by the camera count."""
    from hy_worldplay._prope import prope_qkv

    batch, num_heads, cameras, head_dim = 1, 1, 3, 8
    seqlen = cameras * 2 + 1  # off by one

    q = torch.zeros(batch, num_heads, seqlen, head_dim, dtype=torch.float64)
    viewmats = _make_random_viewmats(batch, cameras)

    with pytest.raises(AssertionError, match="divisible by"):
        prope_qkv(q, q, q, viewmats=viewmats, Ks=None)


def test_build_prope_apply_fns_can_be_called_repeatedly() -> None:
    """The split entry point lets callers cache transforms across layers."""
    from hy_worldplay._prope import build_prope_apply_fns

    batch, cameras, head_dim = 1, 2, 8
    viewmats = _make_random_viewmats(batch, cameras)
    apply_q, apply_kv, apply_o = build_prope_apply_fns(
        head_dim=head_dim, viewmats=viewmats, Ks=None
    )
    feats = torch.randn(batch, 2, cameras * 3, head_dim, dtype=torch.float64)
    # Idempotent in the sense that calling twice yields the same answer
    # (these are pure functions of inputs).
    torch.testing.assert_close(apply_q(feats), apply_q(feats), atol=0, rtol=0)
    torch.testing.assert_close(apply_kv(feats), apply_kv(feats), atol=0, rtol=0)
    torch.testing.assert_close(apply_o(feats), apply_o(feats), atol=0, rtol=0)
