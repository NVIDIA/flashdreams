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

"""CPU-only unit tests for the HY-WorldPlay memory selector (phase 2b.5a).

Covers four surfaces:

* The selection algorithm in :mod:`hy_worldplay._memory` -- output
  shape, sortedness, deduplication, recent-frames invariant, and the
  upstream-mirroring assertion when the budget cannot be filled.
* The :class:`HyWorldPlayCtrl.memory_frame_indices` round-trip through
  the transformer's patchify pass (the same risk profile as the 2b.3
  ``action`` and 2b.4 ``viewmats`` round-trips).
* :meth:`HyWorldPlayWanCtrlEncoder.set_memory_config` /
  :meth:`clear_memory_config` shape validation and per-AR-step gating
  (selection only runs once enough history has accumulated).
* Runner config plumbing: ``use_memory_selection`` defaults off,
  composes with the action + camera swaps, and refuses to arm without
  the camera-conditioning precondition.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.ci_cpu


## ---------------------------------------------------------------------------
## Test helpers
## ---------------------------------------------------------------------------


def _identity_w2c(n: int) -> np.ndarray:
    """Return a stack of ``n`` identity 4x4 W2C matrices.

    All zero translation + identity rotation -- every camera sees the
    same FOV cone, so every historical clip has overlap 1.0 with the
    query clip. Distances tie at 0.0; the selector breaks the tie by
    iteration order (which is what we exercise in the recent-frames
    invariant test below).
    """
    return np.tile(np.eye(4, dtype=np.float64)[None, :, :], (n, 1, 1))


def _small_points_cloud(seed: int = 0) -> torch.Tensor:
    """A cheap 256-point sphere cloud for CPU tests.

    The selection algorithm's correctness doesn't depend on the cloud
    size (only the FOV-overlap estimate's variance does), so we trade
    upstream's 50k-point default for something that runs in
    milliseconds.
    """
    gen = torch.Generator().manual_seed(seed)
    from hy_worldplay._memory import generate_points_in_sphere

    return generate_points_in_sphere(256, radius=8.0, generator=gen)


## ---------------------------------------------------------------------------
## Selection algorithm
## ---------------------------------------------------------------------------


def test_select_memory_frame_indices_invariants() -> None:
    """Output is sorted, unique, sized to ``memory_frames``, and contains the recent window."""
    from hy_worldplay._memory import select_memory_frame_indices

    n = 32  # enough to fill 16-frame budget at current_frame_idx >= 16
    indices = select_memory_frame_indices(
        _identity_w2c(n),
        current_frame_idx=16,
        points_local=_small_points_cloud(),
        memory_frames=16,
        temporal_context_size=12,
        pred_latent_size=4,
    )
    assert indices == sorted(indices), "indices must be sorted"
    assert len(indices) == len(set(indices)), "indices must be unique"
    assert len(indices) == 16, "len must match memory_frames"
    # Temporal context window [current_frame_idx - 12, current_frame_idx)
    # is kept unconditionally; assert that explicitly.
    for i in range(4, 16):
        assert i in indices, (
            f"temporal context frame {i} must be in selected indices; "
            f"got {indices}"
        )


def test_select_memory_frame_indices_min_current_frame_idx_bound() -> None:
    """``current_frame_idx < 3`` raises (mirrors upstream's bounds check)."""
    from hy_worldplay._memory import select_memory_frame_indices

    with pytest.raises(ValueError, match="current_frame_idx must lie in"):
        select_memory_frame_indices(
            _identity_w2c(32),
            current_frame_idx=2,
            points_local=_small_points_cloud(),
            memory_frames=16,
            temporal_context_size=12,
            pred_latent_size=4,
        )


def test_select_memory_frame_indices_oob_current_frame_idx_raises() -> None:
    """``current_frame_idx >= num_total_frames`` raises."""
    from hy_worldplay._memory import select_memory_frame_indices

    with pytest.raises(ValueError, match="current_frame_idx must lie in"):
        select_memory_frame_indices(
            _identity_w2c(8),
            current_frame_idx=8,
            points_local=_small_points_cloud(),
            memory_frames=16,
            temporal_context_size=12,
            pred_latent_size=4,
        )


def test_select_memory_frame_indices_budget_underfill_raises() -> None:
    """Not enough history to fill the budget should surface the
    upstream-mirroring length assertion as a clear ``AssertionError``.

    With ``current_frame_idx=14, temporal_context_size=12`` the
    historical-clip window is ``range(0, 2, 4)`` -- empty. The
    selector can only return the 12 temporal-context frames, but the
    budget asks for 16, so the assertion fires. This is the same
    failure mode upstream has; the test pins the failure as
    intentional so a future refactor doesn't silently swallow it.
    """
    from hy_worldplay._memory import select_memory_frame_indices

    with pytest.raises(AssertionError, match="memory selection produced"):
        select_memory_frame_indices(
            _identity_w2c(32),
            current_frame_idx=14,
            points_local=_small_points_cloud(),
            memory_frames=16,
            temporal_context_size=12,
            pred_latent_size=4,
        )


def test_select_memory_frame_indices_rejects_budget_inversion() -> None:
    """``memory_frames < temporal_context_size`` is a config error."""
    from hy_worldplay._memory import select_memory_frame_indices

    with pytest.raises(ValueError, match="memory_frames"):
        select_memory_frame_indices(
            _identity_w2c(32),
            current_frame_idx=16,
            points_local=_small_points_cloud(),
            memory_frames=8,
            temporal_context_size=12,
            pred_latent_size=4,
        )


def test_calculate_fov_overlap_identity_is_one() -> None:
    """Two identical W2C poses overlap perfectly (similarity == 1.0).

    With identity rotation + zero translation both cameras look down +Z
    from the origin, see the entire forward hemisphere, and the same
    sphere of points falls in both frustums. The historical distance
    gate (8.0 units) doesn't kick in because the cameras share the same
    origin.
    """
    from hy_worldplay._memory import calculate_fov_overlap_similarity

    eye = np.eye(4, dtype=np.float64)
    points = _small_points_cloud()
    sim = calculate_fov_overlap_similarity(eye, eye, points_local=points)
    assert sim == pytest.approx(1.0)


def test_generate_points_in_sphere_lies_within_radius() -> None:
    """All sampled points must be within the requested radius."""
    from hy_worldplay._memory import generate_points_in_sphere

    pts = generate_points_in_sphere(
        128, radius=2.5, generator=torch.Generator().manual_seed(7)
    )
    radii = torch.linalg.norm(pts, dim=-1)
    assert (radii <= 2.5 + 1e-6).all(), (
        f"some points fell outside the requested sphere "
        f"(max radius = {radii.max().item():.6f})"
    )


def test_generate_points_in_sphere_shape() -> None:
    """Returned tensor is ``[n_points, 3]``."""
    from hy_worldplay._memory import generate_points_in_sphere

    pts = generate_points_in_sphere(64, radius=1.0)
    assert pts.shape == (64, 3)


## ---------------------------------------------------------------------------
## HyWorldPlayCtrl round-trip
## ---------------------------------------------------------------------------


def test_hyworldplay_ctrl_memory_field_defaults_to_none() -> None:
    """The 2b.5a ctrl field defaults ``None`` so non-memory callers stay opt-in."""
    from hy_worldplay._action import HyWorldPlayCtrl

    ctrl = HyWorldPlayCtrl(latent=torch.zeros(1, 1, 1, 1, 1), mask=None)
    assert ctrl.memory_frame_indices is None


def test_hyworldplay_transformer_patchify_preserves_memory_indices() -> None:
    """``memory_frame_indices`` must survive the patchify rebuild.

    Same regression target as the 2b.3 action / 2b.4 camera round-trip
    tests: the base transformer rebuilds the ctrl via
    ``I2VCtrl(latent=..., mask=...)`` after patchify, which would drop
    subclass fields unless the subclass overrides the rebuild. We
    exercise the ``_is_patchified=True`` short-circuit (same shape as
    ``test_camera.py::test_hyworldplay_transformer_patchify_preserves_camera_fields``)
    so we don't have to spin up the parent ``super()`` chain just to
    pass the leaf tensors through.
    """
    from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21Transformer

    fake_self = type("F", (), {})()
    fake_self.patchify_and_maybe_split_cp = (
        HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp.__get__(fake_self)
    )

    latent = torch.randn(1, 4, 16, 8, 8)
    mask = torch.zeros(1, 4, 16, 8, 8)
    memory_indices = [0, 1, 2, 3, 12, 13, 14, 15]
    ctrl_patched = HyWorldPlayCtrl(
        latent=latent,
        mask=mask,
        memory_frame_indices=memory_indices,
        _is_patchified=True,
    )
    out = fake_self.patchify_and_maybe_split_cp(ctrl_patched)
    assert out is ctrl_patched
    assert out.memory_frame_indices is memory_indices


## ---------------------------------------------------------------------------
## Encoder plumbing
## ---------------------------------------------------------------------------


def _make_memory_encoder():
    """Build a HY-WorldPlay encoder skeleton without touching ``__init__``.

    Mirrors ``test_camera.py::_make_camera_encoder`` -- the real
    :class:`I2VCtrlEncoder` constructor pulls in the VAE encoder which
    drags diffusers along. We only need the bookkeeping fields the
    memory setter / gate look at.
    """
    from hy_worldplay._action import HyWorldPlayWanCtrlEncoder

    encoder = HyWorldPlayWanCtrlEncoder.__new__(HyWorldPlayWanCtrlEncoder)
    encoder._action_labels = None
    encoder._viewmats = None
    encoder._intrinsics = None
    encoder._memory_config = None
    return encoder


def test_encoder_set_memory_config_validates_shapes() -> None:
    """``set_memory_config`` rejects malformed ``points_local`` up front."""
    encoder = _make_memory_encoder()

    with pytest.raises(ValueError, match="points_local"):
        encoder.set_memory_config(
            points_local=torch.zeros(64),  # missing the 3D coordinate axis
            context_window_length=16,
            memory_frames=16,
            temporal_context_size=12,
            pred_latent_size=4,
            fov_h_deg=60.0,
            fov_v_deg=35.0,
        )


def test_encoder_set_memory_config_rejects_budget_inversion() -> None:
    """``memory_frames < temporal_context_size`` is a config error."""
    encoder = _make_memory_encoder()

    with pytest.raises(ValueError, match="memory_frames"):
        encoder.set_memory_config(
            points_local=_small_points_cloud(),
            context_window_length=16,
            memory_frames=8,
            temporal_context_size=12,
            pred_latent_size=4,
            fov_h_deg=60.0,
            fov_v_deg=35.0,
        )


def test_encoder_clear_memory_config_drops_state() -> None:
    """``clear_memory_config`` disarms selection."""
    encoder = _make_memory_encoder()
    encoder.set_memory_config(
        points_local=_small_points_cloud(),
        context_window_length=16,
        memory_frames=16,
        temporal_context_size=12,
        pred_latent_size=4,
        fov_h_deg=60.0,
        fov_v_deg=35.0,
    )
    assert encoder._memory_config is not None
    encoder.clear_memory_config()
    assert encoder._memory_config is None


def test_encoder_compute_memory_indices_gates_on_history() -> None:
    """FOV selection only kicks in once ``current_frame_idx >= context_window_length``.

    Below that threshold the encoder returns the all-history list
    ``list(range(0, current_frame_idx))`` -- mirroring vendor's
    ``elif use_memory:`` branch in
    ``pipeline_wan_w_mem_relative_rope.py`` line 868-869. The HY
    native path *requires* this fall-back: it overrides
    ``finalize_kv_cache`` to skip the base rolling-KV update and
    resets the rolling cache at every chunk boundary, so without an
    explicit prefill chunk-1+ would attend to nothing from previous
    chunks. AR step 0 is the only case that returns ``None`` (no
    history yet).
    """
    encoder = _make_memory_encoder()
    encoder._viewmats = torch.from_numpy(_identity_w2c(32)).unsqueeze(0)
    encoder.set_memory_config(
        points_local=_small_points_cloud(),
        context_window_length=16,
        memory_frames=16,
        temporal_context_size=12,
        pred_latent_size=4,
        fov_h_deg=60.0,
        fov_v_deg=35.0,
    )

    # AR step 0 / no history -> None
    assert (
        encoder._compute_memory_indices(
            autoregressive_index=0, current_frame_idx=0
        )
        is None
    )

    # Below FOV-selection threshold but past chunk 0 -> all-history
    # fall-back (matches vendor's ``elif use_memory:`` branch).
    indices_below = encoder._compute_memory_indices(
        autoregressive_index=1, current_frame_idx=4
    )
    assert indices_below == [0, 1, 2, 3]

    # Above threshold -> FOV-selected list of length ``memory_frames``.
    indices = encoder._compute_memory_indices(
        autoregressive_index=4, current_frame_idx=16
    )
    assert isinstance(indices, list)
    assert len(indices) == 16
    assert indices == sorted(indices)


def test_encoder_compute_memory_indices_disabled_uses_all_history() -> None:
    """Without ``set_memory_config`` the encoder still emits all-history indices.

    The HY native path relies on the prefill executor for *all*
    cross-chunk attention (see the gating-test docstring above),
    so even when FOV-based selection is disarmed we have to emit
    indices for chunk-1+. The executor consumes the bound
    ``rollout_viewmats`` directly; it doesn't need the FOV
    ``_memory_config`` to run. AR step 0 still returns ``None``.
    """
    encoder = _make_memory_encoder()
    encoder._viewmats = torch.from_numpy(_identity_w2c(32)).unsqueeze(0)
    # No set_memory_config() call -> FOV selection is off, but the
    # all-history fall-back still emits indices for chunk-1+.

    assert (
        encoder._compute_memory_indices(
            autoregressive_index=0, current_frame_idx=0
        )
        is None
    )
    indices = encoder._compute_memory_indices(
        autoregressive_index=4, current_frame_idx=16
    )
    assert indices == list(range(0, 16))


def test_encoder_compute_memory_indices_no_camera_returns_none() -> None:
    """No bound viewmats => no prefill possible => ``None``.

    The prefill executor indexes ``rollout_viewmats`` (and friends)
    at the selected indices, so without camera data bound the
    executor can't run. In that configuration the dual-branch /
    action paths are themselves no-ops, so the missing prefill is
    also a no-op.
    """
    encoder = _make_memory_encoder()
    # No set_camera_data() call -> _viewmats is None.
    assert encoder._viewmats is None

    assert (
        encoder._compute_memory_indices(
            autoregressive_index=4, current_frame_idx=16
        )
        is None
    )


## ---------------------------------------------------------------------------
## Runner config wiring
## ---------------------------------------------------------------------------


def test_use_memory_selection_off_by_default() -> None:
    """Phase 2b.5a feature flag defaults off."""
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(runner_name="hy-worldplay-wan-i2v-5b")
    assert cfg.use_memory_selection is False


def test_use_memory_selection_requires_camera_conditioning() -> None:
    """Memory selection needs the per-rollout viewmats binding.

    Without ``use_camera_conditioning=True`` the encoder has nothing to
    score historical clips against, so ``__post_init__`` rejects the
    combo with a clear error instead of silently emitting
    ``memory_frame_indices=None`` on every AR step.
    """
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

    with pytest.raises(ValueError, match="use_memory_selection"):
        HyWorldPlayWanI2VRunnerConfig(
            runner_name="hy-worldplay-wan-i2v-5b",
            use_native_pipeline=True,
            use_memory_selection=True,
        )


def test_use_memory_selection_composes_with_camera_conditioning() -> None:
    """Camera + memory flags together must instantiate cleanly.

    The camera swap already pulled in :class:`HyWorldPlayWanCtrlEncoderConfig`
    and the HY transformer config; the memory flag just rides on top
    of that with no further pipeline mutation (the encoder binding
    happens at ``run()`` time inside the native runner). This test
    pins the no-op shape so a future runner refactor doesn't try to
    swap the network again.
    """
    from hy_worldplay._action import (
        HyWorldPlayWan21TransformerConfig,
        HyWorldPlayWanCtrlEncoderConfig,
        HyWorldPlayWanDiTNetworkConfig,
    )
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_native_pipeline=True,
        use_camera_conditioning=True,
        use_memory_selection=True,
    )
    assert cfg.use_memory_selection is True
    assert isinstance(cfg.pipeline.encoder, HyWorldPlayWanCtrlEncoderConfig)
    transformer = cfg.pipeline.diffusion_model.transformer
    assert isinstance(transformer, HyWorldPlayWan21TransformerConfig)
    network = transformer.network
    assert isinstance(network, HyWorldPlayWanDiTNetworkConfig)
    assert network.use_prope_blocks is True


def test_use_memory_selection_requires_native_pipeline() -> None:
    """``use_memory_selection`` without ``use_native_pipeline`` is a no-op."""
    from hy_worldplay._vendor_pipeline import _NoopPipelineConfig
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(
        runner_name="hy-worldplay-wan-i2v-5b",
        use_memory_selection=True,
    )
    # Without ``use_native_pipeline`` the whole ``__post_init__`` body
    # is a no-op, so the precondition check on
    # ``use_memory_selection`` is *also* skipped -- the flag just
    # carries through inertly on the vendor wrapper.
    assert isinstance(cfg.pipeline, _NoopPipelineConfig)
    assert cfg.use_memory_selection is True


def test_use_memory_selection_memory_knobs_default_match_upstream() -> None:
    """Defaults track upstream's ``wan/inference/pipeline_wan_w_mem_relative_rope.py`` call site."""
    from hy_worldplay.runner import HyWorldPlayWanI2VRunnerConfig

    cfg = HyWorldPlayWanI2VRunnerConfig(runner_name="hy-worldplay-wan-i2v-5b")
    assert cfg.memory_frames == 16
    assert cfg.temporal_context_size == 12
    assert cfg.memory_pred_latent_size == 4
    assert cfg.memory_fov_h_deg == 60.0
    assert cfg.memory_fov_v_deg == 35.0
    assert cfg.memory_points_count == 50_000
    assert cfg.memory_points_radius == 8.0
