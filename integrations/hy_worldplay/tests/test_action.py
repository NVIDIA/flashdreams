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

"""CPU-only unit tests for the HY-WorldPlay action conditioner (phase 2b.3).

Tests split into three groups:

* Pose-string parser (no torch.nn surface): asserts the per-latent
  ``trans * 9 + rotate`` labels match what upstream's
  ``hyvideo/generate.py`` ``pose_to_input`` would have produced for the
  same inputs.
* DiT subclass surface: constructs a tiny ``HyWorldPlayWanDiTNetwork``
  on CPU (3-layer, 64-dim, 2-head) and asserts the ``action_embedding``
  MLP is wired in with a zero-residual head, so adding action
  conditioning is a strict identity at random / zero init.
* Transformer payload: round-trips a :class:`HyWorldPlayCtrl` through
  the subclass's ``patchify_and_maybe_split_cp`` and asserts the
  ``action`` field survives the rebuild that the base implementation
  otherwise drops.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ci_cpu


def test_pose_string_forward_only_labels() -> None:
    """``w-3`` produces 3 forward-translation latents after the identity start.

    Per upstream's ``pose_to_input``: frame 0 holds the identity pose; the
    next three frames carry pure forward translation. With ``trans=forward``
    (class 1) and ``rotate=identity`` (class 0), each non-zero label is
    ``1 * 9 + 0 = 9``.
    """
    from hy_worldplay._pose import parse_pose_action_labels

    labels = parse_pose_action_labels("w-3", n_latents=4)
    assert list(labels.tolist()) == [0, 9, 9, 9]


def test_pose_string_yaw_right_only_labels() -> None:
    """``right-3`` produces 3 yaw-right rotation latents.

    ``trans=identity`` (class 0), ``rotate=yaw_right`` (class 1):
    ``0 * 9 + 1 = 1`` for each non-identity frame.
    """
    from hy_worldplay._pose import parse_pose_action_labels

    labels = parse_pose_action_labels("right-3", n_latents=4)
    assert list(labels.tolist()) == [0, 1, 1, 1]


def test_pose_string_combined_motion_labels() -> None:
    """``w-2, right-2`` composes a forward run then a pure yaw-right run.

    Frame 0 is identity (label 0); frames 1-2 are forward translation
    (``trans=forward`` = class 1, ``rotate=identity`` = 0 -> ``1*9+0=9``);
    frames 3-4 are pure yaw-right (``trans=identity`` = 0,
    ``rotate=yaw_right`` = 1 -> ``0*9+1=1``). ``right`` is a yaw command
    in upstream's pose-string grammar, not a strafe.
    """
    from hy_worldplay._pose import parse_pose_action_labels

    labels = parse_pose_action_labels("w-2, right-2", n_latents=5)
    assert labels.tolist() == [0, 9, 9, 1, 1]


def test_pose_data_returns_w2c_K_action() -> None:
    """Three-tuple return signature for downstream 2b.4 camera plumbing."""
    from hy_worldplay._pose import parse_pose_data

    w2c, K, action = parse_pose_data("w-3", n_latents=4)
    assert w2c.shape == (4, 4, 4)
    assert K.shape == (4, 3, 3)
    assert action.shape == (4,)
    # Intrinsics normalized to (0.5, 0.5) principal point.
    assert float(K[0, 0, 2]) == pytest.approx(0.5)
    assert float(K[0, 1, 2]) == pytest.approx(0.5)


def test_pose_data_length_mismatch_raises() -> None:
    """Pose script with the wrong frame count must surface clearly."""
    from hy_worldplay._pose import parse_pose_action_labels

    with pytest.raises(ValueError, match="pose corresponds to"):
        parse_pose_action_labels("w-3", n_latents=10)


def test_pose_data_rejects_unknown_action() -> None:
    """Unknown action tokens must raise rather than silently fall through."""
    from hy_worldplay._pose import parse_pose_action_labels

    with pytest.raises(ValueError, match="Unknown action"):
        parse_pose_action_labels("xx-3", n_latents=4)


def test_pose_data_action_labels_in_range() -> None:
    """81-class encoding: every label must fall in ``[0, 81)``."""
    from hy_worldplay._pose import parse_pose_action_labels

    labels = parse_pose_action_labels("w-3, right-3", n_latents=7)
    assert int(labels.min()) >= 0
    assert int(labels.max()) < 81


def _make_tiny_network():
    """Construct a tiny ``HyWorldPlayWanDiTNetwork`` on CPU for surface tests."""
    from hy_worldplay._action import (
        HyWorldPlayWanDiTNetwork,
        HyWorldPlayWanDiTNetworkConfig,
    )

    cfg = HyWorldPlayWanDiTNetworkConfig(
        patch_size=(1, 2, 2),
        in_dim=4,
        out_dim=4,
        dim=64,
        ffn_dim=64,
        freq_dim=64,
        text_dim=64,
        num_heads=2,
        num_layers=1,
        text_len=8,
        cross_attn_enable_img=False,
    )
    network = HyWorldPlayWanDiTNetwork(cfg)
    return network, cfg


def test_dit_subclass_owns_action_embedding_mlp() -> None:
    """Subclass must publish ``action_embedding`` as a Sequential of the same shape as ``time_embedding``."""
    import torch
    import torch.nn as nn

    network, cfg = _make_tiny_network()
    assert hasattr(network, "action_embedding"), (
        "HyWorldPlayWanDiTNetwork must own an ``action_embedding`` MLP for "
        "the action conditioner to attach to."
    )
    assert isinstance(network.action_embedding, nn.Sequential)
    layers = list(network.action_embedding)
    assert len(layers) == 3
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.SiLU)
    assert isinstance(layers[2], nn.Linear)
    assert layers[0].in_features == cfg.freq_dim
    assert layers[0].out_features == cfg.dim
    assert layers[2].in_features == cfg.dim
    assert layers[2].out_features == cfg.dim
    assert torch.all(layers[2].weight == 0), (
        "action_embedding residual head must be zero-initialised so the "
        "conditioner is an identity at random init"
    )
    if layers[2].bias is not None:
        assert torch.all(layers[2].bias == 0), (
            "action_embedding residual bias must be zero-initialised"
        )


def test_action_embedding_zero_init_produces_zero_residual() -> None:
    """With zero-init weights ``_compute_action_embedding`` must return zeros.

    Combined with the additive ``e = e + action_e`` injection in
    :meth:`HyWorldPlayWanDiTNetwork.forward`, this is the parity guarantee
    that flipping on the action conditioner without HY-WorldPlay's distilled
    weights changes nothing in the modulation pathway.
    """
    import torch

    network, cfg = _make_tiny_network()
    L = 8 * 4  # 8 latent frames * 4 tokens/frame
    n_latent = 8
    action = torch.zeros(1, n_latent, dtype=torch.long)
    x = torch.zeros(1, L, cfg.dim)
    out = network._compute_action_embedding(action=action, x=x, L=L)
    assert out.shape == (1, L, cfg.dim)
    assert torch.all(out == 0)


def test_compute_action_embedding_rejects_indivisible_L() -> None:
    """``L`` must be a multiple of the action-frame count for repeat-interleave to fire."""
    import torch

    network, cfg = _make_tiny_network()
    action = torch.zeros(1, 7, dtype=torch.long)
    x = torch.zeros(1, 30, cfg.dim)
    with pytest.raises(ValueError, match="must divide the post-patchify token count"):
        network._compute_action_embedding(action=action, x=x, L=30)


def test_hy_ctrl_action_field_default_none() -> None:
    """The extension field defaults to ``None`` so legacy I2V callsites keep working."""
    import torch
    from hy_worldplay._action import HyWorldPlayCtrl

    ctrl = HyWorldPlayCtrl(latent=torch.zeros(1, 4), mask=torch.zeros(1, 4))
    assert ctrl.action is None


def test_patchify_override_is_declared_on_subclass() -> None:
    """The subclass must declare its own ``patchify_and_maybe_split_cp``.

    Structural guard: the base implementation rebuilds the I2V payload as
    a plain ``I2VCtrl``, which would silently drop the ``action`` slice.
    Re-using the inherited method (without the explicit override) would
    therefore corrupt the action conditioner.
    """
    from hy_worldplay._action import HyWorldPlayWan21Transformer

    from flashdreams.recipes.wan.transformer.wan21 import Wan21Transformer

    assert (
        HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp
        is not Wan21Transformer.patchify_and_maybe_split_cp
    ), (
        "HyWorldPlayWan21Transformer must override patchify_and_maybe_split_cp "
        "to preserve the action field through the patchify rebuild."
    )


def test_patchify_idempotent_on_patchified_ctrl() -> None:
    """An already-patchified ``HyWorldPlayCtrl`` returns as-is and keeps action.

    This is the cheap branch of the override (no ``super()`` call), so it
    can be exercised on a stand-in instance without instantiating the full
    transformer + compile + CUDA-graph stack.
    """
    import torch
    from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21Transformer

    action = torch.tensor([3, 5, 9, 1], dtype=torch.long)
    ctrl = HyWorldPlayCtrl(
        latent=torch.arange(8, dtype=torch.float32),
        mask=torch.zeros(8, dtype=torch.float32),
        action=action,
        _is_patchified=True,
    )
    # The patchified branch is pure ``isinstance`` + early return, so an
    # ``object.__new__(...)`` of the subclass is enough to dispatch
    # ``self.patchify_and_maybe_split_cp(ctrl)`` without running
    # ``__init__``'s heavy stack.
    transformer = object.__new__(HyWorldPlayWan21Transformer)
    out = transformer.patchify_and_maybe_split_cp(ctrl)
    assert out is ctrl
    assert out.action is action


def test_predict_flow_threads_action_via_network_extra_kwargs() -> None:
    """Predict-flow override copies ``input.action`` into network_extra_kwargs.

    Bypasses the heavy ``Wan21Transformer.__init__`` by monkey-patching the
    base ``predict_flow`` on a stand-in instance so the test stays CPU-fast.
    """
    import torch
    from hy_worldplay._action import HyWorldPlayCtrl, HyWorldPlayWan21Transformer

    from flashdreams.recipes.wan.transformer.wan21 import Wan21Transformer

    captured: dict[str, dict] = {}

    def _capture_predict_flow(self, **kwargs):  # noqa: ANN001
        captured["kwargs"] = kwargs
        return torch.zeros(1)

    transformer = object.__new__(HyWorldPlayWan21Transformer)
    action = torch.tensor([3, 5, 9, 1], dtype=torch.long)
    ctrl = HyWorldPlayCtrl(
        latent=torch.zeros(1, 4),
        mask=torch.zeros(1, 4),
        action=action,
    )

    original = Wan21Transformer.predict_flow
    Wan21Transformer.predict_flow = _capture_predict_flow  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
    try:
        HyWorldPlayWan21Transformer.predict_flow(
            transformer,
            noisy_latent=torch.zeros(1, 4),
            timestep=torch.tensor(0.5),
            cache=None,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            input=ctrl,
            network_extra_kwargs=None,
        )
    finally:
        Wan21Transformer.predict_flow = original  # type: ignore[assignment]

    assert "network_extra_kwargs" in captured["kwargs"]
    nek = captured["kwargs"]["network_extra_kwargs"]
    assert nek is not None
    assert "action" in nek
    assert nek["action"] is action
