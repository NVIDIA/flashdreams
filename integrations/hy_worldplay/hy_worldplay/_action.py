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

"""HY-WorldPlay action + camera + memory conditioner glue (phases 2b.3 - 2b.5a).

Adds discrete 81-class action conditioning (2b.3), the per-AR-step
camera data plumbing for the PRoPE branch (2b.4), and the
reconstituted-context **memory-frame selection** (2b.5a) on top of the
Wan 2.2 TI2V 5B stack. The four pieces compose into a drop-in
replacement of the standard encoder + transformer + network used by
``PIPELINE_WAN22_TI2V_5B``:

* :class:`HyWorldPlayCtrl` extends :class:`I2VCtrl` with ``action``,
  ``viewmats``, ``Ks``, and ``memory_frame_indices`` fields. ``action``
  carries the per-latent-frame discrete labels; ``viewmats`` / ``Ks``
  carry the per-frame W2C extrinsics + intrinsics consumed by the PRoPE
  attention branch; ``memory_frame_indices`` is the sorted list of
  historical frame indices selected by
  :func:`hy_worldplay._memory.select_memory_frame_indices` for the
  upcoming KV-prefill pass (consumer lands in 2b.5b together with the
  ``BlockKVCache`` arbitrary-position-write extension).
* :class:`HyWorldPlayWanCtrlEncoder` wraps :class:`I2VCtrlEncoder` and
  slices the per-rollout action labels / camera tensors into the per-AR-
  step :class:`HyWorldPlayCtrl` payload via :meth:`set_action_labels` /
  :meth:`set_camera_data`, and computes the per-AR-step memory frame
  indices on demand via :meth:`set_memory_config` + the bound camera
  history. Each source can be bound independently; unbound sources flow
  through as ``None`` so downstream consumers stay opt-in.
* :class:`HyWorldPlayWanDiTNetwork` extends :class:`WanDiTNetwork` with a
  zero-residual ``action_embedding`` MLP summed into the time embedding
  before the AdaLN modulation projection (mirrors
  ``WanActionTimeTextImageEmbedding`` in upstream's
  ``arwan_w_action_w_mem_relative_rope.py``), and -- when
  :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` is set --
  builds :class:`HyWorldPlayPRoPEBlock` blocks so each self-attention
  runs the dual-branch RoPE + PRoPE path. Both injection paths are
  zero-init residuals so the network stays parity-safe at random /
  base-recipe init.
* :class:`HyWorldPlayWan21Transformer` re-wires :meth:`predict_flow` to
  forward ``action`` / ``viewmats`` / ``Ks`` through
  ``network_extra_kwargs``, and overrides
  :meth:`patchify_and_maybe_split_cp` so all three fields survive the
  patchify-rebuild of the I2V payload.

CP is intentionally restricted to size 1 here; multi-rank action +
PRoPE expansion lands in a follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from flashdreams.recipes.wan.autoencoder.i2v import (
    I2VCtrl,
    I2VCtrlEncoder,
    I2VCtrlEncoderCache,
    WanI2VCtrlEncoderConfig,
)
from flashdreams.recipes.wan.transformer.impl.modules import (
    sinusoidal_embedding_1d,
)
from flashdreams.recipes.wan.transformer.impl.network import (
    Block,
    WanDiTNetwork,
    WanDiTNetworkCache,
    WanDiTNetworkTI2V5BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import (
    Wan21Transformer,
    Wan21TransformerCache,
    Wan21TransformerConfig,
)

# Clean-context timestep for the reconstituted-context KV prefill.
# Mirrors upstream's ``t_ctx = stabilization_level - 1`` constant in
# ``HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py``
# (line 680 ``stabilization_level = 15`` and line 883-887 / 908-913
# where chunk-0 memory positions get ``stabilization_level - 1 = 14``
# as their AdaLN timestep). On the FlowMatch 0..1000 timestep scale
# this is a near-clean modulation that keeps the model in its trained
# distribution; passing the noisy denoising step instead would scale
# the memory K / V as if chunk-0 were still being denoised and
# produces the chunk-1 attention blow-up surfaced by 2b.6.
_HY_STABILIZATION_TIMESTEP: int = 14


def _fp32_sequential(seq: nn.Sequential, x: Tensor) -> Tensor:
    """Run ``seq`` (a chain of ``nn.Linear`` + activations) in fp32.

    Vendor's ``WanTransformer3DModel`` lists ``time_embedder`` in
    ``_keep_in_fp32_modules`` so its weights and biases stay in fp32
    even when the surrounding model is bf16. flashdreams loads the
    same checkpoint with every parameter coerced to the model dtype
    (bf16 for the distilled WAN-5B), which forces the chained
    Linear/SiLU/Linear MLP to accumulate in bf16 and drops ~3 bits
    of precision per op. This helper bridges the two by casting
    *both* the input and any ``nn.Linear`` weights to fp32 for the
    matmul; SiLU / GELU / Tanh activations are dtype-stable and pass
    through unchanged.

    The output is fp32; callers can ``.type_as(x)`` at the boundary
    to keep downstream broadcast targets aligned with the network's
    nominal dtype.

    Limitations: this only handles the subset of ``nn.Sequential``
    layouts used by Wan 2.1's ``time_embedding`` /
    ``time_projection`` and HY-WorldPlay's ``action_embedding``
    (``Linear -> Activation -> Linear`` or
    ``Activation -> Linear``). Layers that hold dtype-coupled state
    (e.g. layer norms with fp32 weights stored as bf16) would need
    the more general treatment in
    :func:`hy_worldplay._camera._fp32_layer_norm`.
    """
    out = x.to(torch.float32)
    for module in seq:
        if isinstance(module, nn.Linear):
            weight = module.weight.to(torch.float32)
            bias = (
                module.bias.to(torch.float32)
                if module.bias is not None
                else None
            )
            out = F.linear(out, weight, bias)
        else:
            out = module(out)
    return out


## ---------------------------------------------------------------------------
## Per-AR-step control payload
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayCtrl(I2VCtrl):
    """I2V control payload extended with per-AR-step action + camera +
    memory slices.

    The ``action`` field carries the integer class labels for the current
    chunk's ``len_t`` latent frames (shape ``[*batch_shape, len_t]``).
    The ``viewmats`` / ``Ks`` fields carry the per-frame world-to-camera
    extrinsic and intrinsic matrices consumed by the PRoPE attention
    branch (2b.4). The ``memory_frame_indices`` list selects the
    historical frame indices the KV-prefill pass should attend to for
    this AR step (2b.5a; the prefill *consumer* lands in 2b.5b). All
    four fields survive the transformer's patchify-rebuild via
    :meth:`HyWorldPlayWan21Transformer.patchify_and_maybe_split_cp`.
    """

    action: Tensor | None = None
    """Per-latent-frame action labels for the current AR chunk."""

    viewmats: Tensor | None = None
    """Per-latent-frame world-to-camera matrices for the current AR chunk;
    shape ``[*batch_shape, len_t, 4, 4]``. Consumed by the PRoPE branch
    of :class:`HyWorldPlayPRoPEBlock` self-attention."""

    Ks: Tensor | None = None
    """Per-latent-frame intrinsics (with cx/cy renormalised to 0.5) for
    the current AR chunk; shape ``[*batch_shape, len_t, 3, 3]``."""

    memory_frame_indices: list[int] | None = None
    """Sorted, deduplicated historical frame indices for the upcoming
    KV-prefill pass. Populated by
    :meth:`HyWorldPlayWanCtrlEncoder.forward` when memory selection is
    armed and there is enough history (``current_frame_idx >=
    context_window_length``). ``None`` for the first AR chunk and when
    memory selection is disabled. Indexes into the per-rollout
    :attr:`rollout_viewmats` / :attr:`rollout_Ks` / :attr:`rollout_action`
    buffers (frame-granular, not token-granular)."""

    rollout_viewmats: Tensor | None = None
    """Per-*rollout* world-to-camera matrices (the *full* trajectory,
    not the current AR chunk's slice). Shape
    ``[*batch_shape, F_total, 4, 4]`` where ``F_total`` is
    ``num_chunk * len_t``. Read by
    :meth:`HyWorldPlayWan21Transformer.prefill_memory_kv_cache` to
    slice the K selected memory frames at
    :attr:`memory_frame_indices` -- without this buffer the prefill
    would have to fall back to the current chunk's slice (the
    ``_slice_per_frame`` stub from the 2b.5b-part2 first cut), which
    is parity-incorrect because memory frames live in past chunks.
    ``None`` when camera conditioning is disabled or the encoder
    hasn't bound camera data via :meth:`HyWorldPlayWanCtrlEncoder.set_camera_data`."""

    rollout_Ks: Tensor | None = None
    """Per-rollout intrinsics buffer; shape
    ``[*batch_shape, F_total, 3, 3]``. Sibling of
    :attr:`rollout_viewmats`; bound together by
    :meth:`HyWorldPlayWanCtrlEncoder.set_camera_data` and read together
    by the prefill driver."""

    rollout_action: Tensor | None = None
    """Per-rollout action labels; shape ``[*batch_shape, F_total]``.
    Same role as :attr:`rollout_viewmats` for the action conditioner:
    when memory selection is armed, the prefill driver slices this at
    :attr:`memory_frame_indices` to get the action label per memory
    frame (used by :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache`
    to compute the AdaLN modulation). ``None`` when action
    conditioning is disabled or the encoder hasn't bound action
    labels via :meth:`HyWorldPlayWanCtrlEncoder.set_action_labels`."""


## ---------------------------------------------------------------------------
## I2V + action encoder
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayWanCtrlEncoderConfig(WanI2VCtrlEncoderConfig):
    """Config for the action-aware I2V control encoder.

    The encoder slices the per-rollout label tensor (set externally via
    :meth:`HyWorldPlayWanCtrlEncoder.set_action_labels`) into the per-AR-step
    ``action`` field of :class:`HyWorldPlayCtrl`; ``latent`` / ``mask`` are
    produced by the inherited Wan I2V branch unchanged.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanCtrlEncoder)


class HyWorldPlayWanCtrlEncoder(I2VCtrlEncoder):
    """Wan I2V encoder that also emits per-AR-step action + camera slices.

    Callers populate the full per-rollout label tensor via
    :meth:`set_action_labels` and (for the PRoPE camera branch added in
    2b.4) the per-rollout extrinsics + intrinsics via
    :meth:`set_camera_data`. Each :meth:`forward` call then slices the
    ``[ar_idx * len_t : (ar_idx + 1) * len_t]`` window from whichever
    sources are bound and attaches them to the :class:`HyWorldPlayCtrl`
    payload; unbound sources fall through as ``None`` so downstream
    consumers stay opt-in.
    """

    def __init__(self, config: HyWorldPlayWanCtrlEncoderConfig) -> None:
        super().__init__(config)
        self._action_labels: Tensor | None = None
        self._viewmats: Tensor | None = None
        self._intrinsics: Tensor | None = None
        # Memory selection (2b.5a) -- knobs + Monte-Carlo point cloud
        # are bound externally via ``set_memory_config``; ``None``
        # means selection is off and the encoder emits
        # ``memory_frame_indices=None`` on every AR step.
        self._memory_config: _MemoryConfig | None = None

    def set_action_labels(self, labels: Tensor) -> None:
        """Bind the per-rollout action labels.

        ``labels`` must have a trailing axis whose length is divisible by the
        transformer's ``len_t`` so successive AR steps see equal-sized
        slices.
        """
        if labels.ndim < 1:
            raise ValueError(
                f"action labels must have at least 1 dim, got shape "
                f"{tuple(labels.shape)}."
            )
        self._action_labels = labels

    def clear_action_labels(self) -> None:
        """Drop the per-rollout label tensor (used when reusing the encoder)."""
        self._action_labels = None

    def set_camera_data(self, viewmats: Tensor, Ks: Tensor) -> None:
        """Bind the per-rollout camera extrinsics + intrinsics.

        Args:
            viewmats: Per-latent-frame world-to-camera matrices, shape
                ``[*, n_latents, 4, 4]`` where ``n_latents`` is divisible
                by the transformer's ``len_t``.
            Ks: Per-latent-frame intrinsics (with cx/cy renormalised to
                0.5 by :func:`hy_worldplay._pose.parse_pose_data`), shape
                ``[*, n_latents, 3, 3]``. Both tensors share the same
                leading axes.
        """
        if viewmats.ndim < 3 or viewmats.shape[-2:] != (4, 4):
            raise ValueError(
                f"viewmats must have trailing shape (n_latents, 4, 4); "
                f"got {tuple(viewmats.shape)}."
            )
        if Ks.ndim < 3 or Ks.shape[-2:] != (3, 3):
            raise ValueError(
                f"Ks must have trailing shape (n_latents, 3, 3); "
                f"got {tuple(Ks.shape)}."
            )
        if viewmats.shape[:-2] != Ks.shape[:-2]:
            raise ValueError(
                f"viewmats and Ks must share the leading dims preceding "
                f"the matrix axes; got viewmats={tuple(viewmats.shape)}, "
                f"Ks={tuple(Ks.shape)}."
            )
        self._viewmats = viewmats
        self._intrinsics = Ks

    def clear_camera_data(self) -> None:
        """Drop the per-rollout camera tensors (used when reusing the encoder)."""
        self._viewmats = None
        self._intrinsics = None

    def set_memory_config(
        self,
        *,
        points_local: Tensor,
        context_window_length: int,
        memory_frames: int,
        temporal_context_size: int,
        pred_latent_size: int,
        fov_h_deg: float,
        fov_v_deg: float,
        device: torch.device | str | None = None,
    ) -> None:
        """Arm reconstituted-context memory selection for this rollout.

        Stashes the Monte-Carlo point cloud + selection knobs so each
        :meth:`forward` call can compute the per-AR-step
        ``memory_frame_indices`` from the bound camera history. The
        first AR step (and any subsequent step where
        ``current_frame_idx < context_window_length``) bypasses the
        selection algorithm and produces ``memory_frame_indices=None``,
        matching upstream's ``elif use_memory`` branch in
        ``pipeline_wan_w_mem_relative_rope.py`` line 868-869.

        Args:
            points_local: Pre-sampled cloud of 3D points, shape
                ``[N, 3]``. Build once via
                :func:`hy_worldplay._memory.generate_points_in_sphere`
                and reuse for the whole rollout.
            context_window_length: Threshold (in latent frames) below
                which the encoder emits ``None`` instead of running
                the FOV-overlap selection (upstream default 16).
            memory_frames: Total budget of memory frames the selector
                returns once armed.
            temporal_context_size: Recent-frames portion of the memory
                budget (kept unconditionally).
            pred_latent_size: Length of the query clip the selector
                scores historical clips against.
            fov_h_deg / fov_v_deg: FOV (degrees) used by the overlap
                computation.
            device: Optional torch device for the overlap math. Pass
                the runner's compute device for GPU rollouts.
        """
        if memory_frames < temporal_context_size:
            raise ValueError(
                f"memory_frames ({memory_frames}) must be >= "
                f"temporal_context_size ({temporal_context_size})."
            )
        if points_local.ndim != 2 or points_local.shape[-1] != 3:
            raise ValueError(
                f"points_local must have shape (N, 3); got "
                f"{tuple(points_local.shape)}."
            )
        self._memory_config = _MemoryConfig(
            points_local=points_local,
            context_window_length=context_window_length,
            memory_frames=memory_frames,
            temporal_context_size=temporal_context_size,
            pred_latent_size=pred_latent_size,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            device=device,
        )

    def clear_memory_config(self) -> None:
        """Disarm memory selection (used when reusing the encoder)."""
        self._memory_config = None

    def initialize_autoregressive_cache(self) -> I2VCtrlEncoderCache:
        # Match the parent's per-rollout reset; action / camera tensors
        # and memory config are bound explicitly after
        # ``initialize_cache`` so we deliberately do *not* clear them
        # here. The runner clears them when it tears down the rollout.
        return super().initialize_autoregressive_cache()

    @torch.no_grad()
    def forward(
        self,
        input: Tensor,
        autoregressive_index: int = 0,
        cache: I2VCtrlEncoderCache | None = None,
    ) -> HyWorldPlayCtrl:
        base = super().forward(
            input=input, autoregressive_index=autoregressive_index, cache=cache
        )
        len_t = base.latent.shape[-4]
        start = autoregressive_index * len_t
        end = start + len_t
        device = base.latent.device

        action_chunk: Tensor | None = None
        if self._action_labels is not None:
            if end > self._action_labels.shape[-1]:
                raise ValueError(
                    f"action labels exhausted at AR step {autoregressive_index}: "
                    f"need {end} entries but only "
                    f"{self._action_labels.shape[-1]} provided."
                )
            action_chunk = self._action_labels[..., start:end].to(device=device)

        viewmats_chunk: Tensor | None = None
        Ks_chunk: Tensor | None = None
        if self._viewmats is not None:
            assert self._intrinsics is not None, (
                "viewmats and Ks must be bound together via set_camera_data; "
                "found viewmats without intrinsics."
            )
            total = self._viewmats.shape[-3]
            if end > total:
                raise ValueError(
                    f"camera tensors exhausted at AR step {autoregressive_index}: "
                    f"need {end} frames but only {total} provided."
                )
            viewmats_chunk = self._viewmats[..., start:end, :, :].to(device=device)
            Ks_chunk = self._intrinsics[..., start:end, :, :].to(device=device)

        memory_indices: list[int] | None = self._compute_memory_indices(
            autoregressive_index=autoregressive_index, current_frame_idx=start
        )

        # Per-rollout buffers (phase 2b.5b-part2-followup): expose the
        # bound full-trajectory tensors so the prefill driver in
        # ``HyWorldPlayWan21Transformer.prefill_memory_kv_cache`` can
        # index them at ``memory_frame_indices`` (which live in
        # *rollout* coordinates, not chunk coordinates). We move the
        # buffers to the latent's device once per AR step rather than
        # once per memory-prefill call so the device-transfer cost is
        # amortised. The encoder's bound storage stays on whatever
        # device the caller put it on (typically CPU during config
        # build), and the per-AR-step ctrl gets the device-correct
        # view.
        rollout_viewmats: Tensor | None = (
            self._viewmats.to(device=device) if self._viewmats is not None else None
        )
        rollout_Ks: Tensor | None = (
            self._intrinsics.to(device=device)
            if self._intrinsics is not None
            else None
        )
        rollout_action: Tensor | None = (
            self._action_labels.to(device=device)
            if self._action_labels is not None
            else None
        )

        return HyWorldPlayCtrl(
            latent=base.latent,
            mask=base.mask,
            action=action_chunk,
            viewmats=viewmats_chunk,
            Ks=Ks_chunk,
            memory_frame_indices=memory_indices,
            rollout_viewmats=rollout_viewmats,
            rollout_Ks=rollout_Ks,
            rollout_action=rollout_action,
        )

    def _compute_memory_indices(
        self, *, autoregressive_index: int, current_frame_idx: int
    ) -> list[int] | None:
        """Pick the historical frame indices for this AR step's KV prefill.

        Mirrors the gating in upstream's
        ``pipeline_wan_w_mem_relative_rope.py`` line 853-869:

        * AR step 0 (``current_frame_idx == 0``) returns ``None`` --
          the first chunk has no history to attend to.
        * FOV-based selection runs when memory selection is configured
          *and* the rollout is past the warm-up window
          (``current_frame_idx >= context_window_length``). Mirrors
          upstream's ``if use_memory and current_frame_idx >=
          context_window_length:`` branch.
        * All other chunk-> 0 steps return ``list(range(0,
          current_frame_idx))`` -- the all-history fall-back of
          upstream's ``elif use_memory:`` branch.

        That all-history fall-back is **required** on the HY native
        path, not just a vendor quirk: the HY override of
        :meth:`HyWorldPlayWan21Transformer.finalize_kv_cache` skips
        the base ``Wan21Transformer`` rolling-KV update, and the HY
        cache ``start`` resets each block's rolling self-attention
        cache at every chunk boundary. Without the explicit prefill
        driven by ``memory_frame_indices``, chunk-1+ would attend to
        *nothing* from previous chunks -- producing the chunk-boundary
        denoising blow-up that 2b.6 was chasing. Vendor avoids this
        because its KV cache accumulates naturally across chunks; we
        give the HY path the same cross-chunk coverage via the
        prefill executor.

        Requires camera data to be bound (``self._viewmats`` set via
        :meth:`set_camera_data`) so the per-rollout buffers are
        attached to the ctrl and the prefill executor can index them.
        When camera data isn't bound the prefill can't run at all, so
        we degrade to ``None`` (the dual-branch and action paths are
        themselves no-ops in that configuration so the missing
        prefill is also a no-op).
        """
        if autoregressive_index == 0 or current_frame_idx == 0:
            return None
        # No camera history => no prefill possible (the executor
        # indexes ``rollout_viewmats`` and friends, which are only
        # populated when the encoder has bound camera data).
        if self._viewmats is None:
            return None
        # FOV-based selection: vendor's ``if use_memory and
        # current_frame_idx >= context_window_length:`` branch.
        if (
            self._memory_config is not None
            and current_frame_idx >= self._memory_config.context_window_length
        ):
            cfg = self._memory_config
            # ``_viewmats`` is shape ``[*batch, n_latents, 4, 4]``; the
            # FOV selector ignores batch and consumes a flat
            # ``[n_latents, 4, 4]`` history. We use the first batch slot
            # since per-batch camera trajectories aren't supported by
            # upstream's selector either (it also takes ``viewmats[0]``).
            viewmats_history = self._viewmats
            while viewmats_history.ndim > 3:
                viewmats_history = viewmats_history[0]
            # Lazy-imported -- the memory module pulls in numpy and the
            # FOV math; keep that out of the import graph when memory
            # selection is disabled.
            from hy_worldplay._memory import select_memory_frame_indices

            # ``.numpy()`` only supports a subset of dtypes; the runner
            # binds ``viewmats`` in the pipeline dtype (bf16 / fp16) so
            # ``prope_qkv`` doesn't promote to fp64, but bf16 has no numpy
            # ABI. Round-trip through fp32 here so the selection math
            # (FOV-overlap on a CPU point cloud) consumes a plain
            # ``np.float32`` array; selection precision is not the
            # bottleneck vs the bf16 dtype used downstream by attention.
            return select_memory_frame_indices(
                viewmats_history.detach().to(dtype=torch.float32).cpu().numpy(),
                current_frame_idx=current_frame_idx,
                points_local=cfg.points_local,
                memory_frames=cfg.memory_frames,
                temporal_context_size=cfg.temporal_context_size,
                pred_latent_size=cfg.pred_latent_size,
                fov_h_deg=cfg.fov_h_deg,
                fov_v_deg=cfg.fov_v_deg,
                device=cfg.device,
            )

        # All-history fall-back (vendor's ``elif use_memory:`` branch).
        # Critical for cross-chunk attention on the HY native path; see
        # the docstring above.
        return list(range(0, current_frame_idx))


@dataclass(frozen=True)
class _MemoryConfig:
    """Internal bag for the memory-selection knobs bound on the encoder.

    Frozen so accidental in-place mutation between AR steps is a hard
    error -- the selection policy is deterministic given the bound
    camera history and these knobs.
    """

    points_local: Tensor
    context_window_length: int
    memory_frames: int
    temporal_context_size: int
    pred_latent_size: int
    fov_h_deg: float
    fov_v_deg: float
    device: torch.device | str | None


## ---------------------------------------------------------------------------
## Action-aware DiT network
## ---------------------------------------------------------------------------


@dataclass
class HyWorldPlayWanDiTNetworkConfig(WanDiTNetworkTI2V5BConfig):
    """Config for the action / camera-aware Wan 2.2 TI2V 5B DiT.

    Shares every field with :class:`WanDiTNetworkTI2V5BConfig`. Adds:

    * :attr:`_target` swap so the network gains the ``action_embedding``
      MLP and the action / camera-injecting forward.
    * :attr:`use_prope_blocks` (2b.4 knob) -- when ``True``,
      :meth:`HyWorldPlayWanDiTNetwork._build_block` returns
      :class:`HyWorldPlayPRoPEBlock` instances so each block runs the
      dual-branch RoPE + PRoPE self-attention. Defaults to ``False`` so
      enabling action conditioning alone (no camera) keeps the standard
      :class:`Block` stack and the network's behaviour stays bit-identical
      to 2b.3.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWanDiTNetwork)

    use_prope_blocks: bool = False
    """Build PRoPE-aware blocks (dual-branch RoPE + PRoPE self-attn)
    instead of the standard :class:`Block`. When ``True`` the encoder must
    bind per-rollout camera data so each AR step's :class:`HyWorldPlayCtrl`
    carries ``viewmats`` + ``Ks`` slices."""


class HyWorldPlayWanDiTNetwork(WanDiTNetwork):
    """Wan DiT with action-modulated AdaLN + optional PRoPE blocks.

    Two extensions on top of :class:`WanDiTNetwork`:

    * ``action_embedding`` MLP (same shape as ``time_embedding``) consumes
      sinusoidally-encoded action class labels and produces a per-latent-
      frame additive term summed into the time embedding before
      ``time_projection``. ``linear_2`` is zero-initialised so the
      conditioner is a strict identity at random / zero init.
    * When :attr:`HyWorldPlayWanDiTNetworkConfig.use_prope_blocks` is set,
      :meth:`_build_block` returns :class:`HyWorldPlayPRoPEBlock` instances
      so each block runs the dual-branch RoPE + PRoPE self-attention. The
      forward routes ``viewmats`` and ``Ks`` (per-AR-step camera data)
      through ``block_extra_kwargs`` so each PRoPE block sees the same
      per-frame extrinsics + intrinsics. ``o_prope`` in the PRoPE attention
      is zero-init so the dual-branch path also stays a strict identity
      until HY-WorldPlay weights are loaded.
    """

    def __init__(self, config: HyWorldPlayWanDiTNetworkConfig) -> None:
        # Pre-init nn.Module so we can stash ``use_prope_blocks`` *before*
        # super().__init__() runs ``_build_block`` via ``self``. The
        # outer ``super().__init__()`` re-initialises ``_parameters`` /
        # ``_modules`` etc. but leaves ordinary attributes (like
        # ``_hy_use_prope_blocks``) alone.
        nn.Module.__init__(self)
        self._hy_use_prope_blocks = config.use_prope_blocks
        super().__init__(config)
        self.action_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        # Zero-init the residual head so the action branch is an identity at
        # construction time (matches upstream
        # ``add_discrete_action_parameters``).
        zero_linear = self.action_embedding[-1]
        assert isinstance(zero_linear, nn.Linear)
        nn.init.zeros_(zero_linear.weight)
        if zero_linear.bias is not None:
            nn.init.zeros_(zero_linear.bias)

    def _build_block(self, layer_idx: int) -> Block:
        if self._hy_use_prope_blocks:
            # Lazy-imported so the PRoPE block module is only pulled in
            # when the camera conditioner is actually enabled (keeps the
            # action-only path free of the extra import cost).
            from hy_worldplay._camera import HyWorldPlayPRoPEBlock

            return HyWorldPlayPRoPEBlock(
                dim=self.dim,
                ffn_dim=self.ffn_dim,
                num_heads=self.num_heads,
                cross_attn_norm=self.cross_attn_norm,
                eps=self.eps,
                i2v=self.cross_attn_enable_img,
                apply_rope_before_kvcache=self.apply_rope_before_kvcache,
            )
        return super()._build_block(layer_idx)

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        current_chunk_idx: int = 0,
        eager_mode: bool = True,
        block_extra_kwargs: dict[str, Any] = {},
        action: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> Tensor:
        """Action / camera-aware variant of :meth:`WanDiTNetwork.forward`.

        Extends the base forward by (a) adding the action embedding to the
        time embedding before the modulation projection and (b) threading
        ``viewmats`` + ``Ks`` through ``block_extra_kwargs`` when PRoPE
        blocks are active. When both ``action`` and ``viewmats`` are
        ``None`` the modulation path is bit-for-bit identical to the base;
        PRoPE blocks still require ``viewmats`` even when nothing else is
        bound.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        per_token_timestep = (
            timesteps.ndim > len(batch_shape) and timesteps.shape[-1] == L
        )
        # Run ``time_embedding`` in fp32 to match vendor's
        # ``_keep_in_fp32_modules = ["time_embedder", ...]`` behaviour.
        # Vendor's diffusers config keeps the *time embedder* in fp32
        # weights so the chained ``Linear -> SiLU -> Linear`` MLP
        # accumulates in fp32 even when the surrounding model is bf16
        # (the output is then cast back via ``.type_as(...)`` before
        # the per-block AdaLN blend). Native loads the same checkpoint
        # in bf16 throughout, so without the explicit upcast we drop
        # ~3 bits of precision per Linear in the embedding head.
        # ``time_projection`` (vendor: ``time_proj``) is *not* in
        # vendor's fp32 list and stays in bf16; we follow suit.
        e_fp32 = _fp32_sequential(
            self.time_embedding,
            sinusoidal_embedding_1d(self.freq_dim, timesteps).to(torch.float32),
        )
        e = e_fp32.type_as(x)

        if action is not None:
            action_e = self._compute_action_embedding(action=action, x=x, L=L)
            # Vendor performs this add in bf16 (both ``temb`` and
            # ``action`` have already been ``.type_as``'d to bf16
            # before the ``temb = temb + action`` line); we mirror
            # that by casting ``e_fp32`` back to ``x.dtype`` before
            # the add.
            e = e + action_e
            per_token_timestep = True

        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))

        if per_token_timestep:
            block_e_shape = batch_shape + (L, 6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (L, self.dim)).unsqueeze(-2)
        else:
            block_e_shape = batch_shape + (6, self.dim)
            head_e = torch.broadcast_to(e, batch_shape + (self.dim,)).unsqueeze(-2)
        block_e = torch.broadcast_to(e0, block_e_shape)

        # Thread camera data per-block when PRoPE blocks are active. We
        # copy ``block_extra_kwargs`` rather than mutate the caller's
        # dict so subsequent forwards (or other branches) see a clean
        # slate.
        block_kwargs = dict(block_extra_kwargs)
        if self._hy_use_prope_blocks:
            if viewmats is None:
                raise ValueError(
                    "use_prope_blocks=True requires viewmats; "
                    "the encoder must bind camera data via "
                    "set_camera_data so HyWorldPlayCtrl.viewmats is populated."
                )
            block_kwargs["viewmats"] = viewmats
            block_kwargs["Ks"] = Ks

        from hy_worldplay import _debug_dump

        if eager_mode:
            cache.before_update(current_chunk_idx)
        for block_idx, block in enumerate(self.blocks):
            assert isinstance(block, Block)
            _debug_dump.set_context(phase="forward", block_idx=block_idx)
            x = block(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                cache=cache[block_idx],
                **block_kwargs,
            )
        _debug_dump.clear_context("phase", "block_idx")
        if eager_mode:
            cache.after_update(current_chunk_idx)

        x = self.head(x, head_e)
        return x

    def prefill_memory_kv_cache(
        self,
        x: Tensor,
        timesteps: Tensor,
        cache: WanDiTNetworkCache,
        rope_freqs: Tensor,
        block_extra_kwargs: dict[str, Any] | None = None,
        action: Tensor | None = None,
        viewmats: Tensor | None = None,
        Ks: Tensor | None = None,
    ) -> None:
        """Populate each block's reconstituted-context memory cache.

        Phase 2b.5b-part2 prefill driver -- mirror :meth:`forward`'s
        patchify + time / action embedding + AdaLN modulation pre-amble,
        then loop over blocks calling
        :meth:`HyWorldPlayPRoPEBlock.prefill_memory_kv` instead of
        ``block(...)``. Cross-attention, FFN, the residual stream, and
        the head are all skipped because the prefill executor only
        needs each block's self-attention K / V at the collapsed RoPE
        positions; everything downstream of self-attention is
        unobservable in the cache.

        Pre-conditions:

        * ``x`` already carries the *memory frames'* patchified latents
          at upstream's collapsed positions ``[0, K * tokens_per_frame)``;
          the caller (``HyWorldPlayWan21Transformer.prefill_memory_kv_cache``)
          is responsible for slicing the per-rollout history at
          ``HyWorldPlayCtrl.memory_frame_indices`` and reshaping.
        * ``rope_freqs`` is built from the rope adapter's
          ``_freq_components`` primitive against the same collapsed
          positions, *not* against the standard chunk-i positions
          ``[i*len_t, (i+1)*len_t)`` -- this is the whole point of the
          dedicated prefill pass: rotate K with the collapsed positions
          so the cached K already encodes "memory at position 0..K-1"
          when concatenated with the current chunk's K.
        * ``viewmats`` / ``Ks`` (when PRoPE blocks are active) carry
          the memory frames' camera matrices, again pre-sliced.
        * ``cache`` is a :class:`WanDiTNetworkCache` whose per-block
          entries are :class:`HyWorldPlayPRoPEBlockCache` instances; the
          ``memory`` slot on each entry is what gets written.

        Args:
            x: Patchified memory latents with shape
                ``[..., L_mem, in_dim]``.
            timesteps: Scalar broadcast or per-token context-noise
                timestep, same contract as :meth:`forward` but pre-
                sliced to the memory tokens. The standard upstream
                value is ``stabilization_level`` (a small near-zero
                noise level distinct from 0 to keep the model in its
                trained distribution); the caller passes this through.
            cache: Per-block cache; only the ``memory`` slots are
                touched.
            rope_freqs: RoPE frequencies remapped to the collapsed
                memory positions ``[0, L_mem)``.
            block_extra_kwargs: Optional extras forwarded to the per-
                block prefill (currently unused; kept for symmetry
                with :meth:`forward`).
            action: Optional action labels for the *memory frames*
                (one label per selected latent frame). When ``None``
                no action embedding is added.
            viewmats: Optional W2C extrinsics for the memory frames.
                Required when ``self._hy_use_prope_blocks`` is set
                (PRoPE branch needs them to compute the camera
                projection).
            Ks: Optional per-frame intrinsics for the memory frames.
        """
        assert self._parameters_updated_after_loading_checkpoint, (
            "We expect to have called update_parameters_after_loading_checkpoint() "
            "after loading the checkpoint"
        )
        if not self._hy_use_prope_blocks:
            raise RuntimeError(
                "HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache requires "
                "use_prope_blocks=True; the prefill executor only meaningfully "
                "writes the dual-branch memory caches owned by HyWorldPlayPRoPEBlock."
            )

        batch_shape = x.shape[:-2]
        L = x.shape[-2]

        if self.patch_embedding_type == "linear":
            x = self.patch_embedding(x)
        elif self.patch_embedding_type == "conv3d":
            _weight = self.patch_embedding.weight.reshape(self.dim, -1)
            _bias = self.patch_embedding.bias
            x = torch.nn.functional.linear(x, _weight, _bias)
        else:
            raise ValueError(
                f"Invalid patch embedding type: {self.patch_embedding_type}"
            )

        # Same per-token timestep dispatch + action injection as forward.
        per_token_timestep = (
            timesteps.ndim > len(batch_shape) and timesteps.shape[-1] == L
        )
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timesteps).type_as(x)
        )
        if action is not None:
            action_e = self._compute_action_embedding(action=action, x=x, L=L)
            e = e + action_e
            per_token_timestep = True
        e0 = self.time_projection(e).unflatten(-1, (6, self.dim))
        if per_token_timestep:
            block_e_shape = batch_shape + (L, 6, self.dim)
        else:
            block_e_shape = batch_shape + (6, self.dim)
        block_e = torch.broadcast_to(e0, block_e_shape)

        if viewmats is None:
            raise ValueError(
                "HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache requires "
                "viewmats; the executor must slice the per-rollout viewmats "
                "by selected_frame_indices before calling."
            )

        # No before_update / after_update on the per-block rolling caches
        # here -- the prefill writes only into cache[block_idx].memory,
        # which has its own reset / write cycle owned by the executor.
        from hy_worldplay import _debug_dump

        for block_idx, block in enumerate(self.blocks):
            block_cache = cache[block_idx]
            # Only PRoPE blocks have a memory slot; the prefill executor
            # caller already validates use_prope_blocks=True so this is a
            # defensive isinstance check rather than a hot branch.
            from hy_worldplay._camera import (
                HyWorldPlayPRoPEBlock,
                HyWorldPlayPRoPEBlockCache,
            )

            assert isinstance(block, HyWorldPlayPRoPEBlock), (
                f"prefill expects HyWorldPlayPRoPEBlock, got {type(block).__name__}"
            )
            assert isinstance(block_cache, HyWorldPlayPRoPEBlockCache), (
                f"prefill expects HyWorldPlayPRoPEBlockCache, got "
                f"{type(block_cache).__name__}"
            )
            _debug_dump.set_context(phase="prefill", block_idx=block_idx)
            # Phase 2b.6.2 -- ``prefill_memory_kv`` now runs the FULL
            # block (self-attn writes ``cache.memory`` *and* returns
            # the attention output that feeds cross-attn + FFN), so the
            # evolving hidden state propagates block-to-block exactly
            # like vendor's ``is_cache=True`` forward. The final-block
            # return value is intentionally discarded -- nothing past
            # the last block reads it on the prefill path (no head,
            # no output projection); only the per-block ``cache.memory``
            # side effects matter for the subsequent chunk's forward.
            x = block.prefill_memory_kv(
                x=x,
                e=block_e,
                rope_freqs=rope_freqs,
                viewmats=viewmats,
                Ks=Ks,
                cache=block_cache,
            )
        _debug_dump.clear_context("phase", "block_idx")

    def _compute_action_embedding(
        self,
        *,
        action: Tensor,
        x: Tensor,
        L: int,
    ) -> Tensor:
        """Lift per-latent-frame action labels to a per-token additive term.

        Sinusoidally encodes the integer labels, runs them through the
        zero-residual MLP, then ``repeat_interleave``s the resulting
        per-frame embedding across the ``tokens_per_frame`` slots of each
        latent frame in the post-patchify token axis. Multi-rank CP is
        gated off here (PRoPE in phase 2b.4 lifts that restriction).
        """
        cp_group = getattr(self, "_cp_group", None)
        if cp_group is not None:
            raise NotImplementedError(
                "HyWorldPlayWanDiTNetwork does not yet support context-parallel "
                "(cp_size > 1) action conditioning; this is enabled together "
                "with PRoPE in phase 2b.4."
            )
        n_latent = action.shape[-1]
        if L % n_latent != 0:
            raise ValueError(
                f"action.shape[-1]={n_latent} must divide the post-patchify "
                f"token count L={L}."
            )
        tokens_per_frame = L // n_latent
        action_freq = sinusoidal_embedding_1d(self.freq_dim, action).type_as(x)
        action_e = self.action_embedding(action_freq)
        return action_e.repeat_interleave(tokens_per_frame, dim=-2)


## ---------------------------------------------------------------------------
## Action-aware Wan 2.1 transformer
## ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class HyWorldPlayWan21TransformerCache(Wan21TransformerCache):
    """Per-rollout cache for the HY-WorldPlay transformer (phase 2b.5b-part2).

    Adds three reconstituted-context state slots on top of the standard
    :class:`Wan21TransformerCache`:

    * ``clean_latent_history`` -- per-rollout patchified clean latents
      from past chunks, concatenated along the post-patchify token axis
      ``[..., total_L, in_dim]`` where ``total_L = sum_i L_chunk_i``.
      Built up by :meth:`HyWorldPlayWan21Transformer.finalize_kv_cache`
      and read by :meth:`HyWorldPlayWan21Transformer.prefill_memory_kv_cache`
      to slice the K selected memory frames at the start of every
      chunk after the first.
    * ``finished_chunks`` -- count of chunks already appended to the
      history. Used by sanity assertions and by the prefill executor
      to decide whether memory is available.
    * ``hy_chunk_size_t`` -- pre-patchify temporal chunk size for the
      *current rollout*, cached at ``initialize_autoregressive_cache``
      time so the prefill / finalize paths can convert frame indices
      to token offsets without re-deriving from the network config.

    Two semantics overrides:

    * :meth:`start` resets the per-block rolling self-attention caches
      at the start of every chunk past the first. Standard mode rolls
      the window across chunks; HY mode pushes that cross-chunk K / V
      into the dedicated memory cache instead, so the rolling window
      only ever holds the *current* chunk's tokens.
    * The history is *not* automatically wiped between rollouts; the
      pipeline rebuilds the cache via
      :meth:`HyWorldPlayWan21Transformer.initialize_autoregressive_cache`
      for each new rollout, which produces a fresh empty history.
    """

    clean_latent_history: Tensor | None = None
    """Per-rollout patchified clean-latent history, concatenated along the
    post-patchify token axis (``dim=-2``). ``None`` until the first
    chunk's :meth:`HyWorldPlayWan21Transformer.finalize_kv_cache` call
    appends to it."""

    finished_chunks: int = 0
    """Count of chunks whose patchified clean latent has been appended to
    :attr:`clean_latent_history`. Equals ``current_chunk_idx`` at
    chunk-start time on the HY path."""

    hy_chunk_size_t: int = 0
    """Pre-patchify temporal chunk size (``len_t``) for the current rollout.
    Cached so the prefill executor can map per-frame indices to per-
    token offsets without re-reading the transformer config (which it
    doesn't have a handle to)."""

    hy_tokens_per_frame: int = 0
    """Post-patchify tokens per latent frame, ``= (height // kh) * (width // kw)``.
    Cached for the same reason as :attr:`hy_chunk_size_t`."""

    prefill_completed_for_chunk: int = -1
    """``autoregressive_index`` of the chunk for which the reconstituted-context
    KV prefill has already run, or ``-1`` if no prefill has run yet on this
    rollout. The HY transformer reads this in
    :meth:`HyWorldPlayWan21Transformer.predict_flow` to skip redundant
    prefill calls on the 2nd / 3rd / 4th denoising step of a chunk
    (``predict_flow`` is called once per scheduler step but the prefill
    K / V are stable across the chunk, so one call per chunk suffices).

    The previous build relied on
    ``cache.network_cache.block_caches[0].self_attn._n_cached == 0`` as
    a "step 0" signal, but that signal only flips at chunk *finalize*
    when ``eager_mode=False`` (the WAN-2.1 fast path -- ``before_update`` /
    ``after_update`` are hoisted out of the network forward, see
    :meth:`Wan21Transformer.start` line 89). Within a chunk
    ``_n_cached`` therefore stays at 0 across all scheduler steps and
    the old check returned ``True`` every step, re-running the
    prefill 4x per chunk. The writes are idempotent so this was a
    perf bug, not a correctness bug -- but each redundant prefill
    pays one full memory-token forward through every block, so it
    inflates HY rollouts measurably. Diagnosed via dump diff in
    2b.6.2 (4x ``prefill.entry`` records vs vendor's 1).
    """

    def start(self, autoregressive_index: int) -> None:
        # On HY path, reset the per-block rolling self-attention caches
        # at the start of every chunk past the first; the dedicated
        # memory cache provides the cross-chunk context. ``before_update``
        # then runs against an empty cache so each chunk's denoising
        # starts with a clean window.
        if autoregressive_index > 0:
            self._reset_per_block_rolling_caches(autoregressive_index)
        # Reset the prefill latch so the next chunk's first
        # ``predict_flow`` call runs the prefill once.
        self.prefill_completed_for_chunk = -1
        super().start(autoregressive_index)

    def _reset_per_block_rolling_caches(self, autoregressive_index: int) -> None:
        """Wipe each block's ``self_attn`` / ``prope_self_attn`` for the new chunk.

        Importantly we *also* poke ``_prev_chunk_idx`` to ``autoregressive_index - 1``
        so the subsequent ``before_update(autoregressive_index)`` from
        :meth:`Wan21TransformerCache.start` accepts the transition (the
        cache's monotonic-chunk-index assertion would otherwise fire
        because ``reset()`` resets ``_prev_chunk_idx`` to ``-1``).
        """
        # Local import to avoid a top-level circular dep (camera imports
        # core attention which imports modules which... etc.).
        from hy_worldplay._camera import HyWorldPlayPRoPEBlockCache

        for net_cache in (self.network_cache, self.network_cache_uncond):
            if net_cache is None:
                continue
            for block_cache in net_cache.block_caches:
                if not isinstance(block_cache, HyWorldPlayPRoPEBlockCache):
                    continue
                block_cache.reset_current_chunk()
                # Pre-set ``_prev_chunk_idx`` so the upcoming
                # ``before_update(autoregressive_index)`` from the parent
                # ``start`` accepts ``chunk_idx == _prev_chunk_idx + 1``.
                block_cache.self_attn._prev_chunk_idx = autoregressive_index - 1
                block_cache.prope_self_attn._prev_chunk_idx = (
                    autoregressive_index - 1
                )


@dataclass(kw_only=True)
class HyWorldPlayWan21TransformerConfig(Wan21TransformerConfig):
    """Config for the action-aware Wan 2.1 transformer.

    Same set of knobs as :class:`Wan21TransformerConfig`; the only delta is
    that the default ``network`` is the action-aware DiT and the
    ``_target`` builds :class:`HyWorldPlayWan21Transformer` so the
    ``predict_flow`` + ``patchify_and_maybe_split_cp`` overrides take
    effect.
    """

    _target: type = field(default_factory=lambda: HyWorldPlayWan21Transformer)

    network: HyWorldPlayWanDiTNetworkConfig = field(
        default_factory=HyWorldPlayWanDiTNetworkConfig
    )


class HyWorldPlayWan21Transformer(Wan21Transformer):
    """Wan 2.1 transformer that threads action / camera / memory through the network.

    Five overrides on top of :class:`Wan21Transformer`:

    * :meth:`predict_flow` reads ``input.action`` / ``input.viewmats`` /
      ``input.Ks`` (when ``input`` is a :class:`HyWorldPlayCtrl`) and
      forwards them through ``network_extra_kwargs`` so they reach
      :meth:`HyWorldPlayWanDiTNetwork.forward`. At the *start* of each
      chunk's denoising loop (the per-AR-step entry that the diffusion
      model calls before any noise prediction step) it also runs the
      reconstituted-context prefill executor when memory frames are
      selected, populating each block's :class:`HyWorldPlayMemoryKVCache`
      from the cached clean-latent history.
    * :meth:`patchify_and_maybe_split_cp` preserves the
      ``action`` / ``viewmats`` / ``Ks`` / ``memory_frame_indices`` slice
      across the in-place patchify pass that the base implementation
      otherwise rebuilds via ``I2VCtrl(...)``, which would drop subclass
      fields. These per-frame metadata tensors do not participate in
      patchify themselves and are passed through unchanged.
    * :meth:`initialize_autoregressive_cache` returns a
      :class:`HyWorldPlayWan21TransformerCache` so the per-rollout
      cache carries the clean-latent history slot needed by the
      prefill executor.
    * :meth:`finalize_kv_cache` appends the completed chunk's clean
      latent to ``cache.clean_latent_history`` and skips the parent's
      ``predict_flow`` re-run -- standard mode uses that re-run to
      stamp the clean K / V into the rolling window, but HY mode wipes
      the rolling window at every chunk start so the re-run would be
      pure waste.
    * :meth:`prefill_memory_kv_cache` (new public method) is the
      transformer-level driver behind the prefill: it picks memory
      frames out of the clean-latent history, slices viewmats / Ks
      / action accordingly, builds RoPE freqs for the collapsed
      positions ``[0, K * tokens_per_frame)``, and dispatches into
      :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache`.
    """

    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor | None = None,
        negative_text_embeddings: Tensor | None = None,
        **_unused: Any,
    ) -> HyWorldPlayWan21TransformerCache:
        """Build a :class:`HyWorldPlayWan21TransformerCache` for a new rollout.

        Same contract as :meth:`Wan21Transformer.initialize_autoregressive_cache`
        but returns the HY subclass so the clean-latent-history slot is
        available to the prefill / finalize hooks. Per-rollout spatial
        layout (``height`` / ``width``) is also stamped into the cache
        as ``hy_tokens_per_frame`` so the prefill executor can map
        memory-frame indices to post-patchify token ranges without
        re-deriving them.
        """
        base = super().initialize_autoregressive_cache(
            height=height,
            width=width,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            negative_text_embeddings=negative_text_embeddings,
            **_unused,
        )
        cfg = self.config
        kt, kh, kw = cfg.network.patch_size
        # tokens_per_frame uses the *post-patchify* spatial layout: each
        # latent frame contributes ``(height // kh) * (width // kw)``
        # tokens after the patchify rearrange, regardless of how many
        # latent frames the rollout pre-patchifies into one AR chunk
        # (which is governed by kt and len_t together).
        tokens_per_frame = (height // kh) * (width // kw)
        return HyWorldPlayWan21TransformerCache(
            network_cache=base.network_cache,
            network_cache_uncond=base.network_cache_uncond,
            rope_adapter=base.rope_adapter,
            rope_freqs=base.rope_freqs,
            autoregressive_index=base.autoregressive_index,
            clean_latent_history=None,
            finished_chunks=0,
            hy_chunk_size_t=cfg.len_t // kt,
            hy_tokens_per_frame=tokens_per_frame,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: I2VCtrl | None = None,
        network_extra_kwargs: dict[str, Any] | None = None,
    ) -> Tensor:
        from hy_worldplay import _debug_dump

        ar_idx = (
            cache.autoregressive_index
            if hasattr(cache, "autoregressive_index")
            else -1
        )
        is_first_step = (
            isinstance(cache, HyWorldPlayWan21TransformerCache)
            and cache.prefill_completed_for_chunk != ar_idx
        )
        # Bind chunk + step context so per-block dumps below carry it.
        _debug_dump.set_context(
            ar_idx=ar_idx,
            is_first_step_of_chunk=is_first_step,
        )
        if _debug_dump.enabled():
            cfg_self = getattr(self, "config", None)
            extra_cfg = {}
            if cfg_self is not None:
                extra_cfg = {
                    "cfg_len_t": getattr(cfg_self, "len_t", None),
                    "cfg_window_size_t": getattr(cfg_self, "window_size_t", None),
                    "cfg_batch_shape": list(getattr(cfg_self, "batch_shape", ())),
                    "cfg_patch_size": list(getattr(cfg_self.network, "patch_size", ())),
                    "_cp_size": getattr(self, "_cp_size", None),
                    "_output_height": getattr(self, "_output_height", None),
                    "_output_width": getattr(self, "_output_width", None),
                }
            _debug_dump.dump(
                "predict_flow.entry",
                None,
                timestep_shape=list(timestep.shape),
                **extra_cfg,
            )
            _debug_dump.dump("predict_flow.noisy_latent", noisy_latent)
            _debug_dump.dump("predict_flow.timestep", timestep)
        network_extra_kwargs = dict(network_extra_kwargs or {})
        # Run the reconstituted-context prefill at the very first
        # denoising step of every chunk past the first, when memory
        # frames have been selected. We detect "first denoising step
        # of the chunk" via cache.network_cache.block_caches[0].self_attn
        # being in the empty-filling state (n_cached == 0): cache.start
        # has just reset / pre-set _prev_chunk_idx, but no prediction
        # has populated the rolling window yet. This avoids re-running
        # the prefill on every scheduler step within the chunk.
        if (
            isinstance(cache, HyWorldPlayWan21TransformerCache)
            and isinstance(input, HyWorldPlayCtrl)
            and input.memory_frame_indices is not None
            and len(input.memory_frame_indices) > 0
            and cache.clean_latent_history is not None
            and cache.prefill_completed_for_chunk != ar_idx
        ):
            self.prefill_memory_kv_cache(cache=cache, input=input, timestep=timestep)
            cache.prefill_completed_for_chunk = ar_idx

        if isinstance(input, HyWorldPlayCtrl):
            if input.action is not None and "action" not in network_extra_kwargs:
                network_extra_kwargs["action"] = input.action
            if (
                input.viewmats is not None
                and "viewmats" not in network_extra_kwargs
            ):
                network_extra_kwargs["viewmats"] = input.viewmats
            if input.Ks is not None and "Ks" not in network_extra_kwargs:
                network_extra_kwargs["Ks"] = input.Ks
        return super().predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input,
            network_extra_kwargs=network_extra_kwargs,
        )

    def finalize_kv_cache(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: Wan21TransformerCache,
        input: Any = None,
    ) -> None:
        """Append the completed chunk's clean latent to the history; skip rolling-cache update.

        Standard ``Wan21Transformer.finalize_kv_cache`` re-runs the
        network at the context-noise timestep so the rolling KV cache
        gets the clean K / V baked in for the next chunk's attention.
        On the HY path we instead reset the rolling cache at every
        chunk start (see :meth:`HyWorldPlayWan21TransformerCache.start`)
        and provide cross-chunk context via the dedicated memory
        cache, so this re-run is wasted work; we skip it and just
        cache the patchified clean latent for the next chunk's
        prefill executor to slice.
        """
        if isinstance(cache, HyWorldPlayWan21TransformerCache):
            cache.clean_latent_history = self._append_clean_latent_to_history(
                history=cache.clean_latent_history,
                clean_latent=noisy_latent,
            )
            cache.finished_chunks += 1
            return
        # Fall back to base behaviour for any non-HY cache type that
        # might somehow get plumbed through (defensive; the runner
        # always builds an HY cache when this transformer is in use).
        super().finalize_kv_cache(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            input=input,
        )

    def patchify_and_maybe_split_cp(self, x: Any) -> Any:
        if isinstance(x, HyWorldPlayCtrl):
            if x._is_patchified:
                return x
            patched_latent = self.patchify_and_maybe_split_cp(x.latent)
            patched_mask = self.patchify_and_maybe_split_cp(x.mask)
            # action / viewmats / Ks / memory_frame_indices and the
            # per-rollout rollout_* siblings are per-latent-frame
            # metadata (not per-token tensors) and do not participate
            # in the patchify reshape; pass them through unchanged so
            # the PRoPE branch and the memory-prefill consumer see the
            # same ``[..., len_t, 4, 4]`` /
            # ``[..., F_total, 4, 4]`` / ``[..., len_t]`` /
            # ``[..., F_total]`` / ``list[int]`` layouts they would on
            # a fresh ctrl.
            return HyWorldPlayCtrl(
                latent=patched_latent,
                mask=patched_mask,
                _is_patchified=True,
                action=x.action,
                viewmats=x.viewmats,
                Ks=x.Ks,
                memory_frame_indices=x.memory_frame_indices,
                rollout_viewmats=x.rollout_viewmats,
                rollout_Ks=x.rollout_Ks,
                rollout_action=x.rollout_action,
            )
        return super().patchify_and_maybe_split_cp(x)

    ## ---- Reconstituted-context prefill driver (phase 2b.5b-part2) ----

    def prefill_memory_kv_cache(  # noqa: C901 (debug instrumentation)
        self,
        cache: HyWorldPlayWan21TransformerCache,
        input: HyWorldPlayCtrl,
        timestep: Tensor,
    ) -> None:
        """Drive the reconstituted-context KV prefill for the current chunk.

        Mirrors upstream's ``model(..., is_cache=True)`` invocation in
        ``wan/inference/pipeline_wan_w_mem_relative_rope.py`` lines
        922-937: at the start of every chunk past the first, slice the
        clean-latent history at the selected memory frame indices,
        build collapsed-position RoPE freqs, slice viewmats / Ks /
        action accordingly, and call into
        :meth:`HyWorldPlayWanDiTNetwork.prefill_memory_kv_cache` for
        each network branch (cond + uncond if CFG is on). Each block's
        :class:`HyWorldPlayMemoryKVCache` is reset before being
        repopulated.

        Args:
            cache: The active per-rollout cache. ``clean_latent_history``
                must already contain ``finished_chunks * L_chunk`` tokens
                (asserted defensively below).
            input: The patchified per-AR-step ctrl payload.
                ``input.memory_frame_indices`` must be a non-empty list.
            timestep: The current denoising step's timestep tensor.
                Used only for its ``dtype`` / ``device``; the memory
                positions are modulated at the clean-context timestep
                :data:`_HY_STABILIZATION_TIMESTEP` instead, mirroring
                vendor's ``t_ctx = stabilization_level - 1`` in
                ``pipeline_wan_w_mem_relative_rope.py`` line 883-887
                (and the same constant in the ``use_kv_cache=True``
                cache-prefill branch at line 908-913, where
                ``t_cache = timestep[:, selected_frame_indices]``
                resolves to 14 for the chunk-0 memory positions).
                The previous build of this driver forwarded the
                noisy ``timestep`` directly, which made the
                attention scale memory K / V as if the chunk-0
                outputs were still noisy and caused the chunk-1
                denoising blow-up that 2b.6 was chasing.
        """
        assert input.memory_frame_indices is not None, (
            "prefill_memory_kv_cache requires non-None memory_frame_indices"
        )
        selected = list(input.memory_frame_indices)
        K = len(selected)
        assert K > 0, "prefill_memory_kv_cache requires at least one memory frame"
        assert cache.clean_latent_history is not None, (
            "prefill_memory_kv_cache requires clean_latent_history; the executor "
            "must run after at least one chunk has finalized."
        )

        tokens_per_frame = cache.hy_tokens_per_frame
        history = cache.clean_latent_history  # [..., total_L, in_dim]
        total_L = history.shape[-2]
        max_frame = total_L // tokens_per_frame
        # Defensive: every selected index must be in range. The
        # encoder's selection algorithm only emits indices < the
        # current frame counter, but we cross-check against the
        # actual history length here in case of plumbing bugs.
        for idx in selected:
            assert 0 <= idx < max_frame, (
                f"memory frame index {idx} out of range for history of "
                f"{max_frame} frames ({total_L} tokens / {tokens_per_frame} "
                f"tokens-per-frame)."
            )

        # Slice the history at the per-frame token ranges. The history
        # is laid out (frame, h, w, ...) flattened along the token axis,
        # so frame ``idx`` occupies tokens ``[idx*tokens_per_frame,
        # (idx+1)*tokens_per_frame)``.
        token_ranges = [
            history[..., idx * tokens_per_frame : (idx + 1) * tokens_per_frame, :]
            for idx in selected
        ]
        memory_x = torch.cat(token_ranges, dim=-2)  # [..., K*TPF, in_dim]

        # Slice the per-rollout camera + action tensors at the same
        # frame indices. These tensors are stored at the
        # *latent-frame* granularity, not the token granularity --
        # one entry per latent frame. The encoder hands us the
        # *rollout-scoped* buffers via ``input.rollout_*``; if those
        # are missing we fall back to the (parity-incorrect, but
        # structurally safe) per-AR-step truncation that 2b.5b-part2
        # used as a stub. The fallback is only reached when the
        # encoder hasn't bound camera / action data, in which case
        # the dual-branch / action paths are themselves no-ops, so
        # the slice values don't matter; the assertion below makes
        # the parity-correct path observable in tests.
        selected_idx_t = torch.as_tensor(
            selected, dtype=torch.long, device=memory_x.device
        )
        memory_viewmats = self._index_rollout_buffer(
            rollout=input.rollout_viewmats,
            per_step=input.viewmats,
            selected=selected_idx_t,
            kind="viewmats",
        )
        memory_Ks = self._index_rollout_buffer(
            rollout=input.rollout_Ks,
            per_step=input.Ks,
            selected=selected_idx_t,
            kind="Ks",
        )
        memory_action = self._index_rollout_buffer(
            rollout=input.rollout_action,
            per_step=input.action,
            selected=selected_idx_t,
            kind="action",
        )

        # Build RoPE freqs for the collapsed memory positions
        # ``[0, K)`` at the *temporal* axis. Mirrors upstream lines
        # 914-915: ``rotary_emb[:, :, 0:current_end, :]`` where
        # current_end = K * tokens_per_frame and the inner positions
        # come from a fresh-zeroed time grid.
        rope_freqs = self._build_collapsed_rope_freqs(
            cache=cache,
            t_positions=torch.arange(
                K, dtype=torch.float32, device=memory_x.device
            ),
        )

        # Build the clean-context timestep tensor that the memory
        # positions get modulated at, mirroring vendor's
        # ``t_ctx = stabilization_level - 1`` (see the docstring and
        # :data:`_HY_STABILIZATION_TIMESTEP` for the full reference).
        # Match ``timestep``'s dtype / device / batch so the network's
        # ``sinusoidal_embedding_1d(...).type_as(x)`` path stays on
        # the same compute graph.
        context_timestep = torch.full_like(
            timestep, fill_value=_HY_STABILIZATION_TIMESTEP
        )

        # Phase 2b.6.2 debug dump (env-var-gated; see _debug_dump.py).
        # Captures the inputs to the prefill executor at chunk-1+ so
        # they can be diffed against vendor's matched call site.
        from hy_worldplay import _debug_dump

        if _debug_dump.enabled():
            _debug_dump.dump(
                "prefill.entry",
                None,
                selected=list(selected),
                K=K,
                tokens_per_frame=tokens_per_frame,
                stabilization_timestep=_HY_STABILIZATION_TIMESTEP,
            )
            _debug_dump.dump("prefill.memory_x", memory_x)
            _debug_dump.dump("prefill.rope_freqs", rope_freqs)
            _debug_dump.dump("prefill.context_timestep", context_timestep)
            _debug_dump.dump("prefill.timestep_input", timestep)
            if memory_viewmats is not None:
                _debug_dump.dump("prefill.memory_viewmats", memory_viewmats)
            if memory_Ks is not None:
                _debug_dump.dump("prefill.memory_Ks", memory_Ks)
            if memory_action is not None:
                _debug_dump.dump("prefill.memory_action", memory_action)

        # Run the prefill on whichever network branches are active.
        # Each branch has its own per-block memory cache; reset
        # *before* the prefill so a previous chunk's leftover
        # content can't leak into the new memory image.
        from hy_worldplay._camera import HyWorldPlayPRoPEBlockCache

        for net_cache in (cache.network_cache, cache.network_cache_uncond):
            if net_cache is None:
                continue
            for block_cache in net_cache.block_caches:
                if isinstance(block_cache, HyWorldPlayPRoPEBlockCache):
                    block_cache.memory.reset()

        # Conditional pass.
        self.network.prefill_memory_kv_cache(
            x=memory_x,
            timesteps=context_timestep,
            cache=cache.network_cache,
            rope_freqs=rope_freqs,
            action=memory_action,
            viewmats=memory_viewmats,
            Ks=memory_Ks,
        )
        # Unconditional pass (when CFG is enabled).
        if cache.network_cache_uncond is not None:
            self.network.prefill_memory_kv_cache(
                x=memory_x,
                timesteps=context_timestep,
                cache=cache.network_cache_uncond,
                rope_freqs=rope_freqs,
                action=memory_action,
                viewmats=memory_viewmats,
                Ks=memory_Ks,
            )

    def _append_clean_latent_to_history(
        self,
        history: Tensor | None,
        clean_latent: Tensor,
    ) -> Tensor:
        """Concat the just-finalized chunk's patchified clean latent into the history.

        Always detaches first (the history outlives the autograd graph
        of the chunk that produced it; even on inference we want a
        clean ``Tensor`` so downstream slices are safe). Concatenation
        is along the post-patchify token axis (``dim=-2``) which is
        equivalent to "next frames added at the end of the rolling
        latent volume".
        """
        if history is None:
            return clean_latent.detach().clone()
        return torch.cat([history, clean_latent.detach()], dim=-2)

    def _index_rollout_buffer(
        self,
        *,
        rollout: Tensor | None,
        per_step: Tensor | None,
        selected: Tensor,
        kind: str,
    ) -> Tensor | None:
        """Slice a per-rollout metadata buffer at the selected memory-frame indices.

        Prefers the rollout-scoped buffer (``rollout``) populated by
        :meth:`HyWorldPlayWanCtrlEncoder.forward`; falls back to the
        per-AR-step ``per_step`` slice when the rollout buffer is
        ``None`` (encoder not configured for this conditioner). The
        fallback is parity-incorrect -- it indexes into the *current
        chunk's* slice rather than the full rollout -- but it is also
        structurally safe: when the rollout buffer is absent, the
        corresponding conditioner is itself disabled (no
        ``set_camera_data`` / ``set_action_labels`` call has run), so
        the prefill executor's downstream consumer (the network's
        AdaLN / PRoPE math) treats the slice as a no-op.

        Args:
            rollout: Per-rollout buffer with the *full* trajectory's
                worth of frames (e.g. ``[*batch_shape, F_total, 4, 4]``
                for viewmats). When non-``None``, indexed at
                ``selected`` along the frame axis -- this is the
                parity-correct path.
            per_step: Per-AR-step slice (``[*batch_shape, len_t, ...]``
                for matrices, ``[*batch_shape, len_t]`` for action).
                Used as a fallback when ``rollout`` is missing; in
                practice this only happens when the conditioner is
                disabled, so its content is not consumed.
            selected: ``LongTensor`` of memory frame indices, shape
                ``[K]``, in *rollout* coordinates (``0 <= idx <
                F_total``).
            kind: Tensor kind name (``"viewmats"`` /  ``"Ks"`` /
                ``"action"``) for the error message; reduces the cost
                of debugging mis-bound buffers in production rollouts.

        Returns:
            Indexed tensor with shape ``[*batch_shape, K, ...]`` for
            matrices or ``[*batch_shape, K]`` for action; ``None`` if
            both ``rollout`` and ``per_step`` are ``None``.
        """
        if rollout is None and per_step is None:
            return None
        if rollout is None:
            # Conditioner is disabled but the prefill is still running
            # because some other conditioner is on. Return the per-
            # step slice unmodified -- ``selected`` won't be applied.
            # The downstream prefill consumer treats this slice as a
            # no-op per the conditioner's own gate.
            return per_step

        # Rollout buffer present: index along the frame axis. Action
        # is rank ``len(batch_shape)+1`` (last axis is the frame); the
        # matrix tensors (viewmats / Ks) are rank ``len(batch_shape)+3``
        # with the matrix axes at -2 / -1 and the frame axis at -3.
        if rollout.dtype in (torch.int32, torch.int64):
            # action: ``[*batch_shape, F_total]`` -> ``[*batch_shape, K]``.
            # The selected indices live in rollout coordinates so a
            # straight ``index_select`` on the trailing axis is what
            # we want; this is also what upstream's
            # ``action_chunk[..., selected_frame_indices]`` produces
            # in ``arwan_w_action_w_mem_relative_rope.py``.
            if rollout.shape[-1] == 0:
                # Defensive: a zero-length rollout would never reach
                # us (the encoder would have raised) but the empty
                # selection branch below would still produce a degenerate
                # tensor. Surface the misconfig instead.
                raise ValueError(
                    f"rollout {kind} buffer has zero-length frame axis"
                )
            return rollout.index_select(-1, selected)
        # matrices: index on axis -3 (the F axis).
        if rollout.ndim < 3 or rollout.shape[-3] == 0:
            raise ValueError(
                f"rollout {kind} buffer must have shape "
                f"[..., F_total, M, N] with F_total > 0; got "
                f"{tuple(rollout.shape)}"
            )
        return rollout.index_select(-3, selected)

    def _build_collapsed_rope_freqs(
        self,
        cache: HyWorldPlayWan21TransformerCache,
        t_positions: Tensor,
    ) -> Tensor:
        """Compute RoPE frequencies for arbitrary temporal positions.

        Phase 2b.5b-part2: the prefill executor needs RoPE freqs at
        the collapsed memory positions ``[0, K)`` (and, in a future
        iteration, at the current chunk's offset positions
        ``[K, K + len_t)``). The base
        :class:`flashdreams.core.attention.rope.RotaryPositionEmbedding3D`
        only exposes ``shift_t(autoregressive_index)`` which always
        produces freqs at ``[c*len_t, (c+1)*len_t)`` for a chunk
        index ``c``. To get arbitrary positions we use the
        ``_freq_components(seq_t)`` primitive that ``shift_t`` itself
        is built on, then concat into the standard cat layout.

        The leading ``_`` does not signal API instability for our
        purposes -- ``_freq_components`` is the documented internal
        builder and is the only way to construct freqs at non-chunk-
        aligned positions today. If upstream flashdreams later
        promotes a public ``freqs_for_positions`` we will swap to
        that.
        """
        rope = cache.rope_adapter
        from flashdreams.core.attention.rope import RotaryPositionEmbedding3D

        if not isinstance(rope, RotaryPositionEmbedding3D):
            raise NotImplementedError(
                f"Reconstituted-context prefill currently supports only "
                f"RotaryPositionEmbedding3D; got {type(rope).__name__}. "
                f"KVCacheRelativeRotaryPositionEmbedding3D support lands "
                f"with multi-resolution / extended-window rollouts."
            )
        if rope.is_context_parallel_enabled():
            raise NotImplementedError(
                "Reconstituted-context prefill does not yet support "
                "context-parallel; CP wiring lands with the multi-rank "
                "action expansion."
            )
        freqs_t, freqs_h, freqs_w = rope._freq_components(t_positions.to(rope.device))
        return rope._cat_freqs(freqs_t, freqs_h, freqs_w)

    def _is_first_step_of_chunk(
        self,
        cache: HyWorldPlayWan21TransformerCache,
    ) -> bool:
        """Detect "first denoising step of the current chunk" via the prefill latch.

        Reads the explicit
        :attr:`HyWorldPlayWan21TransformerCache.prefill_completed_for_chunk`
        latch -- ``True`` when ``cache.start`` has reset it to ``-1`` for
        the new chunk but no ``predict_flow`` call has yet incremented
        it to the current ``autoregressive_index``. The latch replaces
        the previous heuristic that read ``self_attn._n_cached``;
        ``_n_cached`` only flips at chunk *finalize* on the Wan-2.1
        ``eager_mode=False`` fast path
        (``before_update`` / ``after_update`` are hoisted out of
        the network forward; see
        :meth:`Wan21Transformer.start`), so the old check returned
        ``True`` on every scheduler step within a chunk -- causing the
        prefill executor to redundantly recompute the (deterministic)
        memory K / V on every step, 4x the necessary work. Diagnosed
        via dump diff in 2b.6.2.
        """
        return cache.prefill_completed_for_chunk != cache.autoregressive_index
