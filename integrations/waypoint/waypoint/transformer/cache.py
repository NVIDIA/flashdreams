# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sparse causal KV-history policy for the Waypoint transformer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask

from waypoint.spec import WAYPOINT_1_5, WaypointModelSpec


@dataclass(frozen=True, kw_only=True)
class WaypointAttentionPolicy:
    """Choose the causal history visible to one Waypoint attention block.

    Most blocks need dense short-term state for motion continuity.  Every
    fourth block instead sees a stride-sampled long history, which supplies
    long-range scene anchors without giving every block a 128-frame attention
    cost.  The selection is expressed in latent-frame indices so the KV store
    and a future fused attention backend share one unambiguous contract.
    """

    spec: WaypointModelSpec = WAYPOINT_1_5
    """Checkpoint architecture whose attention schedule is represented."""

    def is_global_layer(self, layer_index: int) -> bool:
        """Return whether ``layer_index`` uses sparse global history.

        Args:
            layer_index: Zero-indexed transformer block.

        Raises:
            ValueError: The layer index is outside the checkpoint's block range.
        """
        self._validate_layer_index(layer_index)
        return (
            layer_index - self.spec.global_attention_offset
        ) % self.spec.global_attention_period == 0

    def visible_frame_indices(
        self, *, layer_index: int, frame_index: int
    ) -> tuple[int, ...]:
        """Return causal latent frames visible while denoising ``frame_index``.

        A local block sees a dense trailing window.  A global block sees only
        pinned frames aligned to the configured dilation inside its longer
        trailing horizon; a non-pinned current frame is added so every action
        can attend to itself.

        Args:
            layer_index: Zero-indexed transformer block.
            frame_index: Zero-indexed latent-frame index of the current action.

        Returns:
            Strictly increasing, causal latent-frame indices.

        Raises:
            ValueError: ``frame_index`` is negative or the layer index is invalid.
        """
        self._validate_layer_index(layer_index)
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")

        if not self.is_global_layer(layer_index):
            first = max(0, frame_index - self.spec.local_window + 1)
            return tuple(range(first, frame_index + 1))

        first = max(0, frame_index - self.spec.global_window + 1)
        dilation = self.spec.global_pinned_dilation
        first_pinned = ((first + dilation - 1) // dilation) * dilation
        pinned = tuple(range(first_pinned, frame_index + 1, dilation))
        if pinned and pinned[-1] == frame_index:
            return pinned
        return (*pinned, frame_index)

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.spec.n_layers:
            raise ValueError(
                f"layer_index must be in [0, {self.spec.n_layers}), got {layer_index}"
            )


@dataclass(frozen=True, kw_only=True)
class WaypointKVView:
    """The selected key/value frames for one causal attention evaluation."""

    key: Tensor
    """Concatenated keys in ``[B, H_kv, selected_frames * S, d_h]`` layout."""

    value: Tensor
    """Concatenated values with the same layout as ``key``."""

    frame_indices: tuple[int, ...]
    """Latent-frame origin of each contiguous ``S``-token segment."""

    block_mask: BlockMask | None = None
    """Optional fixed-cache block mask for checkpoint-equivalent attention."""


@dataclass
class _FixedKVLayer:
    """One lazy fixed-capacity K/V store used by the published runtime."""

    kv: Tensor
    written: Tensor
    history_tokens: int
    pinned_dilation: int


@dataclass(kw_only=True)
class WaypointKVCache:
    """Keep exactly the model-visible KV history for each Waypoint block.

    A diffusion step may evaluate the same latent frame more than once.  An
    update for that frame replaces its provisional K/V tensors instead of
    extending history; advancing to a later frame evicts entries the sparse
    policy can no longer expose. The CUDA path uses a fixed-capacity store and
    block mask; CPU uses a compact dictionary representation for the same
    selection policy.
    """

    policy: WaypointAttentionPolicy = field(default_factory=WaypointAttentionPolicy)
    """Frame-selection policy shared by all transformer blocks."""

    use_fixed_attention: bool = False
    """Use the checkpoint runtime's fixed-capacity masked-attention semantics."""

    _layers: dict[int, dict[int, tuple[Tensor, Tensor]]] = field(default_factory=dict)
    """Stored K/V tensors indexed by block, then latent-frame index."""

    _latest_frame_indices: dict[int, int] = field(default_factory=dict)
    """Latest frame written per block; equal writes replace a diffusion provisional."""

    _fixed_layers: dict[int, _FixedKVLayer] = field(default_factory=dict)
    _fixed_frozen: bool = True

    def update(
        self, *, layer_index: int, frame_index: int, key: Tensor, value: Tensor
    ) -> WaypointKVView:
        """Store K/V for one action and return the model-visible sparse view.

        Args:
            layer_index: Zero-indexed transformer block that produced the tensors.
            frame_index: Zero-indexed latent action being evaluated.
            key: RoPE-applied keys in ``[B, H_kv, S, d_h]`` layout.
            value: Values in ``[B, H_kv, S, d_h]`` layout.

        Returns:
            Causally selected keys and values concatenated along their token axis.

        Raises:
            ValueError: Tensor layouts differ, a frame is negative, or a write
                attempts to move a layer's history backwards.
        """
        self._validate_kv(key, value)
        latest = self._latest_frame_indices.get(layer_index)
        if latest is not None and frame_index < latest:
            raise ValueError(
                f"layer {layer_index} cannot move from frame {latest} back to {frame_index}"
            )
        self._latest_frame_indices[layer_index] = frame_index

        if self.use_fixed_attention and key.device.type == "cuda":
            return self._fixed_view(
                layer_index=layer_index,
                frame_index=frame_index,
                key=key,
                value=value,
            )

        # CPU contract tests use this compact reference representation. The
        # CUDA rollout takes the fixed-cache path above and never duplicates
        # K/V tensors in a Python dictionary.
        visible = self.policy.visible_frame_indices(
            layer_index=layer_index, frame_index=frame_index
        )
        layer = self._layers.setdefault(layer_index, {})
        layer[frame_index] = (key, value)

        retained = {index: layer[index] for index in visible if index in layer}
        if frame_index not in retained:
            raise RuntimeError("current frame was not retained by its attention policy")
        self._layers[layer_index] = retained
        return self._view(layer_index=layer_index, frame_indices=visible)

    def reset(self) -> None:
        """Discard all retained K/V tensors while preserving the policy."""
        self._layers.clear()
        self._latest_frame_indices.clear()
        self._fixed_layers.clear()
        self._fixed_frozen = True

    def set_frozen(self, frozen: bool) -> None:
        """Choose whether writes update only the provisional current action."""
        self._fixed_frozen = frozen

    def _fixed_view(
        self,
        *,
        layer_index: int,
        frame_index: int,
        key: Tensor,
        value: Tensor,
    ) -> WaypointKVView:
        """Write one frame into the runtime-shaped fixed cache and mask it."""
        state = self._fixed_layers.get(layer_index)
        tokens_per_frame = key.shape[2]
        if state is None:
            global_layer = self.policy.is_global_layer(layer_index)
            frame_capacity = (
                self.policy.spec.global_window
                if global_layer
                else self.policy.spec.local_window
            )
            pinned_dilation = (
                self.policy.spec.global_pinned_dilation if global_layer else 1
            )
            history_tokens = frame_capacity * tokens_per_frame
            capacity = history_tokens + tokens_per_frame
            state = _FixedKVLayer(
                kv=torch.zeros(
                    2,
                    key.shape[0],
                    key.shape[1],
                    capacity,
                    key.shape[-1],
                    device=key.device,
                    dtype=key.dtype,
                ),
                written=torch.cat(
                    (
                        torch.zeros(
                            history_tokens, device=key.device, dtype=torch.bool
                        ),
                        torch.ones(
                            tokens_per_frame, device=key.device, dtype=torch.bool
                        ),
                    )
                ),
                history_tokens=history_tokens,
                pinned_dilation=pinned_dilation,
            )
            self._fixed_layers[layer_index] = state

        bucket_count = state.history_tokens // tokens_per_frame // state.pinned_dilation
        bucket = (frame_index + state.pinned_dilation - 1) // state.pinned_dilation
        ring_start = (bucket % bucket_count) * tokens_per_frame
        ring_slice = slice(ring_start, ring_start + tokens_per_frame)
        tail_slice = slice(
            state.history_tokens, state.history_tokens + tokens_per_frame
        )
        current = torch.stack((key, value))
        state.kv[..., tail_slice, :].copy_(current)

        write_step = frame_index % state.pinned_dilation == 0
        visible = state.written.clone()
        visible[ring_slice] &= not write_step
        block_mask = _fixed_block_mask(tokens_per_frame, visible)

        if not self._fixed_frozen:
            destination = ring_slice if write_step else tail_slice
            state.kv[..., destination, :].copy_(current)
            state.written[destination] = True

        key_full, value_full = state.kv.unbind(0)
        return WaypointKVView(
            key=key_full,
            value=value_full,
            frame_indices=(frame_index,),
            block_mask=block_mask,
        )

    def _view(
        self, *, layer_index: int, frame_indices: tuple[int, ...]
    ) -> WaypointKVView:
        layer = self._layers[layer_index]
        selected = tuple(index for index in frame_indices if index in layer)
        if not selected:
            raise RuntimeError(f"layer {layer_index} has no K/V tensors to attend to")
        keys, values = zip(*(layer[index] for index in selected), strict=True)
        return WaypointKVView(
            key=torch.cat(keys, dim=2),
            value=torch.cat(values, dim=2),
            frame_indices=selected,
        )

    @staticmethod
    def _validate_kv(key: Tensor, value: Tensor) -> None:
        if key.ndim != 4:
            raise ValueError(
                f"key must have [B, H_kv, S, d_h] layout, got {tuple(key.shape)}"
            )
        if value.shape != key.shape:
            raise ValueError(
                "value must have the same [B, H_kv, S, d_h] shape as key, got "
                f"{tuple(value.shape)} and {tuple(key.shape)}"
            )


def _fixed_block_mask(tokens_per_frame: int, written: Tensor) -> BlockMask:
    """Create one visibility contract for compiled and eager attention.

    ``full_kv_*`` is the compiled kernel's efficient representation of active
    blocks.  ``mask_mod`` independently states the same per-token rule for
    implementations that materialize dense scores.  Keeping both prevents a
    fixed-capacity cache from making unwritten zero slots visible outside the
    compiled path.
    """
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE
    if tokens_per_frame % block_size or written.numel() % block_size:
        raise ValueError("Waypoint fixed attention requires block-aligned cache sizes")
    query_blocks = tokens_per_frame // block_size
    key_blocks = written.numel() // block_size
    visible = written.view(key_blocks, block_size)
    if not torch.equal(visible.any(-1), visible.all(-1)):
        raise RuntimeError("Waypoint fixed cache visibility must be block aligned")
    active = visible.all(-1).nonzero(as_tuple=False).flatten().to(torch.int32)
    # ``BlockMask`` stores a key-block list for *each* query block.  Every
    # Waypoint query token sees the same cache slots, but omitting the query
    # axis makes the mask structurally different and leaves the attention
    # backend to interpret a malformed index tensor.
    full_indices = torch.zeros(
        1, 1, query_blocks, key_blocks, dtype=torch.int32, device=written.device
    )
    full_indices[..., : active.numel()] = active
    full_count = torch.full(
        (1, 1, query_blocks),
        active.numel(),
        dtype=torch.int32,
        device=written.device,
    )
    empty_count = torch.zeros_like(full_count)
    empty_indices = torch.zeros(
        1, 1, query_blocks, key_blocks, dtype=torch.int32, device=written.device
    )

    def visible_slot(
        batch_index: Tensor,
        head_index: Tensor,
        query_index: Tensor,
        key_index: Tensor,
    ) -> Tensor:
        del batch_index, head_index, query_index
        return written[key_index]

    return BlockMask.from_kv_blocks(
        empty_count,
        empty_indices,
        full_count,
        full_indices,
        BLOCK_SIZE=block_size,
        mask_mod=visible_slot,
        seq_lengths=(tokens_per_frame, written.numel()),
        compute_q_blocks=False,
    )
