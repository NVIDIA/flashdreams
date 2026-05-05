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

"""Block-sparse attention backed by ``block_sparse_attn``.

Single-GPU only. Mirrors :class:`NativeAttention`'s constructor /
forward signature so it can be dropped in as a sparse alternative; CP
is intentionally omitted (raises if requested).

Also provides :func:`build_local_spatial_block_mask` and
:func:`build_topk_block_mask` for constructing the per-head 0/1
``base_blockmask`` consumed by :meth:`BlockSparseAttention.forward` —
ported from the FlashVSR reference (a local-spatial neighbourhood
gate combined with a per-head, per-temporal-q-window top-K selection
on averaged window descriptors).
"""

import math
from typing import Literal

import torch
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

import block_sparse_attn.block_sparse_attn_interface as _bsa_interface
from block_sparse_attn import block_sparse_attn_func


def _replace_ones_with_count_capture_safe(
    tensor: Tensor,
) -> tuple[Tensor, int]:
    """Capture-safe replacement for ``block_sparse_attn``'s helper.

    Two changes vs. the library original, both required for the call to
    work under ``torch.cuda.graph(...)`` capture:

    1. Replace ``count[ones_mask]`` + ``masked_scatter`` with a pure
       ``torch.where`` so the rewrite has no value-dependent shapes.
       The original couldn't determine the gathered-source size without
       a host sync — illegal during stream capture
       (``cudaErrorStreamCaptureUnsupported``).
    2. Return ``ones_num`` as a Python ``int`` (taken from the input's
       static shape). Downstream the library does
       ``assert base_blockmask.shape[1] == ones_num``; if ``ones_num``
       is a 0-d GPU tensor (the natural ``ones_mask.sum()`` return),
       ``==`` yields a tensor and ``assert`` triggers a ``bool()``
       host sync — same capture failure, just one line later.

    Item (2) is exact in our use: :class:`BlockSparseAttention` is the
    only call site and always passes a 1-D ``head_mask_type`` made of
    all ones (every head is block-sparse), so the count of ones equals
    ``tensor.shape[-1]``. The library only consumes ``ones_num`` for
    the assertion above, never inside its kernel, so a static int is a
    drop-in replacement for the original tensor return.
    """
    ones_mask = tensor == 1
    count = torch.cumsum(ones_mask, dim=-1).to(tensor.dtype)
    rewritten = torch.where(ones_mask, count, tensor)
    return rewritten, int(tensor.shape[-1])


# Monkey-patch on import so every call site in this venv (only
# :func:`block_sparse_attn_func` for now) routes through the capture-safe
# variant. Cheap and reversible: we hold the original under a private
# attribute in case a future caller wants the lib's untouched behaviour.
_bsa_interface._replace_ones_with_count_original = (
    _bsa_interface.replace_ones_with_count
)
_bsa_interface.replace_ones_with_count = _replace_ones_with_count_capture_safe

# block_sparse_attn's CUDA kernel hard-codes a 128-token block edge; the
# block-mask resolution and any downstream padding follow from this.
_BSA_BLOCK_SIZE: int = 128


class BlockSparseAttention(torch.nn.Module):
    """Block-sparse attention with the same surface as :class:`NativeAttention`.

    Wraps ``block_sparse_attn.block_sparse_attn_func``: every head is
    treated as block-sparse (``head_mask_type == 1``) and gated by a
    per-(batch, head) ``blockmask`` at 128-token granularity. Passing an
    all-ones ``blockmask`` (or ``None``) recovers dense attention.

    Forward synthesizes the blockmask on the fly from a 3D ``q_thw``
    ``(T_q, H, W)`` describing the query token-grid, with an optional
    ``kv_thw`` ``(T_k, H, W)`` for the K/V grid (defaults to ``q_thw``
    for self-attention). It pools per-block descriptors, builds /
    re-uses a cached local-spatial mask, and applies a per-head top-K
    selection where ``K = max(1, round(topk_ratio * slab_size))``.
    ``topk_ratio = 1.0`` short-circuits to an all-ones mask, recovering
    dense attention. Callers must hand in q/k/v that are already in
    ``128``-token block-major order (i.e. each contiguous 128-token
    span along the sequence axis is one window).

    For escape hatches, ``forward`` also accepts an explicit
    ``blockmask`` that overrides the synthesis path entirely; this is
    primarily for tests / fully bespoke masking.
    """

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        *,
        topk_ratio: float = 0.5,
        local_range: int = 9,
        window: tuple[int, int, int] = (2, 8, 8),
    ) -> None:
        """Configure attention layout and auto-mask params.

        Args:
            qkv_format: Layout of the QKV tensors; ``"bhsd"`` is
                ``(B, H, S, D)``, ``"bshd"`` is ``(B, S, H, D)``.
            topk_ratio: Fraction of entries kept per
                ``(head, temporal_q_block)`` slab. Each slab has
                ``Hb * Wb * (Tb * Hb * Wb)`` entries (with
                ``Tb, Hb, Wb`` the block-grid derived from ``layout``
                and ``window``); the per-slab integer top-K is
                ``max(1, round(topk_ratio * slab_size))``. Must be in
                ``[0.0, 1.0]``. ``1.0`` short-circuits to a full-ones
                mask (dense attention).
            local_range: Side length (in blocks) of the square local
                spatial neighbourhood used by the auto-mask path.
                Recommended: ``9`` or ``11``. ``9`` → sharper details;
                ``11`` → more stable results.
            window: ``(wf, wh, ww)`` 3D window mapping the token grid
                ``(T, H, W)`` to the BSA block grid
                ``(T//wf, H//wh, W//ww)``. The product must equal the
                BSA block size (128). Default ``(2, 8, 8)`` matches
                FlashVSR.
        """
        super().__init__()
        assert qkv_format in ("bhsd", "bshd"), f"Invalid qkv format: {qkv_format}"
        assert 0.0 <= topk_ratio <= 1.0, (
            f"topk_ratio must be in [0, 1]; got {topk_ratio}."
        )
        wf, wh, ww = window
        assert wf * wh * ww == _BSA_BLOCK_SIZE, (
            f"window product must equal the BSA block size "
            f"({_BSA_BLOCK_SIZE}); got {wf}*{wh}*{ww}={wf * wh * ww}."
        )
        self.qkv_format = qkv_format
        self.topk_ratio = topk_ratio
        self.local_range = local_range
        self.window = window

        # Lazy local-spatial mask, rebuilt only when (spatial_h,
        # spatial_w) changes. ``local_range`` is fixed at init so it
        # never invalidates the cache.
        self._local_attn_mask: Tensor | None = None
        self._local_attn_mask_key: tuple[int, int] = (-1, -1)

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Reject CP — :class:`BlockSparseAttention` is single-GPU only."""
        if cp_group is not None:
            raise NotImplementedError(
                "BlockSparseAttention does not support context parallel; "
                "use NativeAttention or RingAttention instead."
            )

    def is_context_parallel_enabled(self) -> bool:
        """Always ``False``."""
        return False

    def context_parallel_size(self) -> int:
        """Always ``1``."""
        return 1

    def _get_local_attn_mask(
        self, spatial_h: int, spatial_w: int, device: torch.device
    ) -> Tensor:
        """Lazily build / reuse the cached local-spatial block mask."""
        key = (spatial_h, spatial_w)
        if self._local_attn_mask is None or self._local_attn_mask_key != key:
            self._local_attn_mask = build_local_spatial_block_mask(
                spatial_h,
                spatial_w,
                self.local_range,
                self.local_range,
                include_self=True,
                device=device,
            )
            self._local_attn_mask_key = key
        return self._local_attn_mask

    # Treat the whole forward as opaque to ``torch.compile`` / Dynamo:
    # the body builds shape-dependent control tensors (cu_seqlens via
    # ``torch.arange``, ``base_blockmask`` via ``torch.ones``, the cached
    # local-spatial mask whose grid is derived from ``q_thw``) whose
    # sizes Inductor cannot prove divisibility for once ``kv_thw[0]``
    # is symbolic (streaming KV cache: ``T_k`` flips ``2 → 4``). Inductor
    # gives up with ``CantSplit: 880 not divisible by 48400*((s//7040))
    # + 48400``. Disabling the tracer here lets the surrounding compiled
    # network treat BSA as a single opaque call and run it eagerly,
    # which is the right behaviour anyway — ``block_sparse_attn_func``
    # is already a custom op with its own kernels, nothing to fuse.
    @torch.compiler.disable(recursive=True)
    @torch.no_grad()
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        q_thw: tuple[int, int, int] | None = None,
        kv_thw: tuple[int, int, int] | None = None,
        blockmask: Tensor | None = None,
    ) -> Tensor:
        """Run block-sparse attention.

        Mask selection:

        - Default: synthesize from ``q_thw`` (+ optional ``kv_thw``)
          and ``self.topk_ratio``. ``topk_ratio == 1.0``
          short-circuits to a full-ones mask (dense attention).
        - ``blockmask`` provided: used as-is, overriding the
          synthesis path entirely.

        Args:
            query: Query tensor in ``self.qkv_format``. The sequence
                axis must be in ``128``-token block-major order.
            key: Key tensor in ``self.qkv_format``.
            value: Value tensor in ``self.qkv_format``.
            q_thw: ``(T_q, H, W)`` token-grid dims for the query
                with ``Sq = T_q * H * W``. Each entry must be
                divisible by the matching ``self.window`` entry.
                Required unless ``blockmask`` is provided.
            kv_thw: ``(T_k, H, W)`` token-grid dims for the key /
                value sequence. ``H`` and ``W`` must match
                ``q_thw``'s spatial entries (the local mask is
                spatial-only and shared across q/k); ``T_k`` may
                differ from ``T_q`` (e.g. streaming KV caches longer
                than the query chunk). Defaults to ``q_thw`` —
                self-attention with ``Sq == Sk``.
            blockmask: Optional override 0/1 mask of shape
                ``(B, H, ceil(Sq/128), ceil(Sk/128))`` selecting which
                ``128 x 128`` ``(q_block, k_block)`` tiles each head
                attends to.

        Returns:
            Attention output in the same layout as inputs.
        """
        # block_sparse_attn_func consumes a packed varlen layout
        # (total, H, D); collapse the batch into the token axis once
        # here and undo it on the output.
        if self.qkv_format == "bhsd":
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        B, Sq, H, D = query.shape
        Sk = key.shape[1]
        assert key.shape[2] == H and value.shape[2] == H, (
            "BlockSparseAttention requires query/key/value to share num_heads."
        )
        assert key.shape[0] == B and value.shape[0] == B, (
            "BlockSparseAttention requires query/key/value to share batch size."
        )

        device = query.device

        q_packed = query.reshape(B * Sq, H, D)
        k_packed = key.reshape(B * Sk, H, D)
        v_packed = value.reshape(B * Sk, H, D)

        cu_seqlens_q = torch.arange(
            0, (B + 1) * Sq, Sq, device=device, dtype=torch.int32
        )
        cu_seqlens_k = torch.arange(
            0, (B + 1) * Sk, Sk, device=device, dtype=torch.int32
        )

        # head_mask_type[h] == 1 marks head h as block-sparse; the
        # kernel reindexes ones via replace_ones_with_count, so the
        # blockmask's head axis must enumerate every head we pass in.
        head_mask_type = torch.ones(H, device=device, dtype=torch.int32)

        n_q_blocks = (Sq + _BSA_BLOCK_SIZE - 1) // _BSA_BLOCK_SIZE
        n_k_blocks = (Sk + _BSA_BLOCK_SIZE - 1) // _BSA_BLOCK_SIZE

        if blockmask is not None:
            assert blockmask.shape == (B, H, n_q_blocks, n_k_blocks), (
                f"blockmask shape must be (B={B}, H={H}, "
                f"n_q_blocks={n_q_blocks}, n_k_blocks={n_k_blocks}); "
                f"got {tuple(blockmask.shape)}."
            )
            # Kernel expects a low-bit-width tensor on cuda; convert is
            # safe for both bool and int dtypes.
            base_blockmask = blockmask.to(device=device, dtype=torch.uint8)
        else:
            assert q_thw is not None, (
                "BlockSparseAttention.forward requires either ``q_thw`` "
                "(auto-mask path) or an explicit ``blockmask`` override."
            )
            T_q, H_tok, W_tok = q_thw
            T_k, H_k, W_k = (T_q, H_tok, W_tok) if kv_thw is None else kv_thw
            assert (H_k, W_k) == (H_tok, W_tok), (
                f"kv_thw spatial dims (H={H_k}, W={W_k}) must match "
                f"q_thw's (H={H_tok}, W={W_tok}); the local-spatial "
                f"mask is shared across q and k."
            )
            wf, wh, ww = self.window
            assert (
                T_q % wf == 0
                and T_k % wf == 0
                and H_tok % wh == 0
                and W_tok % ww == 0
            ), (
                f"q_thw=(T_q={T_q}, H={H_tok}, W={W_tok}) and T_k={T_k} "
                f"must each be divisible by window={self.window}."
            )
            L_q = T_q * H_tok * W_tok
            L_k = T_k * H_tok * W_tok
            assert Sq == L_q, f"Sq={Sq} must equal T_q*H*W={L_q} (q_thw={q_thw})."
            assert Sk == L_k, (
                f"Sk={Sk} must equal T_k*H*W={L_k} "
                f"(kv_thw={kv_thw if kv_thw is not None else q_thw})."
            )
            Tb_q = T_q // wf
            Tb_k = T_k // wf
            Hb = H_tok // wh
            Wb = W_tok // ww
            n_blocks_q = Tb_q * Hb * Wb
            n_blocks_k = Tb_k * Hb * Wb
            spatial_q_blocks = Hb * Wb
            slab_size = spatial_q_blocks * n_blocks_k
            topk_int = max(1, int(round(self.topk_ratio * slab_size)))

            if topk_int >= slab_size:
                # ``topk_ratio == 1.0`` (or ratios that round up to the
                # full slab): every q-block sees every k-block, so we
                # skip the descriptor-pool / softmax / topk path
                # entirely and synthesize an all-ones mask in one go.
                base_blockmask = torch.ones(
                    (B, H, n_blocks_q, n_blocks_k),
                    device=device,
                    dtype=torch.uint8,
                )
            else:
                # Mean-pool within each 128-token window to get a
                # single descriptor per (block, head); the spatial-only
                # local mask is broadcast over the temporal axis
                # inside build_topk_block_mask.
                q_desc = query.reshape(
                    B, n_blocks_q, _BSA_BLOCK_SIZE, H, D
                ).mean(dim=2)
                k_desc = key.reshape(
                    B, n_blocks_k, _BSA_BLOCK_SIZE, H, D
                ).mean(dim=2)
                local_mask = self._get_local_attn_mask(Hb, Wb, device)
                base_blockmask = build_topk_block_mask(
                    q_desc,
                    k_desc,
                    num_temporal_q_blocks=Tb_q,
                    num_temporal_k_blocks=Tb_k,
                    topk=topk_int,
                    local_spatial_block_mask=local_mask,
                )

        out_packed = block_sparse_attn_func(
            q_packed,
            k_packed,
            v_packed,
            cu_seqlens_q,
            cu_seqlens_k,
            head_mask_type,
            None,
            base_blockmask,
            Sq,
            Sk,
            0.0,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        )

        out = out_packed.reshape(B, Sq, H, D)
        if self.qkv_format == "bhsd":
            out = out.transpose(1, 2)
        return out


@torch.no_grad()
def build_local_spatial_block_mask(
    spatial_h: int,
    spatial_w: int,
    win_h: int,
    win_w: int,
    *,
    include_self: bool = True,
    device: torch.device | str | None = None,
) -> Tensor:
    """Build a 2D local-neighbourhood mask over a spatial block grid.

    For every block at row-major position ``(r, c)`` in a
    ``spatial_h x spatial_w`` grid, mark every other ``(r', c')`` whose
    row offset lies in ``[r - win_h//2, r - win_h//2 + win_h - 1]`` and
    column offset in the matching window. The window is shifted (not
    centred): the left/top half is ``win_*//2`` cells, so even window
    sizes are biased one cell to the upper-left, matching the FlashVSR
    reference.

    Args:
        spatial_h: Number of block rows.
        spatial_w: Number of block columns.
        win_h: Window height (in blocks).
        win_w: Window width (in blocks).
        include_self: Keep ``(r, c) -> (r, c)`` self-edges.
        device: Output device.

    Returns:
        ``[spatial_h * spatial_w, spatial_h * spatial_w]`` ``bool``
        mask in row-major order over the spatial grid.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    rows = torch.arange(spatial_h, device=device)
    cols = torch.arange(spatial_w, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    r_all = yy.reshape(-1)
    c_all = xx.reshape(-1)

    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r = start_r + win_h - 1
    start_c = c_all - c_half
    end_c = start_c + win_w - 1

    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


@torch.no_grad()
def build_topk_block_mask(
    q_block_descriptor: Tensor,
    k_block_descriptor: Tensor,
    *,
    num_temporal_q_blocks: int,
    num_temporal_k_blocks: int,
    topk: int,
    local_spatial_block_mask: Tensor,
) -> Tensor:
    """Build a per-head top-K block mask for :class:`BlockSparseAttention`.

    Mirrors the FlashVSR ``generate_draft_block_mask`` flow:

    1. Score each ``(q_block, k_block)`` pair per head as
       ``q_block_descriptor @ k_block_descriptor / sqrt(d_head)``.
    2. Add ``-inf`` outside the spatial-only local mask, broadcast over
       the temporal axis (every temporal pair is allowed by the local
       mask).
    3. Softmax over keys.
    4. For each ``(head, temporal_q_block)`` slab of shape
       ``(spatial_q, N_k)``, flatten and keep entries strictly greater
       than the ``(topk + 1)``-th largest value — i.e. at most ``topk``
       entries (modulo ties).

    Args:
        q_block_descriptor: Per-block query summary,
            ``[B, N_q, num_heads, head_dim]``. The reference obtains it
            by 3D-window-partitioning the latent and averaging within
            each window. ``N_q`` must equal
            ``num_temporal_q_blocks * spatial_q`` where ``spatial_q``
            is the leading edge of ``local_spatial_block_mask``.
        k_block_descriptor: Per-block key summary,
            ``[B, N_k, num_heads, head_dim]``.
        num_temporal_q_blocks: Number of temporal blocks in ``N_q``.
        num_temporal_k_blocks: Number of temporal blocks in ``N_k``.
        topk: Top-K count per ``(head, temporal_q_block)`` slab. Must
            be at least ``1``; clamped to ``spatial_q * N_k - 1`` if
            larger so the threshold lookup is valid.
        local_spatial_block_mask: ``[spatial_q, spatial_k]`` ``bool``
            mask gating allowed spatial pairs. Build via
            :func:`build_local_spatial_block_mask`.

    Returns:
        ``[B, num_heads, N_q, N_k]`` ``uint8`` mask suitable for
        :meth:`BlockSparseAttention.forward`'s ``blockmask`` argument.
    """
    assert q_block_descriptor.ndim == 4 and k_block_descriptor.ndim == 4, (
        "q/k_block_descriptor must be [B, N, H, D]; got "
        f"{tuple(q_block_descriptor.shape)} and {tuple(k_block_descriptor.shape)}."
    )
    B, N_q, H, dh = q_block_descriptor.shape
    Bk, N_k, Hk, dhk = k_block_descriptor.shape
    assert (B, H, dh) == (Bk, Hk, dhk), (
        "q/k descriptors must share batch, num_heads and head_dim; got "
        f"{(B, H, dh)} vs {(Bk, Hk, dhk)}."
    )
    assert local_spatial_block_mask.ndim == 2, (
        f"local_spatial_block_mask must be 2D; got {tuple(local_spatial_block_mask.shape)}."
    )
    spatial_q, spatial_k = local_spatial_block_mask.shape
    assert N_q == num_temporal_q_blocks * spatial_q, (
        f"N_q={N_q} must equal num_temporal_q_blocks={num_temporal_q_blocks} "
        f"* spatial_q={spatial_q}."
    )
    assert N_k == num_temporal_k_blocks * spatial_k, (
        f"N_k={N_k} must equal num_temporal_k_blocks={num_temporal_k_blocks} "
        f"* spatial_k={spatial_k}."
    )
    assert topk >= 1, f"topk must be >= 1; got {topk}."

    device = q_block_descriptor.device

    # Compute scores in the descriptor dtype (matching the FlashVSR
    # reference, which scores in bf16 and only promotes to fp32 once
    # the local mask is added). Softmax / topk run in fp32.
    scores = torch.einsum(
        "bnhd,bmhd->bhnm", q_block_descriptor, k_block_descriptor
    ) / math.sqrt(dh)

    # Broadcast the spatial-only mask over the temporal axes:
    #   (spatial_q, spatial_k)
    # → (num_temporal_q_blocks * spatial_q, num_temporal_k_blocks * spatial_k)
    # Every temporal pair (tq, tk) sees the same spatial gate, matching
    # the reference.
    full_mask = local_spatial_block_mask.to(device=device, dtype=torch.bool)
    full_mask = full_mask[None, :, None, :].expand(
        num_temporal_q_blocks, spatial_q, num_temporal_k_blocks, spatial_k
    )
    full_mask = rearrange(
        full_mask, "tq sq tk sk -> (tq sq) (tk sk)"
    ).contiguous()

    # bhnm + nm; broadcast over (B, H). Cast to fp32 here so the
    # masked_fill / softmax / topk path matches the reference's
    # implicit bf16+fp32 → fp32 promotion.
    scores = scores.to(torch.float32).masked_fill(~full_mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)

    # Per-(B, H, tq) top-K over (spatial_q × N_k):
    #   (B, H, N_q, N_k) → (B, H, tq, spatial_q, N_k)
    #                   → (B, H, tq, spatial_q * N_k)
    slab = rearrange(
        attn, "b h (tq sq) nk -> b h tq (sq nk)", tq=num_temporal_q_blocks
    )
    n_entries = slab.shape[-1]
    apply_topk = min(n_entries - 1, topk)
    # values[..., -1] is the (apply_topk+1)-th largest; strict ">"
    # selects at most ``apply_topk`` entries, matching the reference.
    thresholds = torch.topk(
        slab, k=apply_topk + 1, dim=-1, largest=True
    ).values[..., -1:]
    keep = slab > thresholds

    keep = rearrange(
        keep,
        "b h tq (sq nk) -> b h (tq sq) nk",
        sq=spatial_q,
        nk=N_k,
    )
    return keep.to(dtype=torch.uint8).contiguous()
