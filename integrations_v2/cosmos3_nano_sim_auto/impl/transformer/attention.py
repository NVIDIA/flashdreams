# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hyperion's mask/layout adapter over :mod:`flashdreams.accelerated`.

:mod:`cosmos3_nano_sim_auto.impl.transformer.mask` decides visibility and :mod:`cosmos3_nano_sim_auto.impl.packing`
lays the tokens out. Backend selection, compilation, GQA, and kernel tuning are
shared FlashDreams policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import (
    AuxOutput,
    AuxRequest,
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from cosmos3_nano_sim_auto.impl.transformer.fold import FoldPlan, gathered_attention
from flashdreams.accelerated.multi_head_attention.flex import compiled_flex_attention
from flashdreams.accelerated.multi_head_attention.functional import masked_attention
from flashdreams.accelerated.multi_head_attention.optimized import (
    FlexAttentionOptions,
    OptimizedImplConfig,
    QKVFusionOption,
    SDPABackend,
)


@dataclass(frozen=True)
class MasklessFold:
    """The passes of :func:`~cosmos3_nano_sim_auto.impl.transformer.mask.maskless_partitions`, to merge.

    Carried as a mask so that every caller between here and
    :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.chunk_layer_forward` passes it through
    untouched: what a chunk pass is gated by is this module's business, and the
    rollout should not learn a second way to ask for attention.
    """

    passes: tuple[BlockMask, ...]

    def __post_init__(self) -> None:
        if not self.passes:
            raise ValueError("a fold needs at least one pass")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.passes[0].shape)

    def to(self, device: torch.device | str) -> MasklessFold:
        """Move every pass, so a fold stands in for a mask where one is moved.

        :func:`~cosmos3_nano_sim_auto.impl.transformer.forward.run_chunk_layers` moves whatever it is
        given before the first layer reads it, and a fold that quietly lacked
        this would end the run there.
        """
        return MasklessFold(tuple(one.to(device) for one in self.passes))


ChunkMask = Tensor | BlockMask | MasklessFold | FoldPlan
"""What a chunk pass can be gated by: dense for cuDNN, blocks for Flex, or a
fold of blocks for the attention the joint checkpoints were trained under."""

RTX_PRO_6000_BLACKWELL_FLEX_OPTIONS = FlexAttentionOptions(
    block_size=128,
    mask_block_m=256,
    mask_block_n=64,
    compile_dynamic=False,
    block_n=64,
    prescale_qk=True,
    rows_guaranteed_safe=True,
)
"""Measured 11-view kernel policy for the RTX PRO 6000 Blackwell.

The asymmetric 256-by-64 mask passed the full camera/frame gate on seeds 0 and
1 when used by the compiled warm profile. The launch geometry stays
compiler-selected.
"""


def attention_implementation(
    *,
    flex: bool,
    flex_options: FlexAttentionOptions | None = None,
) -> OptimizedImplConfig:
    """Return shared policy for Hyperion's externally projected attention."""
    return OptimizedImplConfig(
        qkv_fusion_option=QKVFusionOption.NONE,
        sdpa_backend=(SDPABackend.FLEX if flex else SDPABackend.CUDNN),
        use_tma=False,
        flex_attention=flex_options or FlexAttentionOptions(),
    )


def rtx_pro_6000_blackwell_attention_implementation(
    views: int = 11,
) -> OptimizedImplConfig:
    """Return the two-seed quality-passing 11-view RTX PRO 6000 policy."""
    if views != 11:
        raise ValueError(
            "the RTX PRO 6000 Blackwell profile is validated only for 11 "
            f"views, not {views}; use --optimization-profile auto"
        )
    return attention_implementation(
        flex=True,
        flex_options=RTX_PRO_6000_BLACKWELL_FLEX_OPTIONS,
    )


def _fused_merge_supported(
    parts: Sequence[tuple[Tensor, Tensor]], output_dtype: torch.dtype | None
) -> bool:
    """The eager inference layouts whose native result is contiguous.

    Exact strides matter: TensorIterator can preserve noncanonical singleton
    strides or follow a noncontiguous LSE. Lazy negative/conjugate views also
    stay native: a raw pointer does not carry their logical value transform.
    """
    # PyTorch has no public guard for vmap's storage-less tensor wrappers.
    if (
        torch.compiler.is_compiling()
        or torch.version.hip is not None
        or torch._C._are_functorch_transforms_active()
    ):
        return False
    if not 1 <= len(parts) <= 4:
        return False
    supported_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if output_dtype is not None and output_dtype not in supported_dtypes:
        return False
    first = parts[0][0]
    if (
        first.device.type != "cuda"
        or first.layout != torch.strided
        or first.ndim != 4
        or first.dtype not in supported_dtypes
        or not 0 < first.numel() < 2**31
    ):
        return False
    batch, heads, queries, width = first.shape
    if min(heads, queries, width) <= 1:
        return False
    contiguous = (heads * queries * width, queries * width, width, 1)
    head_query = (heads * queries * width, width, heads * width, 1)
    lse_shape = (batch, heads, queries)
    lse_stride = (heads * queries, queries, 1)
    return all(
        out.device == first.device
        and out.layout == torch.strided
        and out.dtype == first.dtype
        and out.shape == first.shape
        and out.stride() in (contiguous, head_query)
        and not out.requires_grad
        and torch.autograd.forward_ad.unpack_dual(out).tangent is None
        and not out.is_neg()
        and not out.is_conj()
        and lse.device == first.device
        and lse.layout == torch.strided
        and lse.dtype == torch.float32
        and lse.shape == lse_shape
        and lse.stride() == lse_stride
        and not lse.requires_grad
        and torch.autograd.forward_ad.unpack_dual(lse).tangent is None
        and not lse.is_neg()
        and not lse.is_conj()
        for out, lse in parts
    )


def merge_attentions(
    parts: Sequence[tuple[Tensor, Tensor]],
    *,
    fused: bool = False,
    output_dtype: torch.dtype | None = None,
) -> Tensor:
    """Merge passes over different key sets by their log-sum-exp weights.

    Each pass softmaxes over its own keys, so its output is a mean the merge has
    to reweight rather than a term it can add. The log-sum-exp is that pass's
    denominator, and dividing by their sum recovers what one softmax over the
    concatenated keys would have given -- exactly, for key sets that do not
    overlap, and with any shared key counted once per pass that holds it.

    Args:
        parts: ``(out [B, H, Q, d], lse [B, H, Q])`` a pass.
        fused: Use the optional CUDA inference vector merge for qualified
            layouts. Autograd, compiler and unsupported inputs stay native.
        output_dtype: Optional final cast; omitted preserves the promoted dtype.

    Returns:
        ``[B, H, Q, d]``, in whatever dtype the weights promote to -- which for
        a bf16 pass and its float32 log-sum-exp is float32. Merging in the
        wider of the two is deliberate: the weights are exponentials of
        differences of logs, and bf16 has three decimal digits to spend on
        them. Callers that must hand the result back to the model cast down
        themselves, or request ``output_dtype`` when that dtype is known.
    """
    peak = torch.stack([lse for _out, lse in parts]).amax(dim=0)
    # A row no pass can read peaks at -inf, and `-inf - -inf` is NaN, so the
    # subtraction below needs it grounded first. Such rows do exist: a caption
    # token is a key to every target and a query to nothing, and padding is
    # only ever a key. Grounding the peak leaves their weights at zero, and the
    # divisor below at one, so they come back zeroed the way the single masked
    # pass returns them rather than as a NaN that spreads over the frame.
    readable = torch.isfinite(peak)
    grounded = torch.where(readable, peak, torch.zeros_like(peak))
    # A pass with no keys for a row returns -inf here, whose exp is a clean
    # zero, so that pass drops out of the sum on its own -- unless its output
    # is NaN rather than merely unused, which `where` stops from spreading
    # through the multiply.
    weights = [torch.exp(lse - grounded).unsqueeze(-1) for _out, lse in parts]
    total = torch.stack(weights).sum(dim=0)
    kernel = None
    # Tensor subclasses can advertise CUDA metadata without ordinary storage.
    # Keep their native dispatch semantics, including eager FakeTensor tracing.
    if (
        fused
        and all(
            type(value) in (Tensor, torch.nn.Parameter)
            for part in parts
            for value in part
        )
        and _fused_merge_supported(parts, output_dtype)
    ):
        try:
            from cosmos3_nano_sim_auto.impl.transformer._merge_kernel import (
                merge_weighted_outputs,
            )
        except ImportError:
            pass
        else:
            kernel = merge_weighted_outputs
    merged = (
        None
        if kernel is not None
        else sum(
            torch.where(weight > 0, out * weight, torch.zeros_like(out))
            for (out, _lse), weight in zip(parts, weights, strict=True)
        )
    )
    divisor = torch.where(readable.unsqueeze(-1), total, torch.ones_like(total))
    if kernel is not None:
        return kernel(
            [out for out, _lse in parts],
            weights,
            divisor,
            output_dtype=output_dtype or torch.float32,
        )
    assert merged is not None
    result = merged / divisor
    return result if output_dtype is None else result.to(output_dtype)


def _folded_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    fold: MasklessFold,
    *,
    enable_gqa: bool,
    flex_options: FlexAttentionOptions,
) -> Tensor:
    """One FlexAttention pass a partition, merged by :func:`merge_attentions`.

    Gating each pass by a block mask rather than gathering its keys into a
    dense run is what makes this the same arithmetic as the model's folds without
    being the same kernels: the merge and the double-weighting are theirs, the
    launch shape is ours, and the speed their dense form buys is not here. The
    numbers come first because they are what the checkpoint was trained under,
    and because a fast fold needs something correct to be checked against.

    Returns ``q``'s dtype, the way a single masked pass does. The merge runs
    wider than the model because the log-sum-exp comes back float32, and the
    next thing to touch this is a bf16 ``o_proj`` that would refuse it.
    """
    # Compiled on the device this runs on, through the same entry
    # `masked_attention` uses. Eager FlexAttention does not merely run slower:
    # it materializes the whole score matrix, which at this rig's key stream is
    # tens of gigabytes a pass, and it warns rather than failing -- so it would
    # surface as an out-of-memory bug a long way from its cause.
    #
    # Eager on CPU because Inductor's CPU lowering of FlexAttention has no
    # auxiliary outputs, and the log-sum-exp is not optional here. The same
    # device split as :func:`~flashdreams.accelerated.multi_head_attention.functional.backend_for`,
    # for the same reason: CPU is where the tests run, not where the clip does.
    kernel: Callable[..., object] = (
        compiled_flex_attention(dynamic=flex_options.compile_dynamic)
        if q.is_cuda
        else flex_attention
    )

    parts: list[tuple[Tensor, Tensor]] = []
    for block_mask in fold.passes:
        # Cast because one annotation covers every shape of `flex_attention`'s
        # return and `return_aux` decides which.
        out, aux = cast(
            "tuple[Tensor, AuxOutput]",
            kernel(
                q,
                k,
                v,
                block_mask=block_mask,
                enable_gqa=enable_gqa,
                # Cast because the options carry a plain dict and the kernel
                # declares a TypedDict over the same keys.
                kernel_options=cast(
                    "FlexKernelOptions | None", flex_options.kernel_options or None
                ),
                return_aux=AuxRequest(lse=True),
            ),
        )
        if aux.lse is None:
            # Asked for above, so this is the kernel's contract changing under
            # us. Worth failing on rather than merging without it: the weights
            # are the log-sum-exp, and there is no sensible default for them.
            raise RuntimeError(
                "FlexAttention returned no log-sum-exp despite AuxRequest(lse=True); "
                "the fold cannot merge its passes without it"
            )
        parts.append((out, aux.lse))
    return merge_attentions(parts).to(q.dtype)


def chunk_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: ChunkMask,
    *,
    enable_gqa: bool = False,
    implementation: OptimizedImplConfig | None = None,
) -> Tensor:
    """Masked attention for one chunk's queries against the full key stream.

    The mask type picks the kernel: a dense tensor goes to cuDNN, a
    :class:`BlockMask` to compiled FlexAttention. Both carry the same rules and
    use the default ``1/sqrt(d)`` scale, so they agree to bf16 tolerance.

    Args:
        q: ``[B, Hq, Q, d]``, the current chunk's noisy tokens. Only the chunk
            is ever denoised.
        k, v: ``[B, Hq, KV, d]``, or ``[B, Hkv, KV, d]`` with ``enable_gqa``.
            The key stream is ``[text | chunk | memory]``, in that order,
            matching :func:`cosmos3_nano_sim_auto.impl.packing.build_chunk_metadata`.
        mask: ``[Q, KV]`` bool from :meth:`~cosmos3_nano_sim_auto.impl.packing.ChunkMetadata.mask`,
            or the block mask from
            :meth:`~cosmos3_nano_sim_auto.impl.packing.ChunkMetadata.block_mask`.
        enable_gqa: Pass key/value heads to the kernel unexpanded, letting query
            heads share them, rather than repeating them to match first.
        implementation: Shared backend and kernel policy. When omitted, the
            mask type selects the default FlexAttention or cuDNN policy.

    Returns:
        ``[B, Hq, Q, d]``.
    """
    expected = (q.shape[-2], k.shape[-2])
    if isinstance(mask, FoldPlan):
        return gathered_attention(q, k, v, mask, enable_gqa=enable_gqa)

    if isinstance(mask, MasklessFold):
        for one in mask.passes:
            if tuple(one.shape[-2:]) != expected:
                raise ValueError(
                    f"Fold pass {tuple(one.shape[-2:])} does not match the "
                    f"{expected[0]} queries and {expected[1]} keys it must gate."
                )
        implementation = implementation or attention_implementation(flex=True)
        return _folded_attention(
            q,
            k,
            v,
            mask,
            enable_gqa=enable_gqa,
            flex_options=implementation.flex_attention,
        )

    block_mask = isinstance(mask, BlockMask)
    if block_mask:
        if tuple(mask.shape[-2:]) != expected:
            raise ValueError(
                f"Block mask {tuple(mask.shape[-2:])} does not match the {expected[0]} "
                f"queries and {expected[1]} keys it must gate."
            )
    else:
        if mask.ndim != 2:
            raise ValueError(f"Expected a [Q, KV] mask, got shape {tuple(mask.shape)}.")
        if tuple(mask.shape) != expected:
            # Checked rather than left to SDPA: a mask sized for a different
            # chunk broadcasts instead of failing when one axis happens to be 1.
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match the {expected[0]} "
                f"queries and {expected[1]} keys it must gate."
            )

    implementation = implementation or attention_implementation(flex=block_mask)
    if (implementation.sdpa_backend is SDPABackend.FLEX) is not block_mask:
        raise ValueError(
            f"{implementation.sdpa_backend.value} does not match "
            f"{'a block' if block_mask else 'a dense'} mask"
        )
    return masked_attention(
        q,
        k,
        v,
        mask,
        enable_gqa=enable_gqa,
        flex_options=implementation.flex_attention,
    )
