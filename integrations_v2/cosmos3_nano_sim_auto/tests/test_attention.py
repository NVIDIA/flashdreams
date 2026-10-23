# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for one chunk's attention.

``test_mask.py`` and ``test_packing.py`` check the mask as a boolean tensor. The
question here is whether it reaches the kernel and does anything -- a mask that
is correct and then dropped on the floor passes every test in those files.

So the load-bearing test is not that SDPA agrees with the explicit-softmax
reference; two implementations of one formula would agree even if both ignored
the mask. It is that perturbing a hidden key changes nothing and perturbing a
visible one changes something.

CPU and tiny, so this runs in the local loop. Only the pinned backend's *effect*
needs the GB300; that it is pinned at all is checked here, because for a while
it was not.
"""

from __future__ import annotations

import math

import pytest
import torch
from cosmos3_nano_sim_auto.impl.packing import (
    ClipGeometry,
    build_chunk_metadata,
    build_memory_layout,
)
from cosmos3_nano_sim_auto.impl.transformer.attention import (
    attention_implementation,
    chunk_attention,
    merge_attentions,
)
from cosmos3_nano_sim_auto.impl.transformer.mask import (
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
)
from cosmos_interactive._attention import reference_masked_attention

from flashdreams.accelerated.multi_head_attention.optimized import (
    FlexAttentionOptions,
    SDPABackend,
)

pytestmark = pytest.mark.ci_cpu

# Four frames rather than the three the other test modules use, so that a chunk
# in the middle has controls both behind and ahead of it.
CLIP = ClipGeometry(
    num_views=2,
    frames_per_view=4,
    patch_h=1,
    patch_w=1,
    frames_per_chunk=1,
    condition_frames=1,
    seconds_per_frame=0.5,
)
HEADS, HEAD_DIM = 4, 8
CHUNK_FRAME = 2


def chunk_at(frame: int = CHUNK_FRAME):
    memory = build_memory_layout(
        CLIP,
        history_frame_ranges=[(f, f + 1) for f in range(CLIP.condition_frames, frame)],
    )
    return build_chunk_metadata(
        CLIP, memory, chunk_start=frame, chunk_frames=1, text_tokens=1
    )


def qkv(num_queries: int, num_keys: int, *, kv_heads: int = HEADS, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def rand(heads: int, length: int) -> torch.Tensor:
        return torch.randn(1, heads, length, HEAD_DIM, generator=generator)

    return rand(HEADS, num_queries), rand(kv_heads, num_keys), rand(kv_heads, num_keys)


def index_of(stream, role: int, frame: int, view: int) -> int:
    matches = (
        (
            (stream.token_role_id == role)
            & (stream.frame_id == frame)
            & (stream.view_id == view)
        )
        .nonzero()
        .flatten()
    )
    assert matches.numel() == 1, (
        f"expected one {role=} {frame=} {view=}, got {matches.tolist()}"
    )
    return int(matches[0])


@pytest.mark.parametrize("block_size", [16, (16, 32)])
@pytest.mark.parametrize("enable_gqa", [False, True])
def test_the_block_mask_and_the_dense_mask_produce_the_same_attention(
    enable_gqa, block_size
) -> None:
    """Flex over blocks and cuDNN over the dense mask are the same attention.

    Both take the default ``1/sqrt(d)`` scale and neither is causal, so the only
    thing that could differ is the mask, which is the thing being checked. This
    is the test that says the fast path is a substitution rather than a change.

    Small enough to compile on CPU, so the substitution is checked in the local
    loop rather than only on the machine it is meant to be fast on. The block
    size has to come down with it: at the default 128 a chunk this size is one
    block, and one block is never skipped.
    """
    chunk = chunk_at()
    kv_heads = HEADS // 2 if enable_gqa else HEADS
    q, k, v = qkv(len(chunk.query), len(chunk.kv), kv_heads=kv_heads)

    dense = chunk_attention(q, k, v, chunk.mask(), enable_gqa=enable_gqa)
    blocks = chunk_attention(
        q, k, v, chunk.block_mask(block_size=block_size), enable_gqa=enable_gqa
    )

    assert (dense - blocks).abs().max().item() < 1e-5


def test_a_block_mask_sized_for_another_chunk_is_refused() -> None:
    """The dense path checks this, so the block path has to as well.

    A mask of the wrong width is the failure that does not announce itself: it
    gates the wrong keys and returns a plausible tensor.
    """
    chunk, other = chunk_at(), chunk_at(frame=3)
    q, k, v = qkv(len(chunk.query), len(chunk.kv))

    with pytest.raises(ValueError, match="does not match"):
        chunk_attention(q, k, v, other.block_mask(block_size=16))


def test_matches_the_explicit_softmax_reference() -> None:
    chunk = chunk_at()
    mask = chunk.mask()
    q, k, v = qkv(len(chunk.query), len(chunk.kv))

    out = chunk_attention(q, k, v, mask)
    ref = reference_masked_attention(q, k, v, mask)

    assert out.shape == q.shape
    assert (out - ref).abs().max().item() < 1e-5


@pytest.mark.parametrize(
    ("role", "frame", "view", "visible"),
    [
        (ROLE_CONTROL, 1, 0, True),  # own view, earlier step
        (ROLE_CONTROL, 2, 0, True),  # own view, own step
        (ROLE_CONTROL, 3, 0, False),  # cached from the start, but still the future
        (ROLE_CONTROL, 2, 1, False),  # controls never leave their view
        (ROLE_CLEAN_TARGET, 1, 1, True),  # all_views admits another camera's history
        (ROLE_CURRENT_TARGET, 2, 1, True),  # the same frame in the other camera
    ],
)
def test_output_depends_on_exactly_the_visible_keys(role, frame, view, visible) -> None:
    """Perturb one key and see whether it reaches the output.

    Two assertions per case: the mask says so, and then the kernel agrees. The
    third case is the one that matters most -- a control token from later in the
    clip, cached from the first chunk because the control video is known in
    advance, which the mask has to keep out.
    """
    chunk = chunk_at()
    mask = chunk.mask()
    q, k, v = qkv(len(chunk.query), len(chunk.kv))
    query = index_of(chunk.query, ROLE_CURRENT_TARGET, frame=CHUNK_FRAME, view=0)
    key = index_of(chunk.kv, role, frame, view)

    assert bool(mask[query, key].item()) is visible

    before = chunk_attention(q, k, v, mask)[0, :, query]
    k[0, :, key] += 10.0
    v[0, :, key] += 10.0
    after = chunk_attention(q, k, v, mask)[0, :, query]

    assert ((after - before).abs().max().item() > 1e-4) is visible


def test_hidden_keys_stay_hidden_for_every_query() -> None:
    """The whole hidden column set at once, not just the sampled ones."""
    chunk = chunk_at()
    mask = chunk.mask()
    q, k, v = qkv(len(chunk.query), len(chunk.kv))
    before = chunk_attention(q, k, v, mask)

    hidden = ~mask.any(dim=0)
    assert hidden.any(), (
        "expected this layout to hide at least one key from every query"
    )
    k[0, :, hidden] += 10.0
    v[0, :, hidden] += 10.0

    assert (chunk_attention(q, k, v, mask) - before).abs().max().item() < 1e-6


def test_no_chunk_of_a_rollout_has_an_all_true_mask() -> None:
    """Why ``cached_joint_attention``'s unmasked fast path never applies here.

    It drops the mask when every cached key is visible, which lets SDPA reach a
    fused kernel. Walk a whole rollout and no chunk qualifies: cross-view and
    future controls are in memory throughout and must stay hidden. Every
    attention in this model pays for a dense mask.
    """
    assert CLIP.frames_per_view is not None
    for frame in range(CLIP.condition_frames, CLIP.frames_per_view):
        assert not chunk_at(frame).mask().all()


def test_grouped_key_value_heads_match_repeating_them() -> None:
    chunk = chunk_at()
    mask = chunk.mask()
    q, k, v = qkv(len(chunk.query), len(chunk.kv), kv_heads=2)
    repeats = HEADS // k.shape[1]

    grouped = chunk_attention(q, k, v, mask, enable_gqa=True)
    expanded = chunk_attention(
        q,
        k.repeat_interleave(repeats, dim=1),
        v.repeat_interleave(repeats, dim=1),
        mask,
    )

    assert (grouped - expanded).abs().max().item() < 1e-5


def test_hyperion_expresses_attention_as_shared_optimization_policy() -> None:
    options = FlexAttentionOptions(
        block_size=64, num_warps=4, prescale_qk=True, backend="TRITON"
    )

    sparse = attention_implementation(flex=True, flex_options=options)
    dense = attention_implementation(flex=False)

    assert sparse.sdpa_backend is SDPABackend.FLEX
    assert sparse.flex_attention is options
    assert sparse.flex_attention.kernel_options == {
        "num_warps": 4,
        "PRESCALE_QK": True,
        "BACKEND": "TRITON",
    }
    assert dense.sdpa_backend is SDPABackend.CUDNN


@pytest.mark.parametrize("field", ["compile_dynamic", "prescale_qk", "use_tma"])
def test_flex_tri_state_options_do_not_accept_integer_booleans(field: str) -> None:
    with pytest.raises(TypeError, match="bool or None"):
        FlexAttentionOptions(**{field: 1})  # ty: ignore[invalid-argument-type]  # Exercise invalid-input validation.


def test_flex_kernel_backend_is_validated_before_compilation() -> None:
    with pytest.raises(ValueError, match="TRITON, FLASH"):
        FlexAttentionOptions(backend="fastest")


def test_mask_representation_must_match_the_shared_backend_policy() -> None:
    chunk = chunk_at()
    q, k, v = qkv(len(chunk.query), len(chunk.kv))

    with pytest.raises(ValueError, match="does not match"):
        chunk_attention(
            q,
            k,
            v,
            chunk.mask(),
            implementation=attention_implementation(flex=True),
        )
    with pytest.raises(ValueError, match="does not match"):
        chunk_attention(
            q,
            k,
            v,
            chunk.block_mask(block_size=16),
            implementation=attention_implementation(flex=False),
        )


def test_mask_shape_must_match_the_streams_it_gates() -> None:
    chunk = chunk_at()
    mask = chunk.mask()
    q, k, v = qkv(len(chunk.query), len(chunk.kv))

    with pytest.raises(ValueError, match="does not match"):
        chunk_attention(q, k, v, mask[:, :-1])
    with pytest.raises(ValueError, match=r"\[Q, KV\] mask"):
        chunk_attention(q, k, v, mask[None])


def _pass(q, k, v, allowed):
    """One pass's output and log-sum-exp, by explicit softmax over its keys.

    The merge is checked against arithmetic rather than against a second
    implementation of itself, so this deliberately does not call anything the
    fold uses.
    """
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v, torch.logsumexp(scores, dim=-1)


def _qkv(queries: int, keys: int, dim: int = 8):
    generator = torch.Generator().manual_seed(0)

    def draw(length: int):
        return torch.randn(1, 1, length, dim, generator=generator, dtype=torch.float64)

    return draw(queries), draw(keys), draw(keys)


def test_merging_disjoint_passes_is_one_softmax_over_their_union() -> None:
    """The property that makes `same_view` exact against its mask.

    Splitting a key set in two and merging by log-sum-exp has to reproduce the
    single softmax exactly, or the fold would differ from the mask everywhere
    rather than only on the cell the reference implementation double-weights, and the 12.8% figure would
    be measuring this bug instead of their design.
    """
    q, k, v = _qkv(queries=6, keys=10)
    whole = torch.rand(6, 10) < 0.7
    whole[:, 0] = True  # no empty row: -inf everywhere is a NaN, tested below
    left = whole & (torch.arange(10) < 4)
    right = whole & (torch.arange(10) >= 4)

    merged = merge_attentions([_pass(q, k, v, left), _pass(q, k, v, right)])
    once, _lse = _pass(q, k, v, whole)

    torch.testing.assert_close(merged, once)


def test_merging_a_shared_key_weights_it_twice() -> None:
    """The deliberate difference, isolated to one key.

    the model's folds overlap, and the merge has no way to know a key arrived twice --
    it weights each pass by that pass's own denominator. This pins the size of
    that effect to arithmetic: a key in both passes lands with double the
    softmax weight, which is the same as appending a duplicate column.
    """
    q, k, v = _qkv(queries=4, keys=6)
    left = torch.zeros(4, 6, dtype=torch.bool)
    right = torch.zeros(4, 6, dtype=torch.bool)
    left[:, :4] = True
    right[:, 3:] = True  # key 3 belongs to both

    merged = merge_attentions([_pass(q, k, v, left), _pass(q, k, v, right)])

    doubled_k = torch.cat([k, k[:, :, 3:4]], dim=2)
    doubled_v = torch.cat([v, v[:, :, 3:4]], dim=2)
    duplicated, _lse = _pass(
        q, doubled_k, doubled_v, torch.ones(4, 7, dtype=torch.bool)
    )

    torch.testing.assert_close(merged, duplicated)
    assert not torch.allclose(merged, _pass(q, k, v, left | right)[0])


def test_a_pass_with_nothing_to_read_drops_out_rather_than_poisoning() -> None:
    """Every fold has rows like this: padding reaches no caption, and a
    one-view run has an empty cross-instant pass. Such a pass returns -inf and
    whatever the kernel left in its output buffer, and a merge that multiplied
    that through would turn one idle pass into a NaN frame.
    """
    q, k, v = _qkv(queries=3, keys=5)
    everything = torch.ones(3, 5, dtype=torch.bool)
    nothing = torch.zeros(3, 5, dtype=torch.bool)

    out, lse = _pass(q, k, v, nothing)
    merged = merge_attentions(
        [_pass(q, k, v, everything), (torch.full_like(out, float("nan")), lse)]
    )

    assert torch.isfinite(merged).all()
    torch.testing.assert_close(merged, _pass(q, k, v, everything)[0])


def test_a_row_no_pass_can_read_comes_back_zeroed_not_NaN() -> None:
    """Every fold has rows like this, and the mask path returns them zeroed.

    A caption is a key to every target and a query to nothing, so its row is
    empty in all passes at once. That makes the merge's peak -inf, and
    `-inf - -inf` is NaN -- one row's worth of NaN, which the next layer's
    attention then spreads over the whole frame.
    """
    q, k, v = _qkv(queries=3, keys=5)
    nothing = torch.zeros(3, 5, dtype=torch.bool)

    merged = merge_attentions([_pass(q, k, v, nothing), _pass(q, k, v, nothing)])

    assert torch.isfinite(merged).all()
    assert not merged.any()


@pytest.mark.parametrize(
    ("scope", "agrees"),
    [("same_view", True), ("decomposed", False)],
)
def test_the_fold_agrees_with_the_mask_only_where_it_has_no_overlap(
    scope, agrees
) -> None:
    """The end-to-end check, through the kernel rather than the arithmetic.

    `same_view` partitions the key stream, so folding it and masking it are the
    same attention and must agree to rounding -- which makes this the test that
    the partitions, the block masks built from them and the merge are all
    right, since any one of the three being wrong shows up here.

    `decomposed` overlaps by design, and must *not* agree: the reference implementation measured the two
    22% apart on their sample geometry, and a fold that matched its mask would
    mean the overlap had been quietly dropped somewhere.
    """
    metadata = chunk_at()
    q, k, v = qkv(len(metadata.query), len(metadata.kv))

    folded = chunk_attention(
        q, k, v, metadata.maskless_fold(scope=scope, block_size=16)
    )
    masked = chunk_attention(q, k, v, metadata.block_mask(scope=scope, block_size=16))

    assert torch.isfinite(folded).all()
    apart = (folded - masked).norm() / masked.norm()
    if agrees:
        torch.testing.assert_close(folded, masked, atol=1e-5, rtol=1e-5)
    else:
        assert apart > 0.05, (
            f"the overlap is worth {apart:.1%}, expected a measurable overlap contribution"
        )


def test_a_fold_moves_like_a_mask_does() -> None:
    """`run_chunk_layers` moves whatever it is handed before the first layer.

    A fold is passed down that path in a mask's place, so it has to answer the
    same call -- and a fold that did not would fail on the device the flag
    exists for and nowhere else, since a CPU run moves to the device it is on.
    """
    metadata = chunk_at()
    fold = metadata.maskless_fold(scope="decomposed", block_size=16)

    moved = fold.to(torch.device("cpu"))

    assert len(moved.passes) == len(fold.passes)
    assert moved.shape == fold.shape


def test_a_fold_hands_back_the_dtype_the_model_gave_it() -> None:
    """The log-sum-exp comes back float32 whatever the pass ran in.

    So a bf16 model folding its attention gets float32 out of the merge, and
    the very next call is an `o_proj` whose weights are bf16 -- which does not
    warn or degrade, it refuses. Pinned here because the crash lands in
    `torch.nn.Linear`, several frames from anything that mentions a fold.
    """
    metadata = chunk_at()
    q, k, v = (
        tensor.to(torch.bfloat16)
        for tensor in qkv(len(metadata.query), len(metadata.kv))
    )

    folded = chunk_attention(
        q, k, v, metadata.maskless_fold(scope="decomposed", block_size=16)
    )

    assert folded.dtype is torch.bfloat16
    assert torch.isfinite(folded).all()


@pytest.mark.parametrize("case", ["regular", "transposed_lse", "singleton", "float64"])
def test_optional_merge_fallback_preserves_native_dtype_stride_and_bytes(case):
    """The opt-in cannot silently change TensorIterator's fallback layout."""
    shape = (2, 3, 1, 4) if case == "singleton" else (2, 3, 7, 4)
    out = torch.randn(shape, dtype=torch.bfloat16)
    if case == "singleton":
        out = torch.empty_strided(shape, (12, 4, 12, 1), dtype=out.dtype).normal_()
    lse = torch.randn(shape[:-1])
    if case == "transposed_lse":
        lse = lse.transpose(1, 2).contiguous().transpose(1, 2)
    if case == "float64":
        lse = lse.to(torch.float64)
    parts = [(out, lse)]
    expected = merge_attentions(parts)
    for dtype in (None, torch.bfloat16):
        reference = expected if dtype is None else expected.to(dtype)
        actual = merge_attentions(parts, fused=True, output_dtype=dtype)
        assert actual.dtype == reference.dtype
        assert actual.stride() == reference.stride()
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        )


def test_optional_merge_keeps_autograd():
    out = torch.randn(2, 3, 7, 4, requires_grad=True)
    lse = torch.randn(2, 3, 7, requires_grad=True)
    reference = merge_attentions([(out, lse)])
    actual = merge_attentions([(out, lse)], fused=True)
    expected_grad = torch.autograd.grad(reference.square().sum(), (out, lse))
    actual_grad = torch.autograd.grad(actual.square().sum(), (out, lse))
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        assert torch.equal(actual_value, expected_value)


def test_fused_merge_gate_only_accepts_qualified_cuda_metadata(monkeypatch):
    """Fake CUDA exercises dispatch without initializing or allocating CUDA."""
    # FakeTensor otherwise initializes a real GPU context when one is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    from cosmos3_nano_sim_auto.impl.transformer.attention import (
        _fused_merge_supported,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        out = torch.empty(2, 3, 7, 4, device="cuda:0", dtype=torch.bfloat16)
        lse = torch.empty(2, 3, 7, device="cuda:0")
        interleaved = out.transpose(1, 2).contiguous().transpose(1, 2)
        # Global grad mode is permitted; only tensors needing gradients reject.
        with torch.enable_grad():
            assert _fused_merge_supported([(out, lse)], None)
            assert _fused_merge_supported([(out, lse), (interleaved, lse)], out.dtype)
        assert not _fused_merge_supported([(out.requires_grad_(), lse)], None)
        out.requires_grad_(False)
        assert not _fused_merge_supported([(out, lse.requires_grad_())], None)
        lse.requires_grad_(False)
        transposed_lse = lse.transpose(1, 2).contiguous().transpose(1, 2)
        assert not _fused_merge_supported([(out, transposed_lse)], None)
        singleton = torch.empty_strided(
            (2, 3, 1, 4), (12, 4, 12, 1), device="cuda:0", dtype=out.dtype
        )
        singleton_lse = torch.empty(2, 3, 1, device="cuda:0")
        assert singleton.is_contiguous()
        assert not _fused_merge_supported([(singleton, singleton_lse)], None)
        sparse_out = torch.empty(
            out.shape, device="cuda:0", dtype=out.dtype, layout=torch.sparse_coo
        )
        sparse_lse = torch.empty(lse.shape, device="cuda:0", layout=torch.sparse_coo)
        assert not _fused_merge_supported([(sparse_out, lse)], None)
        assert not _fused_merge_supported([(out, lse), (sparse_out, lse)], None)
        assert not _fused_merge_supported([(out, sparse_lse)], None)
        assert not _fused_merge_supported([(torch._neg_view(out), lse)], None)
        assert not _fused_merge_supported([(out, torch._neg_view(lse))], None)
        conjugate = torch.empty(
            out.shape, device="cuda:0", dtype=torch.complex64
        ).conj()
        assert conjugate.is_conj()
        assert not _fused_merge_supported([(conjugate, lse)], None)
        assert not _fused_merge_supported([(out, lse)] * 5, None)
        assert not _fused_merge_supported([(out, lse)], torch.float64)
        large = torch.empty(1, 2, 2**20, 1024, device="cuda:0", dtype=out.dtype)
        large_lse = torch.empty(1, 2, 2**20, device="cuda:0")
        assert not _fused_merge_supported([(large, large_lse)], None)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert not _fused_merge_supported([(out, lse)], None)


@pytest.mark.parametrize(
    "transform", ["negative_output", "negative_lse", "conjugate_output"]
)
def test_optional_merge_fallback_resolves_lazy_value_transforms(transform):
    out = torch.arange(1, 169, dtype=torch.float32).reshape(2, 3, 7, 4)
    lse = torch.linspace(-2, 2, 42).reshape(2, 3, 7)
    if transform == "negative_output":
        out = torch._neg_view(out)
    elif transform == "negative_lse":
        lse = torch._neg_view(lse)
    else:
        out = (out.to(torch.complex64) * (1 + 2j)).conj()
    parts = [(out, lse), (torch.ones_like(out), torch.zeros_like(lse))]
    resolved = [
        (part.resolve_neg().resolve_conj(), logs.resolve_neg()) for part, logs in parts
    ]
    expected = merge_attentions(resolved)
    actual = merge_attentions(parts, fused=True)
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize("dual_input", ["output", "lse"])
def test_optional_merge_preserves_forward_ad(dual_input, monkeypatch):
    # FakeTensor otherwise initializes a real GPU context when one is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    from cosmos3_nano_sim_auto.impl.transformer.attention import (
        _fused_merge_supported,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    with torch.autograd.forward_ad.dual_level():
        out = torch.randn(2, 3, 7, 4)
        lse = torch.randn(2, 3, 7)
        if dual_input == "output":
            out = torch.autograd.forward_ad.make_dual(out, torch.ones_like(out))
        else:
            lse = torch.autograd.forward_ad.make_dual(lse, torch.ones_like(lse))
        parts = [(out, lse), (torch.ones_like(out), torch.zeros_like(lse))]
        expected = torch.autograd.forward_ad.unpack_dual(merge_attentions(parts))
        actual = torch.autograd.forward_ad.unpack_dual(
            merge_attentions(parts, fused=True)
        )
        assert torch.equal(actual.primal, expected.primal)
        assert actual.tangent is not None
        assert torch.equal(actual.tangent, expected.tangent)
        with FakeTensorMode():
            fake_out = torch.empty(2, 3, 7, 4, device="cuda:0")
            fake_lse = torch.empty(2, 3, 7, device="cuda:0")
            if dual_input == "output":
                fake_out = torch.autograd.forward_ad.make_dual(
                    fake_out, torch.ones_like(fake_out)
                )
            else:
                fake_lse = torch.autograd.forward_ad.make_dual(
                    fake_lse, torch.ones_like(fake_lse)
                )
            assert not fake_out.requires_grad and not fake_lse.requires_grad
            assert not _fused_merge_supported([(fake_out, fake_lse)], None)


def test_optional_merge_keeps_vmap_on_the_native_path(monkeypatch):
    # FakeTensor otherwise initializes a real GPU context when one is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    from cosmos3_nano_sim_auto.impl.transformer.attention import (
        _fused_merge_supported,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    outputs = torch.randn(5, 2, 3, 7, 4)
    logs = torch.randn(5, 2, 3, 7)

    def merge(out, lse, *, fused):
        return merge_attentions(
            [(out, lse), (torch.ones_like(out), torch.zeros_like(lse))], fused=fused
        )

    expected = torch.vmap(lambda out, lse: merge(out, lse, fused=False))(outputs, logs)
    actual = torch.vmap(lambda out, lse: merge(out, lse, fused=True))(outputs, logs)
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)
    with FakeTensorMode():
        outputs = torch.empty(outputs.shape, device="cuda:0")
        logs = torch.empty(logs.shape, device="cuda:0")

        def check(out, lse):
            assert not _fused_merge_supported([(out, lse)], None)
            return out

        torch.vmap(check)(outputs, logs)


def test_optional_merge_keeps_eager_fake_cuda_on_native_ops(monkeypatch):
    # FakeTensor otherwise initializes a real GPU context when one is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    from cosmos3_nano_sim_auto.impl.transformer import attention
    from torch._subclasses.fake_tensor import FakeTensorMode

    def unexpected_gate(*args):
        raise AssertionError("A Tensor subclass reached raw-storage dispatch")

    with FakeTensorMode():
        out = torch.empty(2, 3, 7, 4, device="cuda:0", dtype=torch.bfloat16)
        lse = torch.empty(2, 3, 7, device="cuda:0")
        parts = [(out, lse)]
        # Metadata qualifies; the public boundary must still use native ops.
        assert attention._fused_merge_supported(parts, None)
        monkeypatch.setattr(attention, "_fused_merge_supported", unexpected_gate)
        expected = merge_attentions(parts, output_dtype=torch.bfloat16)
        actual = merge_attentions(parts, fused=True, output_dtype=torch.bfloat16)
        assert type(actual) is type(expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.stride() == expected.stride()


@pytest.mark.parametrize("subclass_input", ["output", "lse"])
def test_optional_merge_preserves_tensor_subclass_dispatch(monkeypatch, subclass_input):
    from cosmos3_nano_sim_auto.impl.transformer import attention

    class Subclass(torch.Tensor):
        pass

    def unexpected_gate(*args):
        raise AssertionError("A Tensor subclass reached raw-storage dispatch")

    out = torch.randn(2, 3, 7, 4)
    lse = torch.randn(2, 3, 7)
    if subclass_input == "output":
        out = out.as_subclass(Subclass)
    else:
        lse = lse.as_subclass(Subclass)
    parts = [(out, lse)]
    expected = merge_attentions(parts)
    monkeypatch.setattr(attention, "_fused_merge_supported", unexpected_gate)
    actual = merge_attentions(parts, fused=True)
    assert type(actual) is type(expected) is Subclass
    assert actual.stride() == expected.stride()
    assert torch.equal(actual, expected)
