"""Numerical equivalence between reference and slim TAEHV decode.

Requires GPU + network (downloads the lighttae checkpoint from S3) and
is therefore marked ``@pytest.mark.manual`` -- opt in via
``pytest -m manual ...``. Compares the upstream reference :class:`TAEHV`
in the sibling :mod:`.impl_reference` module against the rewrite in
:mod:`flashdreams.recipes.taehv.impl` on a streaming causal decode (5
same-shape body chunks).

The default ``_MODES`` table only exercises ``eager`` (no compile, no
CUDA graph). Add a ``compile_cg`` row to also smoke-test the
warmup -> capture -> replay path of :class:`CUDAGraphWrapper` (use loose
bf16 tolerances).
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.recipes.taehv import AVAILABLE_TAEHV_CHECKPOINT_PATHS
from flashdreams.recipes.taehv import impl as _impl_new
from flashdreams.recipes.taehv.impl import TAEHV as TAEHVNew

from . import impl_reference as _impl_reference

TAEHVLegacy = _impl_reference.TAEHV


def _build_pair(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_compile: bool,
    use_cuda_graph: bool,
) -> tuple[TAEHVLegacy, TAEHVNew]:
    """Construct legacy and slim TAEHV from a single shared checkpoint.

    The checkpoint is loaded once, cast to ``dtype``, and re-served via
    a patch on each impl module's ``load_checkpoint`` so both models
    see identical weights without a second S3 round-trip. Each call
    returns a *fresh shallow copy* of the dict, because the legacy
    ``patch_tgrow_layers`` mutates entries in place -- otherwise the
    new impl would see the already-truncated TGrow weights when the
    legacy constructor runs first.
    """
    weights = load_checkpoint(checkpoint_path)
    weights = {k: v.to(dtype) for k, v in weights.items()}  # ty:ignore[call-non-callable]

    def _cached(_path):
        return copy.copy(weights)

    with (
        patch.object(_impl_reference, "load_checkpoint", _cached),
        patch.object(_impl_new, "load_checkpoint", _cached),
    ):
        legacy = TAEHVLegacy(checkpoint_path=checkpoint_path).to(
            device=device, dtype=dtype
        )
        new = TAEHVNew(
            checkpoint_path=checkpoint_path,
            use_cuda_graph=use_cuda_graph,
            use_compile=use_compile,
        ).to(device=device, dtype=dtype)
    return legacy, new


# (mode_id, dtype, use_compile, use_cuda_graph, atol, rtol)
#
# - "eager" runs in fp32 with tight (but not bit-exact) tolerances:
#   isolates the impl-vs-legacy logic. The new impl uses different
#   reduction orders for some reshapes (e.g. ``view`` instead of
#   ``reshape`` paths) so float accumulations can drift by a few ULPs;
#   ``1e-5 / 1.3e-6`` keeps real divergences (>= ~5 ULPs of bf16) easy
#   to spot while tolerating the noise.
_MODES: list[tuple[str, torch.dtype, bool, bool, float, float]] = [
    ("eager", torch.float32, False, False, 1e-5, 1.3e-6),
]


@pytest.mark.manual
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TAEHV equivalence test requires GPU"
)
@torch.no_grad()
@pytest.mark.parametrize("checkpoint_key", ["lighttae"])
@pytest.mark.parametrize(
    ("mode_id", "dtype", "use_compile", "use_cuda_graph", "atol", "rtol"),
    _MODES,
    ids=[m[0] for m in _MODES],
)
def test_taehv_streaming_equivalence(
    checkpoint_key: str,
    mode_id: str,
    dtype: torch.dtype,
    use_compile: bool,
    use_cuda_graph: bool,
    atol: float,
    rtol: float,
) -> None:
    """Legacy and slim TAEHV must match on streaming decode.

    Streaming schedule:
      - Decode rollout 1 (chunk A): T=2 latents -> 5 frames (after
        ``frames_to_trim=3``). Bare / drain path.
      - Decode rollouts 2-5 (chunks B0..B3): T=2 latents -> 8 frames
        each. With a ``compile_cg`` row (not in the default ``_MODES``)
        the wrapper would do 2 warmup + 1 capture + 1 replay over these
        4 calls.
    """
    device = torch.device("cuda")
    checkpoint_path = AVAILABLE_TAEHV_CHECKPOINT_PATHS[checkpoint_key]

    legacy, new = _build_pair(
        checkpoint_path,
        dtype,
        device,
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )

    torch.manual_seed(0)
    # 5 chunks of T=2 latents (10 total) at a small spatial size.
    latents = torch.empty(1, 10, 16, 32, 32, dtype=dtype, device=device).uniform_(-1, 1)

    cache_legacy = legacy.prepare_cache()
    cache_new = new.prepare_cache()

    def _close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{mode_id} / {checkpoint_key} / {label}: {m}",
        )

    # First chunk: 2 latents -> 5 frames (8 - frames_to_trim=3).
    z_a = latents[:, :2]
    out_legacy_a = legacy.decode_video(z_a, parallel=True, cache=cache_legacy)
    out_new_a = new.decode(z_a, cache=cache_new)
    assert out_legacy_a.shape[1] == 5, (
        f"Expected 5 frames after first decode, got {out_legacy_a.shape[1]}"
    )
    _close(out_new_a, out_legacy_a, "decode A")

    # Steady-state body chunks: 4 calls of T=2 latents -> 8 frames each.
    for k in range(4):
        z_chunk = latents[:, 2 + 2 * k : 2 + 2 * (k + 1)]
        out_legacy_k = legacy.decode_video(z_chunk, parallel=True, cache=cache_legacy)
        out_new_k = new.decode(z_chunk, cache=cache_new)
        assert out_legacy_k.shape[1] == 8, (
            f"Expected 8 frames in chunk B[{k}], got {out_legacy_k.shape[1]}"
        )
        _close(out_new_k, out_legacy_k, f"decode B[{k}]")
