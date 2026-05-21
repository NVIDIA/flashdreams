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

"""Numerical equivalence between eager and CUDA-graph projector.

Drives two ``Causal_LQ4x_Proj`` instances through ``forward_streaming``
with the same chunk sequence -- one eager, one with the CUDA-graph
wrapper enabled -- and asserts projector outputs and end-of-chunk cache
slots match within tolerance. Mirrors the structure of
``flashdreams/tests/taehv/test_taehv_equivalence.py``.

Requires CUDA. The ``LQ_proj_in.ckpt`` weights are resolved through the
production :func:`flashdreams.core.checkpoint.load.load_checkpoint`
path against the HF URL in :data:`AVAILABLE_FLASHVSR_CHECKPOINT_PATHS`,
i.e. cached under ``~/.cache/huggingface/hub/`` -- a previously cached
run or network access is required.

Coverage:

- Parameterised ``eager_fp32``, ``cg_bf16`` and ``compile_cg_bf16`` rows
  that drive 4 chunks through ``forward_streaming``: drain (chunk 0)
  -> warmup (chunk 1) -> capture+replay (chunk 2) -> pure replay
  (chunk 3). 4 chunks is the minimum that hits every distinct phase;
  CUDA-graph replays are deterministic so additional replays add no
  coverage. The ``compile_cg_bf16`` row uses ``mode="default"`` +
  ``dynamic=True`` to skip Inductor's multi-second max-autotune in CI --
  production runs ``mode="max-autotune-no-cudagraphs"``, but the
  wrapper-vs-eager correctness this test cares about is mode-independent.
- ``test_projector_cache_swap_resets_wrapper``: swap the bound cache
  mid-stream; the projector's slot-id check must auto-reset the wrapper
  and re-drain into the new cache.
- ``test_projector_cache_dict_rebind_resets_wrapper``: rebind
  ``cache.cache = {"conv1": None, "conv2": None}`` mid-stream; the
  slot-id check must trigger the same auto-reset.
"""

from __future__ import annotations

import pytest
import torch
from flashvsr.config import AVAILABLE_FLASHVSR_CHECKPOINT_PATHS
from flashvsr.encoder.network import (
    Causal_LQ4x_Proj,
    Causal_LQ4x_Proj_Cache,
)

from flashdreams.core.checkpoint.load import load_checkpoint

pytestmark = pytest.mark.ci_gpu

_PROJECTOR_URL = AVAILABLE_FLASHVSR_CHECKPOINT_PATHS["v1.1-tiny-long"]["encoder"]

_GPU_REASON = "projector CUDA-graph equivalence requires CUDA"


def _build_projector(
    *,
    dtype: torch.dtype,
    device: torch.device,
    use_cuda_graph: bool,
    use_compile: bool = False,
    compile_mode: str = "default",
    compile_dynamic: bool | None = True,
) -> Causal_LQ4x_Proj:
    """Match ``UltraFlashVSRUpsampler.__init__`` (layer_num=1 in production).

    ``compile_mode`` / ``compile_dynamic`` default to the cheap-CI settings
    (skip max-autotune, allow symbolic shapes). Production uses
    ``mode="max-autotune-no-cudagraphs"`` with static shapes -- callers
    that need to validate that exact pipeline should pass the prod values
    explicitly.
    """
    proj = Causal_LQ4x_Proj(
        in_dim=3,
        out_dim=1536,
        layer_num=1,
        use_cuda_graph=use_cuda_graph,
        use_compile=use_compile,
        compile_mode=compile_mode,
        compile_dynamic=compile_dynamic,
    ).to(device=device, dtype=dtype)
    proj.load_state_dict(
        load_checkpoint(_PROJECTOR_URL, map_location="cpu"), strict=True
    )
    proj.eval().requires_grad_(False)
    return proj


def _make_upres(
    *,
    chunk_idx: int,
    chunk_size: int,
    target_H: int,
    target_W: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Mirror the shape `UltraFlashVSRUpsampler.forward` hands the projector."""
    gen = torch.Generator(device=device).manual_seed(seed + chunk_idx)
    return torch.randn(
        (1, 3, chunk_size, target_H, target_W),
        device=device,
        dtype=dtype,
        generator=gen,
    ).clamp_(-1, 1)


def _assert_outputs_close(
    out_a: list[torch.Tensor],
    out_b: list[torch.Tensor],
    atol: float,
    rtol: float,
    label: str,
) -> None:
    assert len(out_a) == len(out_b), (
        f"{label}: linear_layers count mismatch {len(out_a)} vs {len(out_b)}"
    )
    for i, (a, b) in enumerate(zip(out_a, out_b)):
        torch.testing.assert_close(
            a, b, atol=atol, rtol=rtol, msg=lambda m: f"{label}/out[{i}]: {m}"
        )


def _assert_cache_close(
    cache_a: Causal_LQ4x_Proj_Cache,
    cache_b: Causal_LQ4x_Proj_Cache,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    for key in ("conv1", "conv2"):
        a = cache_a.cache[key]
        b = cache_b.cache[key]
        assert a is not None and b is not None, f"{label}/{key}: slot is None"
        torch.testing.assert_close(
            a, b, atol=atol, rtol=rtol, msg=lambda m: f"{label}/cache[{key}]: {m}"
        )


# (mode_id, dtype, use_cuda_graph_for_b, use_compile_for_b, atol, rtol)
#
# - "eager_fp32": both projectors run eager fp32 -- catches any drift the
#   eager-vs-eager path picks up from independent kernel launches. Tight
#   tolerances since both paths are bit-equivalent in principle.
# - "cg_bf16": projector B uses the wrapper at bf16. Loose tolerances tolerate
#   bf16 noise plus any numeric ordering difference the static-buffer
#   ``copy_`` introduces during input staging.
# - "compile_cg_bf16": projector B uses ``use_compile=True`` plus the wrapper
#   at bf16. Uses the test-default ``mode="default"`` + ``dynamic=True``
#   compile settings (see :func:`_build_projector`) to keep CI runtime
#   bounded; the wrapper-vs-eager correctness is mode-independent, and
#   widening the tolerance absorbs whatever reduction-order / fused-kernel
#   differences Inductor's default mode introduces on top of bf16 noise.
_MODES: list[tuple[str, torch.dtype, bool, bool, float, float]] = [
    ("eager_fp32", torch.float32, False, False, 1e-5, 1.3e-6),
    ("cg_bf16", torch.bfloat16, True, False, 1e-2, 1e-2),
    ("compile_cg_bf16", torch.bfloat16, True, True, 5e-2, 5e-2),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.no_grad()
@pytest.mark.parametrize("chunk_size", [8, 16])
@pytest.mark.parametrize(
    ("mode_id", "dtype", "use_cuda_graph_b", "use_compile_b", "atol", "rtol"),
    _MODES,
    ids=[m[0] for m in _MODES],
)
def test_projector_streaming_equivalence(
    mode_id: str,
    dtype: torch.dtype,
    use_cuda_graph_b: bool,
    use_compile_b: bool,
    atol: float,
    rtol: float,
    chunk_size: int,
) -> None:
    """Eager and graph projectors agree across drain -> capture -> replay.

    Runs 4 chunks per mode at fixed ``chunk_size``: chunk 0 fills the
    cache (drain), chunk 1 warms up, chunk 2 captures + first replay,
    chunk 3 is a pure replay. CUDA-graph replays are deterministic so
    additional replays add no coverage past this minimum.
    """
    device = torch.device("cuda")
    target_H, target_W = 384 * 2, 640 * 2

    proj_eager = _build_projector(dtype=dtype, device=device, use_cuda_graph=False)
    proj_b = _build_projector(
        dtype=dtype,
        device=device,
        use_cuda_graph=use_cuda_graph_b,
        use_compile=use_compile_b,
    )

    cache_eager = proj_eager.create_external_cache()
    cache_b = proj_b.create_external_cache()

    for chunk_idx in range(4):
        upres = _make_upres(
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            target_H=target_H,
            target_W=target_W,
            dtype=dtype,
            device=device,
            seed=0,
        )
        out_eager = proj_eager.forward_streaming(upres.clone(), cache_eager)
        out_b = proj_b.forward_streaming(upres.clone(), cache_b)
        torch.cuda.synchronize()

        label = f"{mode_id}/cs={chunk_size}/chunk{chunk_idx}"
        _assert_outputs_close(out_eager, out_b, atol, rtol, label)
        _assert_cache_close(cache_eager, cache_b, atol, rtol, label)


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.no_grad()
def test_projector_cache_swap_resets_wrapper() -> None:
    """A new external cache mid-stream must auto-invalidate the captured graph.

    Without the slot-id check, the wrapper would keep replaying against
    the prior cache's slot pointers and silently produce wrong results.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    chunk_size = 16
    target_H, target_W = 384 * 2, 640 * 2
    atol, rtol = 1e-2, 1e-2

    proj_eager = _build_projector(dtype=dtype, device=device, use_cuda_graph=False)
    proj_g = _build_projector(dtype=dtype, device=device, use_cuda_graph=True)

    cache_eager = proj_eager.create_external_cache()
    cache_g = proj_g.create_external_cache()

    # Drive 5 chunks so the graph has been captured + replayed at least once.
    for chunk_idx in range(5):
        upres = _make_upres(
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            target_H=target_H,
            target_W=target_W,
            dtype=dtype,
            device=device,
            seed=0,
        )
        out_eager = proj_eager.forward_streaming(upres.clone(), cache_eager)
        out_g = proj_g.forward_streaming(upres.clone(), cache_g)
        torch.cuda.synchronize()
        _assert_outputs_close(
            out_eager, out_g, atol, rtol, f"pre-swap/chunk{chunk_idx}"
        )

    # Swap to a fresh cache on each side. The graph projector's slot-id
    # check must detect this and call wrapper.reset() before the next
    # forward; otherwise replay would feed in stale slot pointers.
    cache_eager = proj_eager.create_external_cache()
    cache_g = proj_g.create_external_cache()

    for chunk_idx in range(5, 10):
        upres = _make_upres(
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            target_H=target_H,
            target_W=target_W,
            dtype=dtype,
            device=device,
            seed=0,
        )
        out_eager = proj_eager.forward_streaming(upres.clone(), cache_eager)
        out_g = proj_g.forward_streaming(upres.clone(), cache_g)
        torch.cuda.synchronize()
        _assert_outputs_close(
            out_eager, out_g, atol, rtol, f"post-swap/chunk{chunk_idx}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@torch.no_grad()
def test_projector_cache_dict_rebind_resets_wrapper() -> None:
    """Rebinding ``cache.cache = {...}`` mid-stream must also reset the wrapper.

    This is the failure mode the dropped ``Causal_LQ4x_Proj_Cache.clear_cache``
    helper used to expose: same dataclass id, but new inner dict with
    ``None`` slots. ``id(proj_cache)``-based tracking would miss it; the
    slot-id tuple catches it.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    chunk_size = 16
    target_H, target_W = 384 * 2, 640 * 2
    atol, rtol = 1e-2, 1e-2

    proj_eager = _build_projector(dtype=dtype, device=device, use_cuda_graph=False)
    proj_g = _build_projector(dtype=dtype, device=device, use_cuda_graph=True)

    cache_eager = proj_eager.create_external_cache()
    cache_g = proj_g.create_external_cache()

    for chunk_idx in range(4):
        upres = _make_upres(
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            target_H=target_H,
            target_W=target_W,
            dtype=dtype,
            device=device,
            seed=0,
        )
        proj_eager.forward_streaming(upres.clone(), cache_eager)
        proj_g.forward_streaming(upres.clone(), cache_g)
        torch.cuda.synchronize()

    # Rebind the inner dict (simulating a manual reset by a downstream
    # caller). Same Causal_LQ4x_Proj_Cache id; new slot ids.
    cache_eager.cache = {"conv1": None, "conv2": None}
    cache_g.cache = {"conv1": None, "conv2": None}

    for chunk_idx in range(4, 8):
        upres = _make_upres(
            chunk_idx=chunk_idx,
            chunk_size=chunk_size,
            target_H=target_H,
            target_W=target_W,
            dtype=dtype,
            device=device,
            seed=0,
        )
        out_eager = proj_eager.forward_streaming(upres.clone(), cache_eager)
        out_g = proj_g.forward_streaming(upres.clone(), cache_g)
        torch.cuda.synchronize()
        _assert_outputs_close(
            out_eager, out_g, atol, rtol, f"post-rebind/chunk{chunk_idx}"
        )
        _assert_cache_close(
            cache_eager, cache_g, atol, rtol, f"post-rebind/chunk{chunk_idx}"
        )
