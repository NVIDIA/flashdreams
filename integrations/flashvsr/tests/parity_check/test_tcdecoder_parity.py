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

"""TC decoder parity between upstream FlashVSR and FlashDreams.

The legacy reference is loaded directly from upstream's
``examples/WanVSR/utils/TCDecoder.py`` inside the parity-check sibling
tree (``./FlashVSR/...``, staged by ``run.sh``); the candidate is the
live :class:`flashvsr.decoder.network.FlashVSR_TAEHV`. Both load the
same ``TCDecoder.ckpt`` and we compare chunk-by-chunk plus run a CUDA
graph capture smoke test on the candidate.

Upstream's ``TAEHV.forward`` raises ``NotImplementedError`` and its
``parallel=True`` ``decode_video`` path doesn't carry mem across calls
(the local ``mem`` is reassigned inside ``apply_model_with_memblocks``
and never written back to the per-MemBlock list). The test therefore
drives the legacy side via ``decode_video(..., parallel=False)``, which
maintains ``self.mem`` causally and matches the candidate's
``parallel=True`` + cache path frame-for-frame.

Per-frame (legacy) vs per-chunk (candidate) batching feeds cuDNN conv
kernels at different batch sizes, so cuDNN picks different algorithms
and accumulates in different orders. Chunk parity is therefore
asserted at ``atol=2.5e-3 / rtol=1e-3`` (fp32 cross-algorithm conv
tolerance), not bit-for-bit -- see the inline comment in
``test_tcdecoder_chunk_parity`` for the calibration. The candidate
CUDA-graph vs eager smoke test stays at ``1e-5`` because both sides
share the same impl and only differ in launch path.

Skipped automatically when the upstream tree (run ``bash run.sh`` next
to this file) or the FlashVSR-v1.1 weight dir is absent. Set
``$FLASHVSR_WEIGHTS_ROOT`` (default ``~/.cache/flashdreams/upsampler/weights``)
to override the staging root.

The test is invoked from ``run.sh`` via this directory's parity-check
venv; the candidate side (``flashvsr``) is layered on as an editable
install via ``flashdreams-flashvsr = { path = "../.." }`` in
``pyproject.toml``, alongside the legacy ``diffsynth`` install.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from flashvsr.decoder import tcdecoder_state_dict_transform
from flashvsr.decoder.network import FlashVSR_TAEHV as CandidateTAEHV

_HERE = Path(__file__).resolve().parent
_UPSTREAM_TCDECODER = (
    _HERE / "FlashVSR" / "examples" / "WanVSR" / "utils" / "TCDecoder.py"
)
_DEFAULT_WEIGHTS_ROOT = "~/.cache/flashdreams/upsampler/weights"
_WEIGHTS_ROOT = Path(
    os.environ.get("FLASHVSR_WEIGHTS_ROOT", _DEFAULT_WEIGHTS_ROOT)
).expanduser()
_MODEL_NAME = "FlashVSR-v1.1"
_TCDECODER_CKPT = _WEIGHTS_ROOT / _MODEL_NAME / "TCDecoder.ckpt"

FLASHVSR_CHANNELS = (512, 256, 128, 128)
FLASHVSR_LATENT_CHANNELS = 16 + 768
FLASHVSR_CONDITION_PATCH = (4, 8, 8)

_GPU_REASON = "TC decoder parity requires CUDA"
_CKPT_REASON = (
    f"FlashVSR TCDecoder.ckpt not found at {_TCDECODER_CKPT}; "
    f"set $FLASHVSR_WEIGHTS_ROOT or stage with download_flashvsr_weights.sh."
)
_UPSTREAM_REASON = (
    f"Upstream FlashVSR tree not found at {_UPSTREAM_TCDECODER}; "
    f"run ``bash run.sh`` next to this test to clone the pinned commit."
)


def _load_upstream_taehv() -> type:
    """Load upstream's ``examples/WanVSR/utils/TCDecoder.py`` as a module.

    The upstream file is self-contained (only ``torch`` / ``einops`` /
    ``tqdm`` / stdlib), so a raw ``spec_from_file_location`` import is
    enough -- we don't need to mount the surrounding repo on
    ``sys.path``.
    """
    spec = importlib.util.spec_from_file_location(
        "flashvsr_upstream_tcdecoder", _UPSTREAM_TCDECODER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {_UPSTREAM_TCDECODER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TAEHV


def _new_legacy(LegacyTAEHV: type) -> Any:
    """Return type is ``Any`` because ``LegacyTAEHV`` comes from
    ``importlib.spec_from_file_location`` and ty cannot resolve its
    methods. Without this every ``legacy.decode_video(...)`` /
    ``legacy.clean_mem()`` call site would resolve through
    ``nn.Module.__getattr__`` and need a type-ignore.
    """
    return LegacyTAEHV(
        checkpoint_path=str(_TCDECODER_CKPT),
        channels=list(FLASHVSR_CHANNELS),
        latent_channels=FLASHVSR_LATENT_CHANNELS,
    )


def _new_candidate(*, use_cuda_graph: bool = False) -> CandidateTAEHV:
    return CandidateTAEHV(
        checkpoint_path=str(_TCDECODER_CKPT),
        channels=FLASHVSR_CHANNELS,
        latent_channels=FLASHVSR_LATENT_CHANNELS,
        use_cuda_graph=use_cuda_graph,
        # The legacy-keys remap used to be applied unconditionally inside
        # ``TAEHV.load_state_dict``; it now lives next to the FlashVSR
        # checkpoint URL in ``flashvsr.decoder`` and direct instantiations
        # (like this one) opt in explicitly.
        state_dict_transform=tcdecoder_state_dict_transform,
    )


def _flashdreams_key(key: str) -> str:
    if key.startswith("decoder.") and not key.startswith("decoder.blocks."):
        return key.replace("decoder.", "decoder.blocks.", 1)
    return key


def _make_inputs(
    *,
    chunks: int,
    batch: int,
    latent_time: int,
    latent_height: int,
    latent_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    gen = torch.Generator(device="cpu").manual_seed(1234)
    items = []
    for _ in range(chunks):
        z = torch.randn(
            batch,
            latent_time,
            16,
            latent_height,
            latent_width,
            generator=gen,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        cond = torch.randn(
            batch,
            3,
            latent_time * FLASHVSR_CONDITION_PATCH[0],
            latent_height * FLASHVSR_CONDITION_PATCH[1],
            latent_width * FLASHVSR_CONDITION_PATCH[2],
            generator=gen,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        items.append((z, cond))
    return items


@pytest.mark.skipif(not _UPSTREAM_TCDECODER.exists(), reason=_UPSTREAM_REASON)
@pytest.mark.skipif(not _TCDECODER_CKPT.exists(), reason=_CKPT_REASON)
def test_tcdecoder_state_dict_shapes_match() -> None:
    """The candidate state dict matches the checkpoint after the legacy-key remap."""
    LegacyTAEHV = _load_upstream_taehv()
    state = torch.load(_TCDECODER_CKPT, map_location="cpu")

    legacy = _new_legacy(LegacyTAEHV)
    candidate = _new_candidate()

    for label, model_state, ckpt in (
        ("upstream TCDecoder", legacy.state_dict(), state),
        (
            "FlashDreams TAEHV",
            candidate.state_dict(),
            {_flashdreams_key(k): v for k, v in state.items()},
        ),
    ):
        missing = sorted(k for k in model_state if k not in ckpt)
        unexpected = sorted(k for k in ckpt if k not in model_state)
        mismatched = sorted(
            k
            for k in model_state.keys() & ckpt.keys()
            if tuple(model_state[k].shape) != tuple(ckpt[k].shape)
        )
        assert not missing, f"{label}: missing keys vs checkpoint: {missing[:8]}"
        assert not unexpected, f"{label}: unexpected keys: {unexpected[:8]}"
        assert not mismatched, f"{label}: shape mismatches: {mismatched[:8]}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.skipif(not _UPSTREAM_TCDECODER.exists(), reason=_UPSTREAM_REASON)
@pytest.mark.skipif(not _TCDECODER_CKPT.exists(), reason=_CKPT_REASON)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tcdecoder_chunk_parity(dtype: torch.dtype) -> None:
    """Upstream and candidate TC decoders match chunk-by-chunk on real weights."""
    device = torch.device("cuda")
    LegacyTAEHV = _load_upstream_taehv()

    legacy = (
        _new_legacy(LegacyTAEHV)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    candidate = (
        _new_candidate().to(device=device, dtype=dtype).eval().requires_grad_(False)
    )

    legacy.clean_mem()
    candidate_cache = candidate.prepare_cache()
    inputs = _make_inputs(
        chunks=2,
        batch=1,
        latent_time=2,
        latent_height=44,
        latent_width=80,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for idx, (z, cond) in enumerate(inputs):
            # Upstream's ``parallel=True`` path doesn't carry mem across
            # calls (``mem`` is rebound to a local tensor inside
            # ``apply_model_with_memblocks`` and never written back to the
            # per-MemBlock list). ``parallel=False`` walks the work queue
            # frame-by-frame and updates ``self.mem[i]`` in-place, which
            # matches the candidate's ``parallel=True`` + ``cache`` path
            # frame-for-frame.
            legacy_out = legacy.decode_video(z, parallel=False, cond=cond)
            candidate_out = candidate(
                z,
                parallel=True,
                show_progress_bar=False,
                cond=cond,
                cache=candidate_cache,
            )
            # Tolerance is fp32 cross-algorithm conv noise, not a sloppy
            # parity check. Both paths compute the same math (verified by
            # tracing the work-queue / TGrow split ordering: legacy emits
            # ``[t0_a, t0_b, t1_a, t1_b, ...]`` and candidate's flat
            # reshape produces the same order; the ``past`` buffer rule --
            # ``past[i] = input[i-1]`` with prev-chunk-last as left
            # context -- agrees frame-for-frame). The only structural
            # difference is conv batch size: legacy ``parallel=False``
            # drives every conv at ``batch=1`` (depth-first frame-at-a-
            # time work queue), while the candidate runs
            # ``batch=b*t*TGrow_stride_accum`` per conv (here 2 -> 4 -> 8
            # as the two stride-2 ``TGrow``s upsample temporally). cuDNN
            # selects different kernels and accumulators reduce in
            # different orders for each batch size, and the ~20-conv-deep
            # decoder amplifies that ULP-level noise. Calibrated against
            # an observed ``max_abs=1.93e-3`` / ``mean_abs=8.4e-5`` on a
            # DGX H100 box; ``atol=2.5e-3 / rtol=1e-3`` adds ~30% headroom
            # for cuDNN-version / GPU-generation drift while still
            # catching a genuine algorithmic regression (output range is
            # ~[-0.1, 0.7], so the relative bound stays around 0.5%).
            # If this gets noticeably tighter once both paths share a
            # batched-streaming kernel, ratchet it back down.
            torch.testing.assert_close(
                candidate_out.float(),
                legacy_out.float(),
                atol=2.5e-3,
                rtol=1e-3,
                msg=f"chunk {idx} TC decoder parity failed",
            )

    legacy_slots = sum(1 for value in legacy.mem if value is not None)
    candidate_slots = sum(
        1 for value in candidate_cache.dec_state.values() if value is not None
    )
    assert legacy_slots == candidate_slots, (
        f"cache slot count mismatch: legacy={legacy_slots} candidate={candidate_slots}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.skipif(not _TCDECODER_CKPT.exists(), reason=_CKPT_REASON)
def test_tcdecoder_cuda_graph_smoke() -> None:
    """The CUDA-graph wrapper captures by chunk 4 and matches the eager path."""
    device = torch.device("cuda")
    dtype = torch.float32

    eager = _new_candidate().to(device=device, dtype=dtype).eval().requires_grad_(False)
    graphed = (
        _new_candidate(use_cuda_graph=True)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )

    eager_cache = eager.prepare_cache()
    graph_cache = graphed.prepare_cache()
    inputs = _make_inputs(
        chunks=4,
        batch=1,
        latent_time=2,
        latent_height=44,
        latent_width=80,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        for idx, (z, cond) in enumerate(inputs):
            eager_out = eager(z, cond=cond, cache=eager_cache)
            graph_out = graphed(z, cond=cond, cache=graph_cache)
            torch.cuda.synchronize()
            diff = (eager_out - graph_out).float().abs()
            assert torch.allclose(
                eager_out.float(),
                graph_out.float(),
                atol=1e-5,
                rtol=1e-5,
            ), f"chunk {idx} graph parity failed: max_abs={diff.max().item():.6g}"

    wrapper = graphed._decoder_wrapper
    assert wrapper is not None and wrapper._graph is not None, (
        "CUDA graph did not capture by the end of the smoke test"
    )
