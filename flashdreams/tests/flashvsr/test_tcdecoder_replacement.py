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

"""Bit-for-bit TC decoder parity between the legacy module and FlashDreams TAEHV.

Loads the frozen legacy reference at ``_tcdecoder.py`` (sibling) and the
live :class:`flashdreams.recipes.flashvsr.decoder.network.TAEHV` candidate
side-by-side, loads the same ``TCDecoder.ckpt`` into both, and checks
chunk-by-chunk numerical agreement plus a CUDA-graph capture smoke test.

Marked ``manual`` and skipped automatically when the FlashVSR-v1.1 weight
dir is absent. Set ``$FLASHVSR_WEIGHTS_ROOT`` (default
``~/.cache/flashdreams/upsampler/weights``) to override the staging root.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest
import torch

from flashdreams.recipes.flashvsr.decoder.network import TAEHV as CandidateTAEHV

_LEGACY_REF_PATH = Path(__file__).resolve().parent / "_tcdecoder.py"
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


def _load_legacy_taehv():
    """Load the frozen ``_tcdecoder.py`` sibling without packaging it.

    The module is fully self-contained (no ``flashdreams`` imports), so a
    raw spec_from_file_location load is enough.
    """
    spec = importlib.util.spec_from_file_location(
        "flashvsr_legacy_tcdecoder", _LEGACY_REF_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {_LEGACY_REF_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TAEHV


def _new_legacy(LegacyTAEHV: type) -> Any:
    """Return type is ``Any`` because ``LegacyTAEHV`` comes from
    ``importlib.spec_from_file_location`` and ty cannot resolve its
    methods. Without this every ``legacy.prepare_cache()`` /
    ``legacy(...)`` call site would resolve through
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


def _cache_ptrs(cache: object) -> list[int]:
    if hasattr(cache, "mem"):
        values: Iterable[torch.Tensor | None] = getattr(cache, "mem")
        return [0 if value is None else value.data_ptr() for value in values]
    if hasattr(cache, "dec_state"):
        state = getattr(cache, "dec_state")
        return [value.data_ptr() for _, value in sorted(state.items())]
    return []


@pytest.mark.skipif(not _TCDECODER_CKPT.exists(), reason=_CKPT_REASON)
def test_tcdecoder_state_dict_shapes_match() -> None:
    """The candidate state dict matches the checkpoint after the legacy-key remap."""
    LegacyTAEHV = _load_legacy_taehv()
    state = torch.load(_TCDECODER_CKPT, map_location="cpu")

    legacy = _new_legacy(LegacyTAEHV)
    candidate = _new_candidate()

    for label, model_state, ckpt in (
        ("legacy TCDecoder", legacy.state_dict(), state),
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
@pytest.mark.skipif(not _TCDECODER_CKPT.exists(), reason=_CKPT_REASON)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tcdecoder_chunk_parity(dtype: torch.dtype) -> None:
    """Legacy and candidate TC decoders match chunk-by-chunk on real weights."""
    device = torch.device("cuda")
    LegacyTAEHV = _load_legacy_taehv()

    legacy = (
        _new_legacy(LegacyTAEHV)
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    candidate = (
        _new_candidate().to(device=device, dtype=dtype).eval().requires_grad_(False)
    )

    legacy_cache = legacy.prepare_cache()
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
            legacy_out = legacy(
                z,
                parallel=True,
                show_progress_bar=False,
                cond=cond,
                cache=legacy_cache,
            )
            candidate_out = candidate(
                z,
                parallel=True,
                show_progress_bar=False,
                cond=cond,
                cache=candidate_cache,
            )
            diff = (legacy_out - candidate_out).float().abs()
            assert torch.allclose(
                legacy_out.float(),
                candidate_out.float(),
                atol=1e-5,
                rtol=1e-5,
            ), (
                f"chunk {idx} parity failed: "
                f"max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}"
            )

    legacy_slots = sum(ptr != 0 for ptr in _cache_ptrs(legacy_cache))
    candidate_slots = sum(ptr != 0 for ptr in _cache_ptrs(candidate_cache))
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
