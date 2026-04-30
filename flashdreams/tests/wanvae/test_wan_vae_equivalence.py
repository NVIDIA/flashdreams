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

"""Numerical equivalence between reference and slim Wan VAE.

Requires GPU. Marked ``@pytest.mark.manual`` -- opt in via
``pytest -m manual ...``; the default ``tests/run_tests_local.sh`` runs
with ``-m "not manual"`` and skips it. Intended to be re-run manually
after each refactor step. Compares the upstream reference
:class:`WanVAE` in the sibling :mod:`.impl_reference` module against the
rewrite in :mod:`flashdreams.recipes.wan.autoencoder.vae` on a streaming
causal encode + decode.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.recipes.wan.autoencoder import vae as _impl_new
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAE as WanVAENew,
)

from . import impl_reference as _impl_reference

WanVAELegacy = _impl_reference.WanVAE


def _build_pair(
    checkpoint_path: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    use_compile: bool,
    use_cuda_graph: bool,
) -> tuple[WanVAELegacy, WanVAENew]:
    """Construct legacy and slim WanVAE from a single shared checkpoint.

    The checkpoint is loaded once, upcast to ``dtype`` (bf16 on disk -> fp32
    here), and re-served via a patch on each impl module's ``load_checkpoint``
    so both models see identical weights without a second S3 round-trip.
    """
    use_lightvae = "lightvae" in checkpoint_path
    weights = load_checkpoint(checkpoint_path)
    weights = {k: v.to(dtype) for k, v in weights.items()}  # ty:ignore[call-non-callable]

    def _cached(_path):
        return weights

    with (
        patch.object(_impl_reference, "load_checkpoint", _cached),
        patch.object(_impl_new, "load_checkpoint", _cached),
    ):
        legacy = WanVAELegacy(
            vae_path=checkpoint_path,
            use_lightvae=use_lightvae,
            dtype=dtype,
            device=device,
        )
        new = WanVAENew(
            vae_path=checkpoint_path,
            use_lightvae=use_lightvae,
            use_compile=use_compile,
            use_cuda_graph=use_cuda_graph,
        ).to(device=device, dtype=dtype)
    return legacy, new


# (mode_id, dtype, use_compile, use_cuda_graph, atol, rtol)
#
# - "eager" runs in fp32 with strict bit-exact tolerances: this isolates
#   the impl-vs-legacy logic. Inductor reorders accumulations and CUDA
#   graph capture introduces no additional noise here, so any diff is a
#   real impl bug.
_MODES: list[tuple[str, torch.dtype, bool, bool, float, float]] = [
    ("eager", torch.float32, False, False, 1e-5, 1.3e-6),
]


@pytest.mark.manual
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Wan VAE equivalence test requires GPU"
)
@torch.no_grad()
@pytest.mark.parametrize("checkpoint_key", ["lightvae", "vae"])
@pytest.mark.parametrize(
    ("mode_id", "dtype", "use_compile", "use_cuda_graph", "atol", "rtol"),
    _MODES,
    ids=[m[0] for m in _MODES],
)
def test_wan_vae_streaming_equivalence(
    checkpoint_key: str,
    mode_id: str,
    dtype: torch.dtype,
    use_compile: bool,
    use_cuda_graph: bool,
    atol: float,
    rtol: float,
) -> None:
    """Legacy and slim WanVAE must match on streaming encode + decode.

    Streaming schedule -- chosen to exercise both rollouts in
    :class:`WanVAE`:

      - Encode rollout 1 (chunk A): 17 video frames (1 seed + 4 body
        chunks of 4) -> 5 latents. Bare compiled encoder.
      - Encode rollout 2 (chunk B): 32 video frames (8 body chunks
        of 4) -> 8 latents. Wrapped path -> 2 warmup + 1 capture + 5
        replays for ``compile_cg``.
      - Decode rollout 1 (chunk A): first 5 latents. Bare compiled
        decoder.
      - Decode rollout 2-5 (4 chunks of 2 latents): remaining 8
        latents, split into 4 same-shape calls so the wrapped decoder
        sees 2 warmup + 1 capture + 1 replay for ``compile_cg``.
    """
    device = torch.device("cuda")
    checkpoint_path = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS[checkpoint_key]

    legacy, new = _build_pair(
        checkpoint_path,
        dtype,
        device,
        use_compile=use_compile,
        use_cuda_graph=use_cuda_graph,
    )

    torch.manual_seed(0)
    video = torch.empty(1, 3, 49, 64, 64, dtype=dtype, device=device).uniform_(-1, 1)

    def _close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda m: f"{mode_id} / {checkpoint_key} / {label}: {m}",
        )

    cache_legacy_enc = legacy.prepare_cache()
    cache_new_enc = new.prepare_cache()
    enc_a = video[:, :, :17]
    enc_b = video[:, :, 17:]

    z_legacy_a = legacy.encode(enc_a, cache=cache_legacy_enc)
    z_new_a = new.encode(enc_a, cache=cache_new_enc)
    assert z_legacy_a.shape[2] == 5, f"Expected 5 latents, got {z_legacy_a.shape[2]}"
    _close(z_new_a, z_legacy_a, "encode A")

    z_legacy_b = legacy.encode(enc_b, cache=cache_legacy_enc)
    z_new_b = new.encode(enc_b, cache=cache_new_enc)
    assert z_legacy_b.shape[2] == 8, f"Expected 8 latents, got {z_legacy_b.shape[2]}"
    _close(z_new_b, z_legacy_b, "encode B")

    z_legacy = torch.cat([z_legacy_a, z_legacy_b], dim=2)
    z_new = torch.cat([z_new_a, z_new_b], dim=2)

    cache_legacy_dec = legacy.prepare_cache()
    cache_new_dec = new.prepare_cache()

    # Decode rollout 1 (single 5-latent call -> bare compiled).
    dec_a_legacy = legacy.decode(z_legacy[:, :, :5], cache=cache_legacy_dec)
    dec_a_new = new.decode(z_new[:, :, :5], cache=cache_new_dec)
    _close(dec_a_new, dec_a_legacy, "decode A")

    # Decode rollouts 2-5 (4 same-shape calls of 2 latents). The
    # ``compile_cg`` wrapper warms up for 2 calls, captures on the 3rd,
    # and replays on the 4th -- this exercises the capture+replay path
    # for decode.
    for k in range(4):
        z_chunk_legacy = z_legacy[:, :, 5 + 2 * k : 5 + 2 * (k + 1)]
        z_chunk_new = z_new[:, :, 5 + 2 * k : 5 + 2 * (k + 1)]
        dec_legacy_k = legacy.decode(z_chunk_legacy, cache=cache_legacy_dec)
        dec_new_k = new.decode(z_chunk_new, cache=cache_new_dec)
        _close(dec_new_k, dec_legacy_k, f"decode B[{k}]")
