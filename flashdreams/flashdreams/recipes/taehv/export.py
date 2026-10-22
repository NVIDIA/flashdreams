# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a TAEHV decoder to a cache-as-IO ONNX for browser (WebGPU) decode.

The TAEHV decoder keeps a per-``MemBlock`` temporal cache in an
``id(module)``-keyed dict, which is not representable in ONNX. This module
wraps the decoder so that cache is exposed as an ordered list of explicit
tensor inputs (``cache_in_*``) and outputs (``cache_out_*``), letting a
stateless runtime (onnxruntime-web) thread the cache from one chunk to the
next. A non-empty cache selects the steady-state path (no first-chunk trim).

Latent de-normalization, the decode, and the ``[0, 1] -> [-1, 1]`` remap are
all baked into the exported graph, so the client feeds the raw latent and
receives displayable RGB. Validated for multi-chunk continuity by the 1a
spike (bit-exact wrapper vs. native streaming; flat, non-drifting ONNX PSNR).

This lives in the recipe layer because the cache structure is TAEHV-specific;
the generic ``flashdreams.serving.token_stream`` package must not import it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch


def _memblocks(decoder: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the decoder's ``MemBlock`` cache-holding submodules, in order."""
    from flashdreams.recipes.taehv.impl import MemBlock

    # nn.Module attribute access is typed as ``Tensor | Module``; the nested
    # submodule walk is dynamic, so cast to sidestep the union.
    blocks = cast(Any, decoder).taehv.decoder.blocks
    return [b for b in blocks if isinstance(b, MemBlock)]


class _StreamingExportWrapper(torch.nn.Module):
    """Decoder with the temporal cache reshaped as explicit tensor I/O.

    inputs:  latent [B, V, T, Cl, Hl, Wl], cache_in_0..N-1 [B*V, 1, C_i, H_i, W_i]
    outputs: rgb [B, V, Tout, 3, H, W] in [-1, 1], cache_out_0..N-1

    The clone keeps the graph inputs read-only; the block's in-place cache
    write lands on the clone, and the updated slot is returned as cache_out.
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.memblocks = _memblocks(decoder)

    def forward(self, latent: torch.Tensor, *cache_in: torch.Tensor):
        from flashdreams.recipes.taehv.impl import TAEHVCache

        state = {id(mb): cache_in[i].clone() for i, mb in enumerate(self.memblocks)}
        cache = TAEHVCache(dec_state=state)
        rgb = self.decoder(input=latent, autoregressive_index=1, cache=cache)
        cache_out = [cache.dec_state[id(mb)] for mb in self.memblocks]
        return (rgb, *cache_out)


@torch.no_grad()
def _seed_cache(
    decoder: torch.nn.Module,
    memblocks: list[torch.nn.Module],
    latent: torch.Tensor,
) -> list[torch.Tensor]:
    """Populate a realistic cache (correct shapes/values) via one fresh chunk."""
    cache = cast(Any, decoder).initialize_autoregressive_cache()
    decoder(input=latent, autoregressive_index=0, cache=cache)
    return [cache.dec_state[id(mb)].clone() for mb in memblocks]


@torch.no_grad()
def export_streaming_decoder(
    decoder: torch.nn.Module,
    *,
    out_path: str | Path,
    latent_frames: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    device: str | torch.device,
    dtype: torch.dtype,
    version: str,
) -> dict[str, Any]:
    """Export ``decoder`` (a loaded ``TeahvVAEDecoder``) to a cache-as-IO ONNX.

    Writes ``out_path`` (a single self-contained ONNX) and ``out_path`` with a
    ``.spec.json`` suffix describing the input/output tensors the client needs
    to thread the cache. Returns the spec dict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = _StreamingExportWrapper(decoder).eval()
    latent = torch.randn(
        1,
        1,
        latent_frames,
        latent_channels,
        latent_height,
        latent_width,
        device=device,
        dtype=dtype,
    )
    cache_in = _seed_cache(decoder, wrapper.memblocks, latent)
    rgb, *_cache_out = wrapper(latent, *cache_in)

    n = len(cache_in)
    input_names = ["latent"] + [f"cache_in_{i}" for i in range(n)]
    output_names = ["rgb"] + [f"cache_out_{i}" for i in range(n)]

    torch.onnx.export(
        wrapper,
        (latent, *cache_in),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=True,
    )

    # Consolidate weights inline so the browser can load a single file.
    import onnx

    model = onnx.load(str(out_path))
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_name(out_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    precision = {torch.float32: "fp32", torch.float16: "fp16"}.get(dtype, str(dtype))
    spec = {
        "version": version,
        "precision": precision,
        "latent_shape": [latent_frames, latent_channels, latent_height, latent_width],
        "output_shape": list(rgb.shape[2:]),  # [Tout, 3, H, W]
        "input_names": input_names,
        "output_names": output_names,
        "cache": [
            {"name": input_names[1 + i], "shape": list(c.shape)}
            for i, c in enumerate(cache_in)
        ],
    }
    out_path.with_suffix(".spec.json").write_text(json.dumps(spec, indent=2))
    return spec
