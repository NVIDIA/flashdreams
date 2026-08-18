# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# THROWAWAY spike (sub-task 1a) -- NOT product code. Delete before any PR.
#
# Exports the TAEHV decoder as a STEADY-STATE, cache-as-IO ONNX model:
# the internal Dict[id(MemBlock), Tensor] temporal cache is exposed as an
# ordered list of explicit tensor inputs (cache_in_*) and outputs (cache_out_*),
# so the browser can thread the cache chunk-to-chunk. A non-empty cache makes
# the decode steady-state (no first-chunk frame trim).
#
# Denorm + decode + [-1,1] remap are baked in (client feeds the raw latent).
#
# Run on the GPU box in the repo uv env:
#   uv run python spike/1a-streaming-cache/export_streaming.py

from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
DEVICE = "cuda"
DTYPE = torch.float32
B, V, T, CL, HL, WL = 1, 1, 2, 16, 88, 160  # one chunk latent (704x1280 / 8 spatial)


def build_decoder():
    from flashdreams.recipes.taehv import (
        AVAILABLE_TAEHV_CHECKPOINT_PATHS,
        TeahvVAEDecoderConfig,
    )

    cfg = TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        dtype=DTYPE,
        use_cuda_graph=False,
        use_compile=False,
    )
    return cfg.setup().to(DEVICE, DTYPE).eval()


def memblocks_of(decoder):
    from flashdreams.recipes.taehv.impl import MemBlock

    return [b for b in decoder.taehv.decoder.blocks if isinstance(b, MemBlock)]


class StreamingWrapper(torch.nn.Module):
    """Steady-state decode with the temporal cache as explicit tensor I/O.

    inputs:  latent [B, V, T, Cl, Hl, Wl], cache_in_0..N-1 [B*V, 1, C_i, H_i, W_i]
    outputs: rgb [B, V, Tout, 3, H, W] in [-1, 1], cache_out_0..N-1
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.memblocks = memblocks_of(decoder)

    def forward(self, latent: torch.Tensor, *cache_in: torch.Tensor):
        from flashdreams.recipes.taehv.impl import TAEHVCache

        # Reconstruct the id-keyed dict from the ordered list. clone() keeps the
        # graph inputs read-only (the block's in-place cache write hits the clone).
        state = {id(mb): cache_in[i].clone() for i, mb in enumerate(self.memblocks)}
        cache = TAEHVCache(dec_state=state)
        rgb = self.decoder(input=latent, autoregressive_index=1, cache=cache)
        cache_out = [cache.dec_state[id(mb)] for mb in self.memblocks]
        return (rgb, *cache_out)


@torch.no_grad()
def seed_cache(decoder, memblocks):
    """Run one fresh chunk to populate a realistic cache (correct shapes/values)."""
    cache = decoder.initialize_autoregressive_cache()
    z = torch.randn(B, V, T, CL, HL, WL, device=DEVICE, dtype=DTYPE)
    _ = decoder(input=z, autoregressive_index=0, cache=cache)
    return [cache.dec_state[id(mb)].clone() for mb in memblocks]


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    decoder = build_decoder()
    wrapper = StreamingWrapper(decoder).eval()
    n_cache = len(wrapper.memblocks)
    print(f"[info] {n_cache} MemBlock cache slots")

    latent = torch.randn(B, V, T, CL, HL, WL, device=DEVICE, dtype=DTYPE)
    cache_in = seed_cache(decoder, wrapper.memblocks)
    for i, c in enumerate(cache_in):
        print(f"       cache[{i}] shape {tuple(c.shape)}")

    args = (latent, *cache_in)
    in_names = ["latent"] + [f"cache_in_{i}" for i in range(n_cache)]
    out_names = ["rgb"] + [f"cache_out_{i}" for i in range(n_cache)]

    onnx_path = OUT / "vae_decoder_streaming.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(onnx_path),
        input_names=in_names,
        output_names=out_names,
        opset_version=18,
        do_constant_folding=True,
    )

    # Consolidate weights inline (single self-contained file for the browser).
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    (OUT / "meta.json").write_text(
        json.dumps(
            {
                "latent_shape": [B, V, T, CL, HL, WL],
                "cache_shapes": [list(c.shape) for c in cache_in],
                "input_names": in_names,
                "output_names": out_names,
                "notes": "steady-state cache-as-IO export; feed a non-empty cache",
            },
            indent=2,
        )
    )
    print(
        f"[ok] wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e6:.1f} MB) + meta.json"
    )


if __name__ == "__main__":
    main()
