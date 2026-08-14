# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# THROWAWAY 2b feasibility spike -- NOT product code. Delete before any PR.
#
# Exports the OmniDreams serving decoder (TAEHV / LightTAE) for a SINGLE chunk
# with a fresh (zero) cache to ONNX, in BOTH fp32 and fp16, and saves an fp32
# torch "gold" reference so a browser (onnxruntime-web / WebGPU) run can be
# checked for parity + decode-time A/B across the two precisions.
#
# Baked into each ONNX graph (client feeds the raw latent):
#   - latent denormalization (z * std + mean, 16-ch LightTAE constants)
#   - the TAEHV decode (conv / relu / nearest-upsample / temporal-grow / reshape)
#   - the [0,1] -> [-1,1] output remap
#
# Scope/limits: chunk-0 fresh-cache only. Streaming continuity across chunks is
# deferred to productionization (thread the cache frames as ONNX inputs/outputs).
#
# Run on the GPU box in the repo uv env:
#   uv run python spike/2b-vae-parity/export_vae_decoder.py

from __future__ import annotations

import json
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parent
DEVICE = "cuda"

# One chunk latent, matching the live run (704x1280 / 8 spatial, 16 ch,
# 2 latent frames -> 5 output frames after the first-chunk trim). B=1, V=1.
B, V, T, CL, HL, WL = 1, 1, 2, 16, 88, 160

DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def build_decoder(dtype: torch.dtype):
    from flashdreams.recipes.taehv import (
        AVAILABLE_TAEHV_CHECKPOINT_PATHS,
        TeahvVAEDecoderConfig,
    )

    cfg = TeahvVAEDecoderConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        dtype=dtype,
        use_cuda_graph=False,  # off for tracing/export
        use_compile=False,  # off for tracing/export
    )
    return cfg.setup().to(DEVICE, dtype).eval()


class ExportWrapper(torch.nn.Module):
    """latent [B, V, T, Cl, Hl, Wl] -> RGB [B, V, Tout, 3, Hout, Wout] in [-1, 1].

    Fresh (zero) cache created internally so the traced graph is a pure function
    of the latent (the dict-cache side effects drop out of the ONNX graph).
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        cache = self.decoder.initialize_autoregressive_cache()
        return self.decoder(input=z, autoregressive_index=0, cache=cache)


def _consolidate_inline(onnx_path: Path) -> None:
    # onnxruntime-web loads a single self-contained file most easily; the dynamo
    # exporter may split weights into a ".onnx.data" sidecar the browser can't
    # mount. Pull the initializers back inline (TAEHV is tiny, well under 2 GB).
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()


@torch.no_grad()
def export_variant(precision: str, z_fp32: torch.Tensor) -> None:
    dtype = DTYPES[precision]
    decoder = build_decoder(dtype)
    wrapper = ExportWrapper(decoder).eval()
    z = z_fp32.to(DEVICE, dtype)
    rgb = wrapper(z)
    onnx_path = OUT / f"vae_decoder.{precision}.onnx"
    torch.onnx.export(
        wrapper,
        (z,),
        str(onnx_path),
        input_names=["latent"],
        output_names=["rgb"],
        dynamic_axes={
            "latent": {2: "T", 4: "H", 5: "W"},
            "rgb": {1: "Tout", 4: "Hout", 5: "Wout"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    _consolidate_inline(onnx_path)
    size_mb = onnx_path.stat().st_size / 1e6
    print(
        f"[{precision}] latent {tuple(z.shape)} {z.dtype} -> rgb {tuple(rgb.shape)} "
        f"{rgb.dtype}; wrote {onnx_path.name} ({size_mb:.1f} MB)"
    )


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    z = torch.randn(B, V, T, CL, HL, WL, device=DEVICE, dtype=torch.float32)

    # fp32 torch "gold" reference (both ONNX variants are scored against this).
    gold = ExportWrapper(build_decoder(torch.float32)).eval()(z)
    gold.detach().float().cpu().numpy().tofile(OUT / "reference_rgb.f32.bin")
    print(
        f"[ref] fp32 torch gold rgb {tuple(gold.shape)} "
        f"range [{float(gold.min()):.3f}, {float(gold.max()):.3f}]"
    )

    # Latent fixtures. fp16 mirrors the wire (raw_f16 codec), fp32 is the
    # upper-precision baseline. Raw little-endian bytes for the browser to fetch.
    z.detach().float().cpu().numpy().tofile(OUT / "latent.f32.bin")  # float32
    z.detach().half().cpu().numpy().tofile(OUT / "latent.f16.bin")  # float16 bits

    export_variant("fp32", z)
    export_variant("fp16", z)

    meta = {
        "latent_shape": [B, V, T, CL, HL, WL],
        "rgb_shape": list(gold.shape),
        "reference": "reference_rgb.f32.bin",
        "reference_dtype": "float32",
        "variants": {
            "fp32": {
                "onnx": "vae_decoder.fp32.onnx",
                "latent": "latent.f32.bin",
                "latent_dtype": "float32",
            },
            "fp16": {
                "onnx": "vae_decoder.fp16.onnx",
                "latent": "latent.f16.bin",
                "latent_dtype": "float16",
            },
        },
        "notes": (
            "Open index.html?p=fp32 and index.html?p=fp16 and compare warm decode "
            "time + PSNR. fp16 carries the realistic fp16 wire quantization."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ok] wrote fp32+fp16 onnx, latent fixtures, reference, meta.json in {OUT}")


if __name__ == "__main__":
    main()
