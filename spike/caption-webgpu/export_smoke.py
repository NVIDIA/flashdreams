# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# THROWAWAY spike (caption WebGPU de-risk) -- NOT product code. Delete before any PR.
#
# Export a RANDOM-WEIGHTS caption-bank classifier to ONNX and (via smoke.html)
# confirm the graph runs on onnxruntime-web / WebGPU in the browser -- BEFORE
# investing in data + training. The output is meaningless (random weights); this
# only proves the architecture + export + WebGPU op coverage.
#
#   uv run python spike/caption-webgpu/export_smoke.py

from __future__ import annotations

import json
from pathlib import Path

import torch

from model import CaptionBankClassifier

HERE = Path(__file__).resolve().parent

# Matches the omnidreams sv config latent + a representative caption window.
WINDOW_CHUNKS = 4
FRAMES_PER_CHUNK = 2
LATENT = (16, 88, 160)  # Cl, Hl, Wl

# Placeholder caption bank (real captions come from the trained model's spec).
CAPTIONS = [
    "Driving straight on an open road",
    "Approaching an intersection",
    "Turning left at the junction",
    "Turning right at the junction",
    "Slowing to a stop",
    "Accelerating down the road",
    "Following a vehicle ahead",
    "Pedestrian crossing ahead",
    "Merging into traffic",
    "Cruising through an urban street",
    "Passing parked cars",
    "Waiting at a red light",
    "Clear road ahead",
    "Curving along the road",
    "Heavy traffic ahead",
    "Navigating a roundabout",
    "Vehicle stopped ahead",
    "Changing lanes",
    "Driving past buildings",
    "Open highway ahead",
    "Reversing slowly",
    "Sharp turn ahead",
    "Approaching a crosswalk",
    "Cyclist on the right",
    "Road curves to the left",
    "Road curves to the right",
    "Stationary at a stop sign",
    "Traffic moving slowly",
    "Wide boulevard ahead",
    "Entering a tunnel",
    "Bright daylight scene",
    "Overcast driving conditions",
]


def main() -> None:
    twin = WINDOW_CHUNKS * FRAMES_PER_CHUNK
    model = CaptionBankClassifier(
        latent_channels=LATENT[0], num_captions=len(CAPTIONS)
    ).eval()
    dummy = torch.randn(1, twin, *LATENT)
    with torch.no_grad():
        out = model(dummy)
    print(f"[forward] in {tuple(dummy.shape)} -> out {tuple(out.shape)}")

    onnx_path = HERE / "caption_model.fp32.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["latent"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
    )

    # Consolidate weights inline so the browser loads a single file.
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    spec = {
        "kind": "latent-caption-onnx",
        "version": "caption-cnn-untrained",
        "precision": "fp32",
        "input_window_chunks": WINDOW_CHUNKS,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "latent_shape": list(LATENT),
        "input_shape": [1, twin, *LATENT],
        "caption_bank": CAPTIONS,
    }
    (HERE / "caption_model.fp32.spec.json").write_text(json.dumps(spec, indent=2))
    print(
        f"[ok] wrote {onnx_path.name} ({onnx_path.stat().st_size / 1e3:.0f} KB) "
        f"+ spec (input {spec['input_shape']} -> logits [1, {len(CAPTIONS)}])"
    )


if __name__ == "__main__":
    main()
