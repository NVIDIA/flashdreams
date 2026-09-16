# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 5: export a trained caption CNN to ONNX + spec for the browser client.

Loads a checkpoint from :mod:`omnidreams.caption.train`, exports the model to
ONNX (single inlined file, opset 18) plus a ``.spec.json`` describing the class
bank and input window, matching what
:func:`omnidreams.webrtc.caption_artifacts.build_descriptor` reads. With
``--to-cache`` it writes straight into the served cache dir so the next
token-stream session advertises the model to the client::

    uv run --project integrations/omnidreams \
      python -m omnidreams.caption.export \
        --checkpoint DATA/checkpoints/caption_cnn_best.pt --to-cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from omnidreams.caption.model import (
    CAPTION_BANK,
    CAPTION_CLASSES,
    CaptionBankClassifier,
)

#: Must match the capture window (K chunks x T frames) and latent geometry.
WINDOW_CHUNKS = 5
FRAMES_PER_CHUNK = 2
LATENT = (16, 88, 160)  # Cl, Hl, Wl


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the caption CNN to ONNX.")
    ap.add_argument("--checkpoint", required=True, help="Trained .pt checkpoint.")
    ap.add_argument(
        "--out", default=None, help="Output dir (default: the served cache dir)."
    )
    ap.add_argument(
        "--to-cache",
        action="store_true",
        help="Write into the served caption-model cache dir (auto-served).",
    )
    ap.add_argument("--version", default="caption-cnn-v1")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    classes = ckpt.get("classes", list(CAPTION_CLASSES))
    # Display captions for exactly the trained classes, in logits order.
    bank = [CAPTION_BANK[CAPTION_CLASSES.index(c)] for c in classes]
    model = CaptionBankClassifier(
        num_frames=ckpt.get("num_frames", WINDOW_CHUNKS * FRAMES_PER_CHUNK),
        num_classes=len(classes),
        hidden=ckpt.get("hidden", 256),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    twin = WINDOW_CHUNKS * FRAMES_PER_CHUNK
    dummy = torch.randn(1, twin, *LATENT)
    if args.precision == "fp16":
        model, dummy = model.half(), dummy.half()

    if args.to_cache or args.out is None:
        from omnidreams.webrtc import caption_artifacts

        out_dir = caption_artifacts.cache_dir()
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / f"caption_model.{args.precision}.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["latent"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,  # legacy exporter: matches the WebGPU-validated op set,
        # and avoids the dynamo path's onnxscript dependency.
    )
    # Consolidate weights inline so the browser fetches a single file.
    import onnx

    onnx.save_model(
        onnx.load(str(onnx_path)), str(onnx_path), save_as_external_data=False
    )
    sidecar = onnx_path.with_name(onnx_path.name + ".data")
    if sidecar.exists():
        sidecar.unlink()

    spec = {
        "kind": "latent-caption-onnx",
        "version": args.version,
        "precision": args.precision,
        "input_window_chunks": WINDOW_CHUNKS,
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "latent_shape": list(LATENT),
        "input_shape": [1, twin, *LATENT],
        "labels": list(classes),
        "caption_bank": bank,
    }
    spec_path = out_dir / f"caption_model.{args.precision}.spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    kb = onnx_path.stat().st_size / 1e3
    print(
        f"[export] {onnx_path} ({kb:.0f} KB) + spec (v={args.version}, "
        f"input {spec['input_shape']} -> logits [1, {len(classes)}])",
        flush=True,
    )


if __name__ == "__main__":
    main()
