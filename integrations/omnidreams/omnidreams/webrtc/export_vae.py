# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the OmniDreams TAEHV VAE decoder to a cache-as-IO ONNX for the
browser token-streaming client.

Writes ``$FLASHDREAMS_CACHE_DIR/omnidreams-vae-decoders/
taehv_decoder.<precision>.onnx`` (+ a ``.spec.json`` sidecar the client uses to
size and thread the temporal cache). Pre-export step, run once per model
version; the WebRTC server serves the artifact and advertises it in the
token-stream session header.

    uv run --package flashdreams-omnidreams python -m omnidreams.webrtc.export_vae --precision fp32

NOTE: ``--latent-frames`` must match the launched pipeline's ``len_t`` (2 for
the default perf config); a later revision should derive it from the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import torch
from omnidreams.webrtc.vae_artifacts import cache_dir

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the TAEHV VAE decoder to a cache-as-IO ONNX."
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=704)
    parser.add_argument(
        "--latent-frames", type=int, default=2, help="latent frames per chunk (len_t)"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    from flashdreams.recipes.taehv import (
        AVAILABLE_TAEHV_CHECKPOINT_PATHS,
        TeahvVAEDecoderConfig,
    )
    from flashdreams.recipes.taehv.export import export_streaming_decoder

    dtype = _DTYPES[args.precision]
    checkpoint = AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"]
    decoder = (
        TeahvVAEDecoderConfig(
            checkpoint_path=checkpoint,
            dtype=dtype,
            use_cuda_graph=False,
            use_compile=False,
        )
        .setup()
        .to(args.device, dtype)
        .eval()
    )

    spatial = int(getattr(decoder, "spatial_compression_ratio", 8))
    latent_channels = int(getattr(decoder.taehv, "latent_channels", 16))
    latent_h = args.video_height // spatial
    latent_w = args.video_width // spatial
    version = Path(urlparse(str(checkpoint)).path or str(checkpoint)).stem

    out_dir = Path(args.out_dir) if args.out_dir else cache_dir()
    out_path = out_dir / f"taehv_decoder.{args.precision}.onnx"

    print(
        f"[export] {version} {args.precision}  "
        f"latent [T={args.latent_frames}, C={latent_channels}, H={latent_h}, W={latent_w}]  "
        f"spatial={spatial}"
    )
    spec = export_streaming_decoder(
        decoder,
        out_path=out_path,
        latent_frames=args.latent_frames,
        latent_channels=latent_channels,
        latent_height=latent_h,
        latent_width=latent_w,
        device=args.device,
        dtype=dtype,
        version=version,
    )
    print(f"[ok] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"[ok] spec {out_path.with_suffix('.spec.json')}: "
        f"output_shape={spec['output_shape']} cache_slots={len(spec['cache'])}"
    )


if __name__ == "__main__":
    main()
