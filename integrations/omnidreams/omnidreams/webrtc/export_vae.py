# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the launched OmniDreams VAE decoder to a token-stream ONNX.

Config-driven: ``--pipeline-config-name`` selects the pipeline whose decoder is
exported, so the browser decodes with the same decoder the server generates with
(the decoder, its weights, and ``len_t`` all come from that config). Writes
``$FLASHDREAMS_CACHE_DIR/omnidreams-vae-decoders/taehv_decoder.<precision>.onnx``
plus a ``.spec.json`` sidecar; the WebRTC server serves and advertises them.

    uv run --package flashdreams-omnidreams \\
        python -m omnidreams.webrtc.export_vae --precision fp32
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omnidreams.config import DEFAULT_PIPELINE_CONFIG_NAME
from omnidreams.webrtc.vae_export import (
    export_decoder_from_config,
    resolve_pipeline_config,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export the launched pipeline's VAE decoder to a cache-as-IO ONNX."
    )
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument(
        "--pipeline-config-name",
        "--pipeline_config_name",
        dest="pipeline_config_name",
        default=DEFAULT_PIPELINE_CONFIG_NAME,
        help="Pipeline config whose decoder is exported; match the server launch.",
    )
    parser.add_argument("--video-width", type=int, default=1280)
    parser.add_argument("--video-height", type=int, default=704)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    config = resolve_pipeline_config(args.pipeline_config_name)
    spec = export_decoder_from_config(
        config,
        video_width=args.video_width,
        video_height=args.video_height,
        precision=args.precision,
        device=args.device,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(
        f"[ok] exported {config.name} decoder ({args.precision}): "
        f"version={spec['version']} output_shape={spec['output_shape']} "
        f"cache_slots={len(spec['cache'])}"
    )


if __name__ == "__main__":
    main()
