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

"""Wan 2.1 demo on top of :class:`Wan21Pipeline`.

One entry point for both flavors of the non-streaming Wan 2.1 demo.
The mode is picked by the presence of ``--image_path``:

- **T2V (1.3B)** — no ``--image_path``. Builds
  :func:`build_wan21_t2v_1pt3b_480p`. The infra
  :class:`StreamInferencePipeline` has no encoder slot and the project
  pipeline only needs the long-lived :class:`UMT5TextEncoder`.
- **I2V (14B 480P)** — ``--image_path`` provided. Builds
  :func:`build_wan21_i2v_14b_480p`. Adds:

  * a long-lived :class:`CLIPImageEncoder` for the 14B I2V
    network's image cross-attention,
  * a streaming :class:`I2VCtrlEncoder` on the infra
    encoder slot (defined in :mod:`flashdreams.recipes.wan.autoencoder.i2v`).
    Per AR step the project pipeline pads the user's first frame along T
    to one full latent chunk and hands the pixel chunk to that
    encoder; the encoder VAE-encodes it and packages the result as
    an :class:`ImageCtrl` (encoded latent + binary
    first-frame mask) which the infra patchifies and forwards
    to the transformer as the ``input`` argument.

Both flavors run a single AR step (one chunk covers the whole 81-frame
video) and call :meth:`finalize` for API symmetry.

Run::

    # T2V (1.3B)
    python -m recipes.wan21.run

    # I2V (14B 480P)
    python -m recipes.wan21.run --image_path path/to/first_frame.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import mediapy as media
import numpy as np
import torch
from einops import rearrange

from flashdreams.recipes.wan.config.wan21 import (
    build_wan21_i2v_14b_480p,
    build_wan21_t2v_1pt3b_480p,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_T2V_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)
DEFAULT_I2V_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow "
    "of neon lights and animated city signs casting vibrant reflections."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wan 2.1 demo. Without --image_path runs T2V (1.3B); "
            "with --image_path runs I2V (14B 480P)."
        )
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help=(
            "Optional path to a first-frame image. When provided the "
            "script switches to the I2V (14B 480P) preset; otherwise "
            "T2V (1.3B) is used."
        ),
    )
    parser.add_argument(
        "--prompt_or_txt_path",
        type=str,
        default=None,
        help=(
            "Text prompt, or path to a .txt file containing one. When "
            "omitted, a mode-specific default prompt is used."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=832,
        help="Output video height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Output video width.",
    )
    return parser.parse_args()


def _resolve_prompt(prompt_or_txt_path: str | None, default: str) -> str:
    if prompt_or_txt_path is None:
        return default
    if prompt_or_txt_path.endswith(".txt"):
        with open(prompt_or_txt_path, "r") as f:
            return f.readline().strip()
    return prompt_or_txt_path


def main() -> None:
    args = parse_args()
    is_i2v = args.image_path is not None
    prompt = _resolve_prompt(
        args.prompt_or_txt_path,
        DEFAULT_I2V_PROMPT if is_i2v else DEFAULT_T2V_PROMPT,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if is_i2v:
        pipeline = (
            build_wan21_i2v_14b_480p(
                video_height=args.height,
                video_width=args.width,
                enable_sync_and_profile=True,
            )
            .setup()
            .to(device=device)
        )

        first_frame = media.read_image(args.image_path)[..., :3]  # drop alpha
        first_frame = cv2.resize(first_frame, (args.width, args.height))  # [H, W, 3]
        first_frame = (
            torch.from_numpy(first_frame).to(device=device, dtype=dtype) / 127.5 - 1.0
        )  # [H, W, 3] in range [-1, 1]
        image = rearrange(first_frame, "h w c -> 1 c h w")  # [T, 3, H, W]
        assert isinstance(pipeline, WanInferencePipeline)
        cache = pipeline.initialize_cache(text=[prompt], image=image)
    else:
        pipeline = (
            build_wan21_t2v_1pt3b_480p(
                video_height=args.height,
                video_width=args.width,
                enable_sync_and_profile=True,
            )
            .setup()
            .to(device=device)
        )

        assert isinstance(pipeline, WanInferencePipeline)
        cache = pipeline.initialize_cache(text=[prompt])

    generated_video = pipeline.generate(autoregressive_index=0, cache=cache)
    # Single-AR-step rollouts don't need finalize; run it for API
    # symmetry (and to log this step's per-stage profiling). Call it
    # before .cpu() so the host sync from the device->host copy isn't
    # attributed to the "finalize" stage in the per-AR-step timing.
    stats = pipeline.finalize(autoregressive_index=0, cache=cache)
    generated_video = generated_video.cpu()
    print("Generated video shape:", generated_video.shape)

    # Save the generated video
    canvas = rearrange(generated_video, "t c h w -> t h w c")
    canvas = (canvas.float().numpy() + 1.0) / 2.0
    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
    suffix = "i2v_14b_480p" if is_i2v else "t2v_1.3b"
    save_path = f"{REPO_ROOT}/outputs/wan21_{suffix}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, canvas, fps=16)
    print(f"saved generated video to {save_path}")

    if stats is not None:
        stats_path = f"{REPO_ROOT}/outputs/stats_wan21_{suffix}.json"
        with open(stats_path, "w") as f:
            json.dump([{"autoregressive_index": 0, **stats}], f, indent=2)
        print(f"saved per-AR-step stats to {stats_path}")


if __name__ == "__main__":
    main()
