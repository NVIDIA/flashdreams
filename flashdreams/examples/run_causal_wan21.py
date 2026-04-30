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

"""Causal Wan 2.1 demo on top of :class:`WanInferencePipeline`.

Distributed streaming inference entrypoint for the Self-Forcing /
Causal-Forcing Wan 2.1 checkpoints. The mode is picked by the presence
of ``--image_path``:

- **T2V** — no ``--image_path``. Builds one of the
  :data:`CAUSAL_WAN21_CONFIG_BUILDERS` presets with ``i2v=False``; the
  infra :class:`StreamInferencePipeline` has no encoder slot and the
  recipe pipeline only needs the long-lived :class:`UMT5TextEncoder`.
- **I2V** — ``--image_path`` provided. Flips the builder's ``i2v``
  flag, which wires a streaming :class:`I2VCtrlEncoder` on the infra
  encoder slot. Per AR step the recipe pipeline pads the user's first
  frame along T to one full latent chunk and hands the pixel chunk to
  that encoder; the encoder VAE-encodes it and packages the result as
  an :class:`I2VCtrl` (encoded latent + binary first-frame mask)
  which the infra patchifies and forwards to the transformer as the
  ``input`` argument.

Rollouts are multi-AR-step by design (streaming); ``--total_blocks``
controls how many AR chunks to generate.

Run::

    # T2V
    torchrun --nproc_per_node=N \\
        examples/run_causal_wan21.py \\
        --total_blocks 60 \\
        --config_name self_forcing

    # I2V
    torchrun --nproc_per_node=N \\
        examples/run_causal_wan21.py \\
        --total_blocks 60 \\
        --config_name self_forcing \\
        --image_path path/to/first_frame.png
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

from flashdreams.core.distributed import init as distributed_init
from flashdreams.recipes.wan.config.causal_wan21 import (
    CAUSAL_WAN21_CONFIG_BUILDERS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_T2V_PROMPT = (
    "A stylish woman strolls down a bustling Tokyo street, the warm glow of "
    "neon lights and animated city signs casting vibrant reflections. She "
    "wears a sleek black leather jacket paired with a flowing red dress and "
    "black boots, her black purse slung over her shoulder. Sunglasses "
    "perched on her nose and a bold red lipstick add to her confident, "
    "casual demeanor. The street is damp and reflective, creating a "
    "mirror-like effect that enhances the colorful lights and shadows. "
    "Pedestrians move about, adding to the lively atmosphere. The scene is "
    "captured in a dynamic medium shot with the woman walking slightly to "
    "one side, highlighting her graceful strides."
)
DEFAULT_I2V_PROMPT = DEFAULT_T2V_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal Wan 2.1 streaming demo. Without --image_path runs T2V; "
            "with --image_path runs I2V (mask-injection)."
        )
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="self_forcing",
        choices=sorted(CAUSAL_WAN21_CONFIG_BUILDERS.keys()),
        help="Streaming checkpoint preset to load.",
    )
    parser.add_argument(
        "--total_blocks",
        type=int,
        default=60,
        help="Number of AR chunks to generate.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help=(
            "Optional path to a first-frame image. When provided the "
            "script switches to I2V (mask-injection): the image is "
            "VAE-encoded per AR step and the first temporal frame's "
            "noisy/clean latent is overridden at AR step 0."
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
        "--no_compile",
        action="store_true",
        help=(
            "Disable torch.compile of the DiT network (faster startup, slower steps)."
        ),
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

    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    distributed_init()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    print(
        f"initialized distributed inference with world size {world_size} "
        f"and rank {rank}"
    )
    print(f"Running causal Wan 2.1 inference with config: {args.config_name}")

    builder = CAUSAL_WAN21_CONFIG_BUILDERS[args.config_name]
    pipeline = (
        builder(
            cp_size=world_size,
            compile_network=not args.no_compile,
            seed=42 + rank,
            i2v=is_i2v,
            enable_sync_and_profile=True,
        )
        .setup()
        .to(device=device)
    )

    # Optional first-frame for I2V. Resize to the pixel resolution baked
    # into the transformer config (latent_h/w * decoder spatial compression).
    image: torch.Tensor | None = None
    if is_i2v:
        transformer_cfg = pipeline.diffusion_model.transformer.config
        decoder_sp = pipeline.decoder.spatial_compression_ratio  # ty:ignore[unresolved-attribute]
        pixel_h = transformer_cfg.height * decoder_sp  # ty:ignore[unresolved-attribute]
        pixel_w = transformer_cfg.width * decoder_sp  # ty:ignore[unresolved-attribute]
        first_frame = media.read_image(args.image_path)[..., :3]  # drop alpha
        first_frame = cv2.resize(first_frame, (pixel_w, pixel_h))  # [H, W, 3]
        first_frame_t = (
            torch.from_numpy(first_frame).to(device=device, dtype=transformer_cfg.dtype)  # ty:ignore[unresolved-attribute]
            / 127.5
            - 1.0
        )  # [H, W, 3] in range [-1, 1]
        image = rearrange(first_frame_t, "h w c -> 1 1 c h w")  # [B=1, T=1, C=3, H, W]

    cache = pipeline.initialize_cache(text=[prompt], image=image)  # ty:ignore[unknown-argument]

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---------------------------------------------------------------- rollout
    chunks: list[torch.Tensor] = []
    stats_history: list[dict[str, float]] = []
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_output_frames(i)  # ty:ignore[call-non-callable]
        print(f"autoregressive_index: {i}, num_frames: {num_frames}")
        video_chunk = pipeline.generate(i, cache)
        stats = pipeline.finalize(i, cache)
        if stats is not None:
            stats_history.append({"autoregressive_index": i, **stats})
        chunks.append(video_chunk.cpu())
    generated_video = torch.cat(chunks, dim=1)  # [B, T, C, H, W]
    print("end of streaming inference, generated_video.shape:", generated_video.shape)

    if rank == 0:
        canvas = rearrange(generated_video, "1 t c h w -> t h w c")
        canvas = (canvas.float().numpy() + 1.0) / 2.0
        canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        suffix = "i2v" if is_i2v else "t2v"
        save_path = (
            f"{REPO_ROOT}/outputs/causal_wan21_{args.config_name}"
            f"_{suffix}_{world_size}gpus.mp4"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=16)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = (
                f"{REPO_ROOT}/outputs/stats_causal_wan21_{args.config_name}"
                f"_{suffix}_{world_size}gpus.json"
            )
            with open(stats_path, "w") as f:
                json.dump(stats_history, f, indent=2)
            print(f"saved per-AR-step stats to {stats_path}")

    # Drop captured CUDA graphs / private mempools BEFORE NCCL teardown so
    # they don't hold workspace buffers across the destroy. Otherwise the
    # private mempool can outlive the communicator and rank-0's destroy
    # can hang waiting for already-exited peers.
    del cache
    del pipeline
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Hold every rank here until rank 0 finishes its mp4 encode. Without
    # this barrier rank>0 races to destroy_process_group() and exits while
    # rank 0 is still encoding; rank 0's later destroy then deadlocks
    # trying to talk to peers that no longer exist.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
