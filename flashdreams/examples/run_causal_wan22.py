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

"""Causal Wan 2.2 demo on top of :class:`WanInferencePipeline`.

Distributed streaming inference entrypoint for the FastVideo
CausalWan2.2-A14B distilled checkpoint. **T2V only** for now: Wan 2.2
I2V with this checkpoint uses a first-frame VAE-seed warmup that
doesn't fit the unified pipeline's per-AR-step mask-injection I2V, so
``--image_path`` is deliberately not wired here.

Structurally this is the :mod:`examples.run_causal_wan21` twin; the
only functional diff is the transformer — Wan 2.2's MoE backbone (two
Wan 2.1 14B networks with timestep-based dispatch) replaces the Wan
2.1 14B backbone, and the scheduler runs FastVideo's 8-step
distillation schedule instead of Self-Forcing's 4-step.

Rollouts are multi-AR-step by design (streaming); ``--total_blocks``
controls how many AR chunks to generate.

Run::

    torchrun --nproc_per_node=N \\
        examples/run_causal_wan22.py \\
        --total_blocks 60 \\
        --config_name fastvideo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mediapy as media
import numpy as np
import torch
from einops import rearrange

from flashdreams.core.distributed import init as distributed_init
from flashdreams.recipes.wan.config.causal_wan22 import (
    CAUSAL_WAN22_CONFIG_BUILDERS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal Wan 2.2 streaming T2V demo (FastVideo distilled checkpoint)."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="fastvideo",
        choices=sorted(CAUSAL_WAN22_CONFIG_BUILDERS.keys()),
        help="Streaming checkpoint preset to load.",
    )
    parser.add_argument(
        "--total_blocks",
        type=int,
        default=60,
        help="Number of AR chunks to generate.",
    )
    parser.add_argument(
        "--prompt_or_txt_path",
        type=str,
        default=None,
        help=(
            "Text prompt, or path to a .txt file containing one. When "
            "omitted, a default prompt is used."
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
    prompt = _resolve_prompt(args.prompt_or_txt_path, DEFAULT_T2V_PROMPT)

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
    print(f"Running causal Wan 2.2 inference with config: {args.config_name}")

    builder = CAUSAL_WAN22_CONFIG_BUILDERS[args.config_name]
    pipeline = (
        builder(
            compile_network=not args.no_compile,
            seed=42 + rank,
            enable_sync_and_profile=True,
        )
        .setup()
        .to(device=device)
    )

    assert isinstance(pipeline, WanInferencePipeline)
    cache = pipeline.initialize_cache(
        text=[prompt],
        image=None,
        height=DEFAULT_VIDEO_HEIGHT // WAN_VAE_SPATIAL_COMPRESSION,
        width=DEFAULT_VIDEO_WIDTH // WAN_VAE_SPATIAL_COMPRESSION,
    )

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---------------------------------------------------------------- rollout
    chunks: list[torch.Tensor] = []
    stats_history: list[dict[str, float]] = []
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_output_frames(i)
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
        save_path = (
            f"{REPO_ROOT}/outputs/causal_wan22_{args.config_name}"
            f"_t2v_{world_size}gpus.mp4"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=16)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = (
                f"{REPO_ROOT}/outputs/stats_causal_wan22_{args.config_name}"
                f"_t2v_{world_size}gpus.json"
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
