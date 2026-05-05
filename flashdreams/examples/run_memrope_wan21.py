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

"""MemRoPE Wan 2.1 T2V demo on top of :class:`WanInferencePipeline`.

Distributed streaming inference entrypoint for MemRoPE variants of the
Self-Forcing Wan 2.1 checkpoint. Rollouts are multi-AR-step by design;
``--total_blocks`` controls how many AR chunks to generate.

Run::

    torchrun --nproc_per_node=1 \\
        examples/run_memrope_wan21.py \\
        --total_blocks 60 \\
        --config_name self_forcing_memrope_s3m2r13
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
from flashdreams.recipes.wan.config.memrope_wan21 import MEMROPE_WAN21_CONFIG_BUILDERS
from flashdreams.recipes.wan.memrope_diffusion import MemRoPEDiffusionModel
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_T2V_PROMPT = (
    "A stylish woman walks down a bustling Tokyo street filled with warm "
    "glowing neon and animated city signage. She wears a black leather "
    "jacket over a long red dress and black boots, carrying a black purse. "
    "She sports sunglasses and red lipstick, walking confidently and "
    "casually. The street is damp and reflective, creating a mirror effect "
    "of the colorful lights. Many pedestrians walk about, adding to the "
    "vibrant atmosphere. Medium shot, dynamic walking perspective."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemRoPE Wan 2.1 T2V demo.")
    parser.add_argument(
        "--config_name",
        type=str,
        default="self_forcing_memrope_s3m2r13",
        choices=sorted(MEMROPE_WAN21_CONFIG_BUILDERS.keys()),
        help="MemRoPE streaming checkpoint preset to load.",
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
            "omitted, the default T2V prompt is used."
        ),
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help=(
            "Disable torch.compile of the DiT network (faster startup, slower steps)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Each distributed rank adds its rank to this value.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Optional suffix for the output mp4/json names.",
    )
    return parser.parse_args()


def _resolve_prompt(prompt_or_txt_path: str | None, default: str) -> str:
    if prompt_or_txt_path is None:
        return default
    if prompt_or_txt_path.endswith(".txt"):
        with open(prompt_or_txt_path, "r") as f:
            return f.readline().strip()
    return prompt_or_txt_path


def _output_suffix(args: argparse.Namespace, world_size: int) -> str:
    suffix = f"{args.config_name}_{args.total_blocks}blocks_seed{args.seed}"
    if args.no_compile:
        suffix += "_nocompile"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    return f"{suffix}_t2v_{world_size}gpus"


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
    print(f"Running MemRoPE Wan 2.1 inference with config: {args.config_name}")

    builder = MEMROPE_WAN21_CONFIG_BUILDERS[args.config_name]
    pipeline = (
        builder(
            cp_size=world_size,
            compile_network=not args.no_compile,
            seed=args.seed + rank,
            i2v=False,
            enable_sync_and_profile=True,
        )
        .setup()
        .to(device=device)
    )
    assert isinstance(pipeline, WanInferencePipeline)
    assert isinstance(pipeline.diffusion_model, MemRoPEDiffusionModel)

    cache = pipeline.initialize_cache(text=[prompt], image=None)

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

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

    generated_video = torch.cat(chunks, dim=1)
    print("end of streaming inference, generated_video.shape:", generated_video.shape)

    if rank == 0:
        canvas = rearrange(generated_video, "1 t c h w -> t h w c")
        canvas = (canvas.float().numpy() + 1.0) / 2.0
        canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        suffix = _output_suffix(args, world_size)
        save_path = f"{REPO_ROOT}/outputs/memrope_wan21_{suffix}.mp4"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=16)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = f"{REPO_ROOT}/outputs/stats_memrope_wan21_{suffix}.json"
            with open(stats_path, "w") as f:
                json.dump(stats_history, f, indent=2)
            print(f"saved per-AR-step stats to {stats_path}")

    del cache
    del pipeline
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
