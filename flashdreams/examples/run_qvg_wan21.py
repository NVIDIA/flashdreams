# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""QVG Wan 2.1 T2V demo on top of :class:`WanInferencePipeline`.

This runner is intentionally T2V-only for the first QVG integration. It uses
the Self-Forcing Wan 2.1 checkpoint and opt-in QVG KV compression configs.
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
from flashdreams.recipes.wan.config.causal_wan21 import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.wan.config.qvg_wan21 import QVG_WAN21_CONFIG_BUILDERS
from flashdreams.recipes.wan.pipeline import WanInferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
QVG_CONFIG_NAMES = ("self_forcing_qvg_int2", "self_forcing_qvg_int4")

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
    parser = argparse.ArgumentParser(description="QVG Wan 2.1 T2V demo.")
    parser.add_argument(
        "--config_name",
        type=str,
        default="self_forcing_qvg_int2",
        choices=QVG_CONFIG_NAMES,
        help="QVG KV-compression preset to load.",
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
        help="Text prompt, or path to a .txt file containing one.",
    )
    parser.add_argument(
        "--prompt_index",
        type=int,
        default=0,
        help="Zero-based prompt index when --prompt_or_txt_path points to a text file.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile of the DiT network.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--window_size_t",
        type=int,
        default=None,
        help="Override transformer self-attention window in latent frames.",
    )
    parser.add_argument(
        "--output_tag",
        type=str,
        default=None,
        help="Optional suffix for output mp4/json names.",
    )
    parser.add_argument(
        "--qvg_kmeans_max_iters",
        type=int,
        default=None,
        help="Override QVG k-means iterations for benchmark/tuning runs.",
    )
    parser.add_argument(
        "--qvg_quant_block_size",
        type=int,
        default=None,
        help="Override QVG residual quantization block size.",
    )
    parser.add_argument(
        "--qvg_cache_num_k_centroids",
        type=int,
        default=None,
        help="Override number of K centroids.",
    )
    parser.add_argument(
        "--qvg_cache_num_v_centroids",
        type=int,
        default=None,
        help="Override number of V centroids.",
    )
    parser.add_argument(
        "--qvg_cache_k_num_bits",
        type=int,
        default=None,
        choices=(2, 4),
        help="Override K residual bit width for mixed-bit QVG tuning.",
    )
    parser.add_argument(
        "--qvg_cache_v_num_bits",
        type=int,
        default=None,
        choices=(2, 4),
        help="Override V residual bit width for mixed-bit QVG tuning.",
    )
    parser.add_argument(
        "--qvg_num_prq_stages",
        type=int,
        default=None,
        help="Override number of PRQ stages.",
    )
    parser.add_argument(
        "--qvg_scale_dtype",
        type=str,
        default=None,
        choices=("bfloat16", "float16", "float32", "float8_e4m3fn"),
        help="Override QVG scale storage dtype.",
    )
    parser.add_argument(
        "--qvg_kmeans_init",
        type=str,
        default=None,
        choices=("linspace", "random"),
        help="Override QVG k-means centroid initialization.",
    )
    parser.add_argument(
        "--qvg_kmeans_seed",
        type=int,
        default=None,
        help="Local seed for random QVG k-means initialization.",
    )
    parser.add_argument(
        "--qvg_kernel_impl",
        type=str,
        default="official_triton",
        choices=("native", "official_triton"),
        help="Override QVG implementation: native PyTorch or official QVG Triton.",
    )
    parser.add_argument(
        "--qvg_compress_every_n_chunks",
        type=int,
        default=None,
        help="Override QVG compression cadence.",
    )
    parser.add_argument(
        "--qvg_protected_recent_chunks",
        type=int,
        default=None,
        help="Override number of recent chunks left dense.",
    )
    return parser.parse_args()


def _resolve_prompt(prompt_or_txt_path: str | None, prompt_index: int) -> str:
    if prompt_or_txt_path is None:
        return DEFAULT_T2V_PROMPT
    if prompt_or_txt_path.endswith(".txt"):
        with open(prompt_or_txt_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise IndexError(
                f"prompt_index {prompt_index} is outside {len(prompts)} prompts"
            )
        return prompts[prompt_index]
    return prompt_or_txt_path


def _output_suffix(args: argparse.Namespace, world_size: int) -> str:
    suffix = f"{args.config_name}_{args.total_blocks}blocks_seed{args.seed}"
    if args.no_compile:
        suffix += "_nocompile"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    return f"{suffix}_t2v_{world_size}gpus"


def _apply_qvg_overrides(
    pipeline_config: object,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Apply runner-only QVG tuning overrides before setup."""
    transformer_config = pipeline_config.diffusion_model.transformer
    kv_config = transformer_config.kv_compression
    assert kv_config is not None, "QVG runner expects kv_compression config"

    backend_overrides = {
        "kmeans_max_iters": args.qvg_kmeans_max_iters,
        "quant_block_size": args.qvg_quant_block_size,
        "cache_num_k_centroids": args.qvg_cache_num_k_centroids,
        "cache_num_v_centroids": args.qvg_cache_num_v_centroids,
        "cache_k_num_bits": args.qvg_cache_k_num_bits,
        "cache_v_num_bits": args.qvg_cache_v_num_bits,
        "num_prq_stages": args.qvg_num_prq_stages,
        "scale_dtype": args.qvg_scale_dtype,
        "kmeans_init": args.qvg_kmeans_init,
        "kmeans_seed": args.qvg_kmeans_seed,
        "kernel_impl": args.qvg_kernel_impl,
    }
    backend_config = dict(kv_config.backend_config)
    for key, value in backend_overrides.items():
        if value is not None:
            backend_config[key] = value
    kv_config.backend_config = backend_config

    schedule = dict(kv_config.schedule)
    if args.qvg_compress_every_n_chunks is not None:
        schedule["compress_every_n_chunks"] = args.qvg_compress_every_n_chunks
    kv_config.schedule = schedule

    if args.qvg_protected_recent_chunks is not None:
        kv_config.protected_recent_chunks = args.qvg_protected_recent_chunks

    if args.window_size_t is not None:
        transformer_config.window_size_t = args.window_size_t

    return {
        "backend": kv_config.backend,
        "backend_config": dict(kv_config.backend_config),
        "schedule": dict(kv_config.schedule),
        "protected_recent_chunks": kv_config.protected_recent_chunks,
        "window_size_t": transformer_config.window_size_t,
    }


def main() -> None:
    args = parse_args()
    prompt = _resolve_prompt(args.prompt_or_txt_path, args.prompt_index)

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
    print(f"Running QVG Wan 2.1 inference with config: {args.config_name}")

    builder = QVG_WAN21_CONFIG_BUILDERS[args.config_name]
    pipeline_config = builder(
        cp_size=world_size,
        compile_network=not args.no_compile,
        seed=args.seed + rank,
        i2v=False,
        enable_sync_and_profile=True,
    )
    qvg_run_config = _apply_qvg_overrides(pipeline_config, args)
    print("QVG runtime config:", json.dumps(qvg_run_config, sort_keys=True))

    pipeline = pipeline_config.setup().to(device=device)
    assert isinstance(pipeline, WanInferencePipeline)

    latent_h = DEFAULT_VIDEO_HEIGHT // WAN_VAE_SPATIAL_COMPRESSION
    latent_w = DEFAULT_VIDEO_WIDTH // WAN_VAE_SPATIAL_COMPRESSION
    cache = pipeline.initialize_cache(
        text=[prompt],
        image=None,
        height=latent_h,
        width=latent_w,
    )

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    chunks: list[torch.Tensor] = []
    stats_history: list[dict[str, float | int]] = []
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
        save_path = f"{REPO_ROOT}/outputs/qvg_wan21_{suffix}.mp4"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=16)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = f"{REPO_ROOT}/outputs/stats_qvg_wan21_{suffix}.json"
            with open(stats_path, "w") as f:
                json.dump(stats_history, f, indent=2)
            print(f"saved per-AR-step stats to {stats_path}")

        metadata_path = f"{REPO_ROOT}/outputs/metadata_qvg_wan21_{suffix}.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "config_name": args.config_name,
                    "total_blocks": args.total_blocks,
                    "seed": args.seed,
                    "no_compile": args.no_compile,
                    "qvg": qvg_run_config,
                },
                f,
                indent=2,
            )
        print(f"saved run metadata to {metadata_path}")

    del cache
    del pipeline
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
