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
from flashdreams.infra.diffusion.noise import (
    load_initial_noise_rollout,
    select_initial_noise_chunk,
    select_temporal_initial_noise_chunk,
    stack_initial_noise_chunks,
)
from flashdreams.recipes.wan.config.causal_wan21 import (
    CAUSAL_WAN21_CONFIG_BUILDERS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    WAN_VAE_SPATIAL_COMPRESSION,
)
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
        "--attention_backend",
        choices=("cudnn", "flash", "sdpa_flash"),
        default=None,
        help="Override Wan attention backend. Use flash for official QVG alignment probes.",
    )
    parser.add_argument(
        "--store_prerope_keys",
        action="store_true",
        help="Store BF16 self-attention keys pre-RoPE for official QVG alignment probes.",
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
    parser.add_argument(
        "--initial_noise_path",
        type=Path,
        default=None,
        help=(
            "Optional torch Tensor path for explicit initial noise. Accepts "
            "[num_chunks, B, L, D] patchified noise or [B, total_T, C, H, W]."
        ),
    )
    parser.add_argument(
        "--save_initial_noise_path",
        type=Path,
        default=None,
        help=(
            "Optional path to save the exact initial noise used by this run "
            "in [B, total_T, C, H, W] layout."
        ),
    )
    parser.add_argument(
        "--text_embeddings_path",
        type=Path,
        default=None,
        help=(
            "Optional torch Tensor or {'prompt_embeds': Tensor} path to use "
            "instead of FlashDreams text encoding."
        ),
    )
    parser.add_argument(
        "--renoise_noise_folder",
        type=Path,
        default=None,
        help=(
            "Optional folder containing official scheduler re-noise tensors "
            "named <idx>_<chunk>_<step>.pt."
        ),
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
    if args.attention_backend is not None:
        transformer_config.network.attention_backend = args.attention_backend
    if args.store_prerope_keys:
        transformer_config.store_prerope_keys = True

    return {
        "backend": kv_config.backend,
        "backend_config": dict(kv_config.backend_config),
        "schedule": dict(kv_config.schedule),
        "protected_recent_chunks": kv_config.protected_recent_chunks,
        "window_size_t": transformer_config.window_size_t,
        "attention_backend": transformer_config.network.attention_backend,
        "store_prerope_keys": transformer_config.store_prerope_keys,
    }


def _wan_unpatchified_noise_shape(pipeline: WanInferencePipeline) -> tuple[int, ...]:
    transformer = pipeline.diffusion_model.transformer
    transformer_config = transformer.config
    channels = transformer_config.network.in_dim
    if getattr(transformer_config, "concat_image_mask_to_latent", False):
        channels -= 4 + 16
    height = getattr(transformer, "_output_height", None)
    width = getattr(transformer, "_output_width", None)
    if height is None:
        height = DEFAULT_VIDEO_HEIGHT // WAN_VAE_SPATIAL_COMPRESSION
    if width is None:
        width = DEFAULT_VIDEO_WIDTH // WAN_VAE_SPATIAL_COMPRESSION
    return (
        *transformer_config.batch_shape,
        transformer_config.len_t,
        channels,
        height,
        width,
    )


def _prepare_initial_noise(
    *,
    pipeline: WanInferencePipeline,
    device: torch.device,
    rollout: torch.Tensor | None,
    autoregressive_index: int,
    should_draw: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return generation noise plus unpatchified noise for saving."""
    noise_in_unpatchified_shape = (
        pipeline.diffusion_model.config._noise_in_unpatchified_shape
    )
    if rollout is not None:
        latent_shape = pipeline.diffusion_model.latent_shape
        if rollout.ndim == len(latent_shape) + 1 or (
            rollout.ndim == len(latent_shape) and len(latent_shape) == 5
        ):
            initial_noise = select_initial_noise_chunk(
                rollout,
                autoregressive_index=autoregressive_index,
                latent_shape=latent_shape,
            )
            initial_noise = initial_noise.to(
                device=device,
                dtype=pipeline.diffusion_model.dtype,
            )
            if len(latent_shape) == 5:
                save_noise = initial_noise
            else:
                save_noise = pipeline.diffusion_model.transformer.unpatchify_and_maybe_gather_cp(
                    initial_noise
                )
            if noise_in_unpatchified_shape:
                initial_noise = save_noise
            return (
                initial_noise.to(device=device, dtype=pipeline.diffusion_model.dtype),
                save_noise,
            )

        unpatchified = select_temporal_initial_noise_chunk(
            rollout,
            autoregressive_index=autoregressive_index,
            chunk_shape=_wan_unpatchified_noise_shape(pipeline),
        )
        unpatchified = unpatchified.to(
            device=device,
            dtype=pipeline.diffusion_model.dtype,
        )
        if noise_in_unpatchified_shape:
            return unpatchified, unpatchified
        initial_noise = pipeline.diffusion_model.transformer.patchify_and_maybe_split_cp(
            unpatchified
        )
        return initial_noise, unpatchified

    if not should_draw:
        return None, None

    if noise_in_unpatchified_shape:
        initial_noise = torch.randn(
            _wan_unpatchified_noise_shape(pipeline),
            device=device,
            dtype=pipeline.diffusion_model.dtype,
            generator=pipeline.diffusion_model.rng,
        )
        save_noise = initial_noise
    else:
        initial_noise = torch.randn(
            pipeline.diffusion_model.latent_shape,
            device=device,
            dtype=pipeline.diffusion_model.dtype,
            generator=pipeline.diffusion_model.rng,
        )
        save_noise = pipeline.diffusion_model.transformer.unpatchify_and_maybe_gather_cp(
            initial_noise
        )
    return initial_noise, save_noise


def _load_text_embeddings(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload["prompt_embeds"]
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"text embeddings must be a Tensor, got {type(payload)}")
    return payload


def _load_renoise_replay(
    folder: Path | None,
    prompt_index: int = 0,
) -> list[torch.Tensor] | None:
    if folder is None:
        return None
    paths = sorted(folder.glob(f"{prompt_index}_*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No scheduler re-noise replay tensors found in {folder}"
        )
    return [torch.load(path, map_location="cpu") for path in paths]


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

    builder = CAUSAL_WAN21_CONFIG_BUILDERS[args.config_name]
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
    pipeline.diffusion_model.transformer.set_scheduler_renoise_replay(
        _load_renoise_replay(args.renoise_noise_folder)
    )
    assert isinstance(pipeline, WanInferencePipeline)

    initial_noise_rollout = (
        load_initial_noise_rollout(args.initial_noise_path)
        if args.initial_noise_path is not None
        else None
    )
    latent_h = DEFAULT_VIDEO_HEIGHT // WAN_VAE_SPATIAL_COMPRESSION
    latent_w = DEFAULT_VIDEO_WIDTH // WAN_VAE_SPATIAL_COMPRESSION
    cache = pipeline.initialize_cache(
        text=[prompt],
        image=None,
        height=latent_h,
        width=latent_w,
        text_embeddings=_load_text_embeddings(args.text_embeddings_path),
    )

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    chunks: list[torch.Tensor] = []
    stats_history: list[dict[str, float | int]] = []
    initial_noise_chunks: list[torch.Tensor] = []
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_output_frames(i)
        print(f"autoregressive_index: {i}, num_frames: {num_frames}")
        initial_noise, save_noise = _prepare_initial_noise(
            pipeline=pipeline,
            device=device,
            rollout=initial_noise_rollout,
            autoregressive_index=i,
            should_draw=args.save_initial_noise_path is not None,
        )

        if save_noise is not None and args.save_initial_noise_path is not None:
            initial_noise_chunks.append(save_noise.detach().cpu())

        video_chunk = pipeline.generate(i, cache, initial_noise=initial_noise)
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

        if args.save_initial_noise_path is not None:
            args.save_initial_noise_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                stack_initial_noise_chunks(initial_noise_chunks),
                args.save_initial_noise_path,
            )
            print(f"saved initial noise to {args.save_initial_noise_path}")

        metadata_path = f"{REPO_ROOT}/outputs/metadata_qvg_wan21_{suffix}.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "config_name": args.config_name,
                    "total_blocks": args.total_blocks,
                    "seed": args.seed,
                    "no_compile": args.no_compile,
                    "initial_noise_path": (
                        str(args.initial_noise_path)
                        if args.initial_noise_path is not None
                        else None
                    ),
                    "save_initial_noise_path": (
                        str(args.save_initial_noise_path)
                        if args.save_initial_noise_path is not None
                        else None
                    ),
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
