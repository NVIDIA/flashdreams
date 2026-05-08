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

"""Alpadreams demo on top of :class:`AlpadreamsPipeline`.

Distributed streaming inference entrypoint for the alpadreams
driving-scene video generation recipe (Cosmos DiT + HDMap + I2V mask).
Picks one of :data:`ALPADREAMS_CONFIG_BUILDERS` based on
``--n_cameras``:

- ``--n_cameras 1`` — single front-facing camera, defaults to
  ``sv_2steps_chunk2_loc6_lightvae_lighttae``.
- ``--n_cameras 4`` — four surrounding cameras, defaults to
  ``mv_2steps_chunk4_loc8_pshuffle_lighttae``.
- ``--overwrite_config_name`` can select other registered configs, including
  the single-block bidirectional Alpadreams recipe. For configs that expose
  ``num_chunks``, ``--num_chunks`` forwards a user-chosen latent length.

Autoregressive configs consume one per-chunk HDMap pixel tensor (pre-extracted
from the example MP4s) at each AR step. Single-block bidirectional configs consume 
one full-block HDMap tensor. At step/block 0, the first-frame pixel tensor seeds 
the I2V mask injection inside :class:`CosmosTransformer`.

Run::

    # Single front-facing camera
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 1 \\
        --total_blocks 60

    # 4 surrounding cameras
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 4 \\
        --total_blocks 60

    # Use precomputed embeddings (skip Cosmos-Reason1 + Wan-VAE-encoder
    # load entirely; saves ~14 GB of VRAM). Step 1: dump embeddings to
    # disk by running this same script with --save_embeddings_path (no
    # distributed init, no AR rollout); step 2: run inference pointing
    # at that file via --embeddings_path.
    python examples/run_alpadreams.py \\
        --n_cameras 1 \\
        --save_embeddings_path outputs/alpadreams_sv_embeddings.pt
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 1 \\
        --total_blocks 60 \\
        --embeddings_path outputs/alpadreams_sv_embeddings.pt

    # Same idea but in-process: load the encoders, compute embeddings,
    # free the encoders, then load the AR pipeline. No on-disk artifact,
    # lower peak VRAM than the default path:
    torchrun --nproc_per_node=N \\
        examples/run_alpadreams.py \\
        --n_cameras 1 \\
        --total_blocks 60 \\
        --offload_text_encoder
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
from flashdreams.core.io.s3_sync import sync_s3_dir_to_local
from flashdreams.recipes.alpadreams.config import (
    ALPADREAMS_CONFIG_BUILDERS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    WAN_VAE_SPATIAL_COMPRESSION,
)
from flashdreams.recipes.alpadreams.constants import NEGATIVE_PROMPT
from flashdreams.recipes.alpadreams.pipeline import (
    AlpadreamsPipeline,
    AlpadreamsPipelineConfig,
)
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformerConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoder, TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.vae import WanVAEDecoder, WanVAEDecoderConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR_S3 = "s3://flashdreams/assets/example_data/alpadreams"
EXAMPLE_DATA_DIR_LOCAL = str(REPO_ROOT / "assets/example_data/alpadreams")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _needs_negative_text(pipeline_config: AlpadreamsPipelineConfig) -> bool:
    transformer_config = pipeline_config.diffusion_model.transformer
    assert isinstance(transformer_config, CosmosTransformerConfig)
    return transformer_config.requires_negative_text_embeddings


def _config_uses_num_chunks(config_name: str) -> bool:
    return config_name in [
        "sv_35steps_chunk48_loc48_cosmos2_2B_res720p_30fps_hdmap_vae_mads1m"
    ]


def _num_chunks_args(
    config_name: str, requested_num_chunks: int | None
) -> dict[str, int]:
    if requested_num_chunks is None:
        return {}
    if not _config_uses_num_chunks(config_name):
        raise ValueError("--num_chunks is only supported by the bidirectional config.")
    return {"num_chunks": requested_num_chunks}


def _split_user_paths(
    value: str | None, *, n_cameras: int, name: str
) -> list[str] | None:
    if value is None:
        return None
    paths = [path.strip() for path in value.split(",") if path.strip()]
    if len(paths) != n_cameras:
        raise ValueError(
            f"{name} expects {n_cameras} path(s), got {len(paths)}. "
            "Use comma-separated paths for multi-view runs."
        )
    return paths


def _apply_data_overrides(
    data: list[dict],
    *,
    hdmap_video_path: str | None,
    first_frame_path: str | None,
) -> None:
    hdmap_paths = _split_user_paths(
        hdmap_video_path, n_cameras=len(data), name="--hdmap_video_path"
    )
    first_frame_paths = _split_user_paths(
        first_frame_path, n_cameras=len(data), name="--first_frame_path"
    )
    for i, entry in enumerate(data):
        if hdmap_paths is not None:
            entry["hdmap_video_path"] = hdmap_paths[i]
        if first_frame_paths is not None:
            entry["first_frame_path"] = first_frame_paths[i]


def _read_first_frame(path: str) -> np.ndarray:
    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
        return media.read_image(path)[..., :3]
    video = media.read_video(path)
    assert video.shape[0] > 0, f"Video has no frames: {path}"
    return video[0, ..., :3]


def _build_data(n_cameras: int) -> tuple[list[str], list[dict]]:
    """Hardcoded example data per camera count, matching legacy alpadreams."""
    if n_cameras == 1:
        camera_names = ["camera_front_wide_120fov"]
        prompt = (
            "Driving scene from a front-facing car camera. Urban environment with roads, "
            "vehicles, pedestrians, traffic signs, and buildings. Clear visibility, "
            "realistic lighting, photorealistic quality. High resolution dashcam footage "
            "of city driving."
        )
        config_name = "sv_2steps_chunk2_loc6_lightvae_lighttae"
    elif n_cameras == 4:
        camera_names = [
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
            "camera_front_wide_120fov",
        ]
        prompt = (
            "Wide-angle urban street scene from a low, dashboard-level viewpoint. "
            "A straight two-lane road with a faded center line and curbside parking on "
            "both sides. Parked sedans and SUVs in neutral colors line the curbs. On the "
            "right, a white stucco mid-rise building with blue fabric awnings, rectangular "
            "windows, and small storefronts at street level. On the left, a low commercial "
            "strip with dark trim, glass fronts, signage, and shaded sidewalks. Mature green "
            "trees punctuate both sides. Clear blue sky with sparse soft clouds. Bright midday "
            "sunlight, natural colors, realistic materials, crisp shadows, clean asphalt texture."
        )
        config_name = "mv_2steps_chunk4_loc8_pshuffle_lighttae"
    else:
        raise ValueError(f"Number of cameras must be 1 or 4, got {n_cameras}")

    data = [
        {
            "hdmap_video_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.mp4"),
            "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, f"{name}.png"),
            "prompt": prompt,
        }
        for name in camera_names
    ]
    return [config_name, *camera_names], data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n_cameras", type=int, default=1, help="Number of cameras (1 or 4)."
    )
    parser.add_argument(
        "--total_blocks", type=int, default=60, help="Total blocks to generate."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Base random seed. Each distributed rank uses seed + rank.",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=30,
        help="FPS used when saving the HDMap + generated video canvas.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help=(
            "Optional latent chunk override for configs that expose num_chunks, "
            "currently only the bidirectional recipe."
        ),
    )
    parser.add_argument(
        "--overwrite_config_name",
        type=str,
        default=None,
        choices=[None, *sorted(ALPADREAMS_CONFIG_BUILDERS.keys())],
        help="Optionally override the per-n_cameras default config name.",
    )
    parser.add_argument(
        "--hdmap_video_path",
        type=str,
        default=None,
        help=(
            "Optional HDMap video path override. For multi-view runs, pass "
            "one comma-separated path per camera in the default camera order."
        ),
    )
    parser.add_argument(
        "--first_frame_path",
        type=str,
        default=None,
        help=(
            "Optional first-frame image or video path override. If a video is "
            "provided, frame 0 is used. For multi-view runs, pass one "
            "comma-separated path per camera in the default camera order."
        ),
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile of the DiT network.",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=None,
        help=(
            "Optional path to a .pt file produced by a previous "
            "--save_embeddings_path run. When set, the Cosmos-Reason1 "
            "text encoder and Wan VAE first-frame image encoder are NOT "
            "loaded (saving ~14 GB of VRAM); the cache is hydrated from "
            "the precomputed tensors instead."
        ),
    )
    parser.add_argument(
        "--offload_text_encoder",
        action="store_true",
        help=(
            "Load only the one-shot encoders first, compute text + "
            "first-frame embeddings, free the encoders, and only then "
            "build the AR pipeline. Lowers peak VRAM compared to the "
            "default path (which holds the encoders and the DiT in "
            "memory simultaneously) without requiring a separate "
            "precompute step / saved file. Mutually exclusive with "
            "--embeddings_path and --save_embeddings_path."
        ),
    )
    parser.add_argument(
        "--save_embeddings_path",
        type=str,
        default=None,
        help=(
            "Run as an offline precompute: load ONLY the one-shot "
            "encoders (Cosmos-Reason1 text encoder + Wan VAE first-frame "
            "image encoder), dump their outputs to this .pt path, then "
            "exit. Skips distributed init, the AR pipeline, and the "
            "rollout. Pair with --embeddings_path on a subsequent "
            "run. Mutually exclusive with --embeddings_path and "
            "--offload_text_encoder."
        ),
    )
    return parser.parse_args()


def _save_embeddings_and_exit(args: argparse.Namespace) -> None:
    """Offline precompute path: dump text + first-frame embeddings, then exit.

    Loads ONLY the one-shot encoders from the chosen config -- the DiT,
    per-AR-step encoder, and decoder are NOT loaded (they're not needed
    to produce the embeddings, and skipping them keeps precompute
    lightweight). No distributed init: this is a single-GPU producer.
    """
    config_meta, data = _build_data(args.n_cameras)
    _apply_data_overrides(
        data,
        hdmap_video_path=args.hdmap_video_path,
        first_frame_path=args.first_frame_path,
    )
    config_name = (
        args.overwrite_config_name
        if args.overwrite_config_name is not None
        else config_meta[0]
    )
    camera_names = config_meta[1:]
    output_path = args.save_embeddings_path

    print(
        f"Precomputing alpadreams embeddings for {args.n_cameras} cameras "
        f"with config: {config_name}"
    )

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    credential_path = str(REPO_ROOT / "credentials/s3_checkpoint.secret")
    assert os.path.exists(credential_path), (
        f"Credential file not found at {credential_path}"
    )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=credential_path,
        cache_dir=EXAMPLE_DATA_DIR_LOCAL,
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing from S3",
    )

    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"

    builder = ALPADREAMS_CONFIG_BUILDERS[config_name]
    # Build config metadata only; the DiT/decoder are not instantiated in this path.
    num_chunks_args = _num_chunks_args(config_name, args.num_chunks)
    pipeline_config = builder(
        compile_network=False,
        seed=0,
        **num_chunks_args,
    )

    assert (
        pipeline_config.text_encoder is not None
        and pipeline_config.image_encoder is not None
    ), (
        "Cannot precompute: the chosen config has text_encoder/image_encoder "
        "set to None. Use a config that keeps both encoders configured."
    )

    needs_negative_text = _needs_negative_text(pipeline_config)
    transformer_cfg = pipeline_config.diffusion_model.transformer
    assert isinstance(transformer_cfg, CosmosTransformerConfig)
    assert isinstance(
        pipeline_config.decoder, (WanVAEDecoderConfig, TeahvVAEDecoderConfig)
    )
    # Per-rollout latent (h, w) is no longer baked into the transformer
    # config; use the recipe's canonical pixel-space defaults.
    pixel_h = DEFAULT_VIDEO_HEIGHT
    pixel_w = DEFAULT_VIDEO_WIDTH

    text_encoder = pipeline_config.text_encoder.setup().to(device=device)
    image_encoder = pipeline_config.image_encoder.setup().to(device=device)

    first_frames: list[torch.Tensor] = []
    prompts: list[str] = []
    for entry in data:
        first_frame = _read_first_frame(entry["first_frame_path"])
        first_frame = cv2.resize(first_frame, (pixel_w, pixel_h))
        first_frame_t = (
            torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
        )
        first_frames.append(rearrange(first_frame_t, "h w c -> 1 c h w"))
        prompts.append(entry["prompt"])

    # [B=1, V, 1, C, H, W]
    first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(0)
    prompts_2d: list[list[str]] = [prompts]  # [B=1, V]

    with torch.no_grad():
        text_embeddings = torch.stack(
            [text_encoder(t) for t in prompts_2d], dim=0
        )  # [B, V, L, D]
        image_embeddings = image_encoder(first_frames_t)  # [B, V, 1, Cl, Hl, Wl]

    payload = {
        "text_embeddings": text_embeddings.cpu(),
        "image_embeddings": image_embeddings.cpu(),
        "view_names": camera_names,
        "metadata": {
            "config_name": config_name,
            "n_cameras": args.n_cameras,
            "prompts": prompts,
            "pixel_h": pixel_h,
            "pixel_w": pixel_w,
        },
    }
    if needs_negative_text:
        with torch.no_grad():
            negative_text_embeddings = torch.stack(
                [
                    text_encoder([NEGATIVE_PROMPT for _ in prompt_row])
                    for prompt_row in prompts_2d
                ],
                dim=0,
            )
        payload["negative_text_embeddings"] = negative_text_embeddings.cpu()
        payload["metadata"]["negative_prompt"] = NEGATIVE_PROMPT
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    print(
        f"saved precomputed embeddings to {output_path} "
        f"(text {tuple(text_embeddings.shape)} {text_embeddings.dtype}, "
        f"image {tuple(image_embeddings.shape)} {image_embeddings.dtype})"
    )


def main() -> None:
    args = parse_args()
    assert args.n_cameras in (1, 4), "Only 1 or 4 cameras are supported"
    n_modes = sum(
        bool(x)
        for x in (
            args.embeddings_path,
            args.offload_text_encoder,
            args.save_embeddings_path,
        )
    )
    assert n_modes <= 1, (
        "--embeddings_path, --offload_text_encoder, and "
        "--save_embeddings_path are mutually exclusive: pick at most "
        "one."
    )

    # Offline-precompute path: dump embeddings and exit before any
    # distributed init or AR pipeline construction.
    if args.save_embeddings_path is not None:
        _save_embeddings_and_exit(args)
        return

    config_meta, data = _build_data(args.n_cameras)
    _apply_data_overrides(
        data,
        hdmap_video_path=args.hdmap_video_path,
        first_frame_path=args.first_frame_path,
    )
    config_name = (
        args.overwrite_config_name
        if args.overwrite_config_name is not None
        else config_meta[0]
    )
    camera_names = config_meta[1:]

    print(
        f"Running alpadreams inference with {args.n_cameras} cameras and config: "
        f"{config_name}"
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    distributed_init()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    print(
        f"initialized distributed inference with world size {world_size} and rank {rank}"
    )
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    credential_path = str(REPO_ROOT / "credentials/s3_checkpoint.secret")
    assert os.path.exists(credential_path), (
        f"Credential file not found at {credential_path}"
    )
    sync_s3_dir_to_local(
        s3_dir=EXAMPLE_DATA_DIR_S3,
        s3_credential_path=credential_path,
        cache_dir=EXAMPLE_DATA_DIR_LOCAL,
        max_workers=10,
        show_progress=True,
        verify_checksum=True,
        desc="Syncing from S3",
    )

    # ``huggingface_hub`` already uses ``HF_TOKEN`` automatically as the
    # bearer for every API call when the env var is set, so calling
    # ``login()`` here is redundant -- and harmful in distributed runs:
    # every rank would hit ``GET /api/whoami-v2`` concurrently from the
    # same IP and trip the per-IP 429 bucket. Just assert the token is
    # present and let the lib pick it up on demand.
    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"
    if rank == 0:
        print("HF_TOKEN detected; using env-var auth for huggingface_hub")

    builder = ALPADREAMS_CONFIG_BUILDERS[config_name]
    num_chunks_args = _num_chunks_args(config_name, args.num_chunks)
    if num_chunks_args and rank == 0:
        print(
            "Using bidirectional num_chunks="
            f"{num_chunks_args['num_chunks']} for this runtime."
        )
    pipeline_config = builder(
        compile_network=not args.no_compile,
        seed=args.seed + rank,
        **num_chunks_args,
    )
    needs_negative_text = _needs_negative_text(pipeline_config)

    # Offload-text-encoder path: stand up ONLY the one-shot encoders
    # here, compute the embeddings, free the encoders, then null the
    # configs so pipeline.setup() below skips them entirely. Peak VRAM
    # is now max(encoders, AR pipeline) instead of their sum.
    precomputed_embeddings: dict[str, torch.Tensor] | None = None
    if args.offload_text_encoder:
        assert (
            pipeline_config.text_encoder is not None
            and pipeline_config.image_encoder is not None
        ), "Cannot precompute: encoder configs are already None on this builder."

        # The recipe's canonical pixel-space defaults govern resizing;
        # per-rollout latent dims are derived from the resulting image
        # later in ``pipeline.initialize_cache``.
        assert isinstance(
            pipeline_config.diffusion_model.transformer, CosmosTransformerConfig
        )
        assert isinstance(
            pipeline_config.decoder, (WanVAEDecoderConfig, TeahvVAEDecoderConfig)
        )
        pre_pixel_h = DEFAULT_VIDEO_HEIGHT
        pre_pixel_w = DEFAULT_VIDEO_WIDTH

        pre_first_frames: list[torch.Tensor] = []
        pre_prompts: list[str] = []
        for entry in data:
            ff = _read_first_frame(entry["first_frame_path"])
            ff = cv2.resize(ff, (pre_pixel_w, pre_pixel_h))
            ff_t = torch.from_numpy(ff).to(dtype=dtype, device=device) / 127.5 - 1.0
            pre_first_frames.append(rearrange(ff_t, "h w c -> 1 c h w"))
            pre_prompts.append(entry["prompt"])
        pre_first_frames_t = torch.stack(pre_first_frames, dim=0).unsqueeze(0)
        pre_prompts_2d = [pre_prompts]

        if rank == 0:
            print("[offload text encoder] loading encoders and computing embeddings")
        text_encoder = pipeline_config.text_encoder.setup().to(device=device)
        image_encoder = pipeline_config.image_encoder.setup().to(device=device)
        with torch.no_grad():
            text_embeddings = torch.stack(
                [text_encoder(t) for t in pre_prompts_2d], dim=0
            ).cpu()
            image_embeddings = image_encoder(pre_first_frames_t).cpu()
        precomputed_embeddings = {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
        }
        if needs_negative_text:
            with torch.no_grad():
                precomputed_embeddings["negative_text_embeddings"] = torch.stack(
                    [
                        text_encoder([NEGATIVE_PROMPT for _ in prompt_row])
                        for prompt_row in pre_prompts_2d
                    ],
                    dim=0,
                ).cpu()

        del text_encoder, image_encoder, pre_first_frames, pre_first_frames_t
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if rank == 0:
            print(
                f"[offload text encoder] done; freed encoders. "
                f"text {tuple(text_embeddings.shape)} {text_embeddings.dtype}, "
                f"image {tuple(image_embeddings.shape)} {image_embeddings.dtype}"
            )
        pipeline_config.text_encoder = None
        pipeline_config.image_encoder = None

    if args.embeddings_path is not None:
        # Skip the one-shot encoder load entirely; embeddings are
        # hydrated below from the precomputed file.
        pipeline_config.text_encoder = None
        pipeline_config.image_encoder = None
    pipeline = pipeline_config.setup()
    assert isinstance(pipeline, AlpadreamsPipeline)
    pipeline.to(device=device)

    # Per-rollout (latent) resolution is no longer baked into the
    # transformer config; resize pixel-space inputs to the recipe's
    # canonical defaults so the encoded image latent's spatial dims drive
    # the per-rollout (height, width) inside ``pipeline.initialize_cache``.
    transformer_cfg = pipeline.diffusion_model.transformer.config
    assert isinstance(transformer_cfg, CosmosTransformerConfig)
    assert isinstance(pipeline.decoder, (WanVAEDecoder, TeahvVAEDecoder))
    pixel_h = DEFAULT_VIDEO_HEIGHT
    pixel_w = DEFAULT_VIDEO_WIDTH

    first_frames: list[torch.Tensor] = []
    hdmap_videos: list[torch.Tensor] = []
    prompts: list[str] = []
    # First frames are only needed when the image encoder will run
    # below; in both precomputed paths it has already been consumed
    # (offload: above; from-disk: never, since embeddings are loaded).
    needs_first_frames = args.embeddings_path is None and not args.offload_text_encoder
    for entry in data:
        if needs_first_frames:
            first_frame = _read_first_frame(entry["first_frame_path"])
            first_frame = cv2.resize(first_frame, (pixel_w, pixel_h))
            first_frame_t = (
                torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5
                - 1.0
            )
            first_frames.append(rearrange(first_frame_t, "h w c -> 1 c h w"))

        hdmap_video_np = media.read_video(entry["hdmap_video_path"])[..., :3]
        if hdmap_video_np.shape[1:3] != (pixel_h, pixel_w):
            hdmap_video_np = np.stack(
                [cv2.resize(f, (pixel_w, pixel_h)) for f in hdmap_video_np], axis=0
            )
        hdmap_video_t = (
            torch.from_numpy(hdmap_video_np).to(dtype=dtype, device=device) / 127.5
            - 1.0
        )
        hdmap_videos.append(rearrange(hdmap_video_t, "t h w c -> t c h w"))

        prompts.append(entry["prompt"])

    hdmap_videos_t = torch.stack(hdmap_videos, dim=0).unsqueeze(
        0
    )  # [B=1, V, T, C, H, W]
    hdmap_num_frames = hdmap_videos_t.shape[2]
    print("loaded hdmap_videos.shape:", hdmap_videos_t.shape)

    if args.embeddings_path is not None:
        print(f"loading precomputed embeddings from {args.embeddings_path}")
        payload = torch.load(args.embeddings_path, map_location="cpu")
        # Trust the saved view ordering; sanity-check it matches the
        # camera ordering for this run so the embeddings line up
        # correctly when the multi-view CP split is applied.
        saved_view_names = payload["view_names"]
        assert saved_view_names == camera_names, (
            f"view_names mismatch: saved {saved_view_names} vs current "
            f"{camera_names}. Re-run precompute with the matching --n_cameras."
        )
        cache = pipeline.initialize_cache_from_embeddings(
            text_embeddings=payload["text_embeddings"],
            image_embeddings=payload["image_embeddings"],
            negative_text_embeddings=(
                payload["negative_text_embeddings"] if needs_negative_text else None
            ),
            view_names=saved_view_names,
        )
    elif precomputed_embeddings is not None:
        cache = pipeline.initialize_cache_from_embeddings(
            text_embeddings=precomputed_embeddings["text_embeddings"],
            image_embeddings=precomputed_embeddings["image_embeddings"],
            negative_text_embeddings=(
                precomputed_embeddings["negative_text_embeddings"]
                if needs_negative_text
                else None
            ),
            view_names=camera_names,
        )
    else:
        first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(
            0
        )  # [B=1, V, 1, C, H, W]
        prompts_2d: list[list[str]] = [prompts]  # [B=1, V]
        cache = pipeline.initialize_cache(
            text=prompts_2d,
            image=first_frames_t,
            view_names=camera_names,
        )
        # This demo runs a single rollout, so drop the one-shot text and
        # first-frame image encoders before the AR loop to free VRAM
        # (Cosmos-Reason1-7B alone is ~14 GB in bf16). The gRPC server keeps
        # them around since it reuses the pipeline across sessions.
        pipeline.release_oneshot_encoders()

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    generated_video: list[torch.Tensor] = []
    stats_history: list[dict[str, float]] = []
    start = 0
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_frames(i)
        end = start + num_frames
        if end > hdmap_num_frames:
            break
        video_chunk = pipeline.generate(
            autoregressive_index=i,
            cache=cache,
            hdmap=hdmap_videos_t[:, :, start:end],
        )
        stats = pipeline.finalize(i, cache)
        if stats is not None:
            stats_history.append({"autoregressive_index": i, **stats})
        generated_video.append(video_chunk.cpu())
        start = end

    video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W]
    generated_num_frames = video.shape[2]

    if rank == 0:
        print("end of streaming inference, generated_video.shape:", video.shape)

    if rank == 0:
        condition = hdmap_videos_t[:, :, :generated_num_frames].cpu()
        canvas = rearrange(
            torch.cat([condition, video], dim=-2),
            "1 v t c h w -> t h (v w) c",
        )
        canvas = (canvas.float().numpy() + 1.0) / 2.0
        canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        output_prefix = (
            config_name
            if config_name.startswith("alpadreams_")
            else f"alpadreams_{config_name}"
        )
        save_path = f"{REPO_ROOT}/outputs/{output_prefix}_{world_size}gpus.mp4"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=args.output_fps)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = (
                f"{REPO_ROOT}/outputs/stats_{output_prefix}_{world_size}gpus.json"
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
