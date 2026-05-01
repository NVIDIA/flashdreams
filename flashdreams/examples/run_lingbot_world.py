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

"""Lingbot World streaming demo on top of :class:`LingbotWorldInferencePipeline`.

Distributed streaming inference entrypoint for the LingBot-World-Fast
camera-control I2V checkpoints. Lingbot World is I2V-only: every
rollout needs a first frame *and* a per-AR-step camera stream.

- The first frame is stashed once in the cache via
  ``initialize_cache(image=...)``.
- At each AR step the camera slice (intrinsics + poses + world
  scale) is packed into a :class:`CamCtrlInput` and handed to
  ``pipeline.generate(..., input=...)``; the pipeline internally
  bundles it with the matching first-frame pixel chunk into the
  :class:`I2VCamCtrlInput` that :class:`I2VCamCtrlEncoder` expects.
  The composite encoder (Wan-VAE I2V control encoder + PixelShuffle
  pseudo-VAE over Plücker rays) feeds a :class:`LingbotWorldTransformer`
  (a :class:`Wan21Transformer` subclass with per-block
  ``CamCtrlBlock``).

Rollouts are multi-AR-step by design (streaming); the loop stops
when the camera stream is exhausted or ``--total_blocks`` AR chunks
have been produced, whichever comes first.

Run::

    torchrun --nproc_per_node=N \\
        flashdreams/examples/run_lingbot_world.py \\
        --total_blocks 60 \\
        --config_name LingBot-World-Fast
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
from flashdreams.recipes.lingbot_world.config import (
    LINGBOT_WORLD_CONFIG_BUILDERS,
)
from flashdreams.recipes.lingbot_world.encoder.camctrl import CamCtrlInput
from flashdreams.recipes.lingbot_world.encoder.utils import compute_relative_poses
from flashdreams.recipes.lingbot_world.pipeline import LingbotWorldInferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR_S3 = "s3://flashdreams/assets/example_data/lingbot_world"
EXAMPLE_DATA_DIR_LOCAL = str(REPO_ROOT / "assets/example_data/lingbot_world")

CAMERA_NAMES = ["default"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lingbot World streaming camera-control I2V demo on top of "
            "LingbotWorldInferencePipeline."
        )
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="LingBot-World-Fast",
        choices=sorted(LINGBOT_WORLD_CONFIG_BUILDERS.keys()),
        help="Streaming checkpoint preset to load.",
    )
    parser.add_argument(
        "--total_blocks",
        type=int,
        default=60,
        help="Upper bound on the number of AR chunks to generate.",
    )
    parser.add_argument("--video_height", type=int, default=464)
    parser.add_argument("--video_width", type=int, default=832)
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help=(
            "Disable torch.compile of the DiT network (faster startup, slower steps)."
        ),
    )
    return parser.parse_args()


def _should_init_distributed() -> bool:
    return os.getenv("RANK") is not None and os.getenv("WORLD_SIZE") is not None


def main() -> None:
    args = parse_args()
    print(f"Running Lingbot World inference with config: {args.config_name}")

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if _should_init_distributed():
        distributed_init()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(
            f"initialized distributed inference with world size {world_size} "
            f"and rank {rank}"
        )
    else:
        world_size = 1
        rank = 0
        print("running single-process inference (torch.distributed disabled)")

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
    # same IP and trip the per-IP 429 bucket.
    assert os.getenv("HF_TOKEN") is not None, "HF_TOKEN is not set"
    if rank == 0:
        print("HF_TOKEN detected; using env-var auth for huggingface_hub")

    # ---------------------------------------------------------------- data
    data = [
        {
            "pose_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "poses.npy"),
            "intrinsic_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "intrinsics.npy"),
            "first_frame_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "image.jpg"),
            "text_prompt_path": os.path.join(EXAMPLE_DATA_DIR_LOCAL, "prompt.txt"),
        }
        for _ in CAMERA_NAMES
    ]

    camera_intrinsics: list[torch.Tensor] = []
    camera_poses: list[torch.Tensor] = []
    first_frames: list[torch.Tensor] = []
    prompts: list[str] = []
    trans_normalizer: torch.Tensor | float = 1.0
    for entry in data:
        first_frame = media.read_image(entry["first_frame_path"])
        first_frame = cv2.resize(first_frame, (args.video_width, args.video_height))
        first_frame_t = (
            torch.from_numpy(first_frame).to(dtype=dtype, device=device) / 127.5 - 1.0
        )
        first_frames.append(rearrange(first_frame_t, "h w c -> 1 c h w"))

        Ks = np.load(entry["intrinsic_path"])  # [T, 4]
        Ks_t = torch.from_numpy(Ks).to(device=device, dtype=torch.float32)
        camera_intrinsics.append(Ks_t)

        c2ws = np.load(entry["pose_path"])  # [T, 4, 4]
        c2ws_t = torch.from_numpy(c2ws).to(device=device, dtype=torch.float32)
        camera_poses.append(c2ws_t)

        # Only needed for the world-scale normalizer; the actual Plücker
        # volume is rendered per-AR-step inside I2VCamCtrlEncoder.
        _, trans_normalizer = compute_relative_poses(c2ws_t, framewise=True)

        with open(entry["text_prompt_path"], "r") as f:
            prompts.append(f.readlines()[0])

    first_frames_t = torch.stack(first_frames, dim=0).unsqueeze(
        0
    )  # [B=1, V, 1, C, H, W]
    camera_intrinsics_t = torch.stack(camera_intrinsics, dim=0).unsqueeze(
        0
    )  # [B=1, V, T, 4]
    camera_poses_t = torch.stack(camera_poses, dim=0).unsqueeze(0)  # [B=1, V, T, 4, 4]
    # Lingbot's batch_shape is (1, 1) -> flat len(text) == 1 prompt.
    assert len(prompts) == 1, (
        f"Lingbot demo wires a (1, 1) batch; expected 1 prompt, got {len(prompts)}"
    )
    total_camera_frames = camera_poses_t.shape[2]
    print(
        f"loaded first_frames.shape: {tuple(first_frames_t.shape)}, "
        f"camera_poses.shape: {tuple(camera_poses_t.shape)}"
    )

    # --------------------------------------------------------- pipeline init
    builder = LINGBOT_WORLD_CONFIG_BUILDERS[args.config_name]
    pipeline = (
        builder(
            cp_size=world_size,
            compile_network=not args.no_compile,
            seed=42 + rank,
            enable_sync_and_profile=True,
        )
        .setup()
        .to(device=device)
    )
    assert isinstance(pipeline, LingbotWorldInferencePipeline)
    cache = pipeline.initialize_cache(text=prompts, image=first_frames_t)

    torch.cuda.synchronize()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---------------------------------------------------------------- rollout
    generated_video: list[torch.Tensor] = []
    stats_history: list[dict[str, float]] = []
    start = 0
    for i in range(args.total_blocks):
        num_frames = pipeline.get_num_output_frames(i)
        end = start + num_frames
        if end > total_camera_frames:
            break
        print(
            f"autoregressive_index: {i}, num_frames: {num_frames}, "
            f"start: {start}, end: {end}"
        )
        # Slice the camera stream for this AR step. The first-frame
        # pixel chunk is built internally by the pipeline from
        # ``cache.image``.
        camctrl_input = CamCtrlInput(
            intrinsics=camera_intrinsics_t[:, :, start:end],
            poses=camera_poses_t[:, :, start:end],
            world_scale=float(trans_normalizer),
        )
        video_chunk = pipeline.generate(
            autoregressive_index=i,
            cache=cache,
            input=camctrl_input,
        )
        stats = pipeline.finalize(i, cache)
        if stats is not None:
            stats_history.append({"autoregressive_index": i, **stats})
        generated_video.append(video_chunk.cpu())
        start = end

    video = torch.cat(generated_video, dim=2)  # [B, V, T, C, H, W]
    print("end of streaming inference, generated_video.shape:", video.shape)

    if rank == 0:
        canvas = rearrange(video, "1 v t c h w -> t h (v w) c")
        canvas = (canvas.float().numpy() + 1.0) / 2.0
        canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
        save_path = (
            f"{REPO_ROOT}/outputs/lingbot_{args.config_name}_{world_size}gpus.mp4"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        media.write_video(save_path, canvas, fps=16)
        print(f"saved generated video to {save_path}")

        if stats_history:
            stats_path = (
                f"{REPO_ROOT}/outputs/"
                f"stats_lingbot_{args.config_name}_{world_size}gpus.json"
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
