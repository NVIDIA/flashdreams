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

"""Source-video corpus for the edit-SFT data pipeline (Tier-2b, recipe A).

Rolls the distilled model over every locally cached sample clip under its
own prompt and saves the decoded videos plus a manifest — the *source*
side of the (source, instruction, edited) triplets that JoyAI-Video-Edit
produces downstream. Per-chunk latents are saved alongside so the SFT
context-replay phase does not need to re-encode the videos.

Env knobs: ``N_CLIPS``, ``N_CHUNKS``, ``SEED``, ``OUT_DIR``.

Run from the repo root::

    .venv/bin/python integrations/omnidreams/edit_sft/generate_sources.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    load_video_tensor,
    write_video_tensor,
)

SAMPLES_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/datasets--nvidia--omni-dreams-samples/snapshots"
)
N_CLIPS = int(os.environ.get("N_CLIPS", "20"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "28"))
SEED = int(os.environ.get("SEED", "42"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/edit_sft/outputs/sources")
)


def _local_clips() -> list[tuple[str, Path, Path, str]]:
    """(uuid, hdmap, first_frame, prompt) for every fully cached sample clip."""
    clips = []
    for prompt_path in sorted(SAMPLES_ROOT.glob("*/data/single_view/*/prompt.txt")):
        clip_dir = prompt_path.parent
        hdmaps = sorted(clip_dir.glob("*_hdmap.mp4"))
        frames = sorted(clip_dir.glob("first_frame.png"))
        if hdmaps and frames:
            clips.append(
                (clip_dir.name, hdmaps[0], frames[0], prompt_path.read_text().strip())
            )
    return clips


@torch.no_grad()
def main() -> None:
    """Roll and save the source corpus."""
    clips = _local_clips()[:N_CLIPS]
    assert clips, f"no cached sample clips under {SAMPLES_ROOT}"
    total_frames = 5 + (N_CHUNKS - 1) * 8

    cfg = derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            seed=SEED,
            transformer=dict(compile_network=False, use_cuda_graph=False),
        ),
    )
    pipe = cfg.setup()
    assert isinstance(pipe, OmnidreamsPipeline)
    pipe = pipe.to("cuda")
    device = pipe.device

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for uuid, hdmap_path, frame_path, prompt in clips:
        out_mp4 = OUT_DIR / f"{uuid}.mp4"
        out_lat = OUT_DIR / f"{uuid}_latents.pt"
        if out_mp4.exists() and out_lat.exists():
            print(f"skip {uuid} (exists)", flush=True)
        else:
            hdmap = load_video_tensor(
                hdmap_path,
                pixel_height=DEFAULT_VIDEO_HEIGHT,
                pixel_width=DEFAULT_VIDEO_WIDTH,
                device=device,
                dtype=torch.bfloat16,
            )
            if hdmap.shape[0] < total_frames:
                print(f"skip {uuid}: only {hdmap.shape[0]} HDMap frames", flush=True)
                continue
            hdmap = hdmap[:total_frames][None, None]
            first = load_first_frame_tensor(
                frame_path,
                pixel_height=DEFAULT_VIDEO_HEIGHT,
                pixel_width=DEFAULT_VIDEO_WIDTH,
                device=device,
                dtype=torch.bfloat16,
            )[None, None]

            pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)
            cache = pipe.initialize_cache(text=[[prompt]], image=first)
            chunks, latents, start = [], [], 0
            for ar_idx in range(N_CHUNKS):
                num_frames = pipe.get_num_frames(ar_idx)
                chunk = pipe.generate(
                    ar_idx, cache, hdmap=hdmap[:, :, start : start + num_frames]
                )
                final_state = cache.final_state
                assert final_state is not None
                latents.append(final_state.clean_latent.detach().to("cpu"))
                pipe.finalize(ar_idx, cache)
                chunks.append(chunk[0, 0].float().cpu())
                start += num_frames
            video = torch.cat(chunks, dim=0)
            write_video_tensor(video, out_mp4, fps=30, layout="tchw")
            torch.save({"latents": latents, "prompt": prompt, "seed": SEED}, out_lat)
            del cache
            torch.cuda.empty_cache()
            print(f"rolled {uuid}: {video.shape[0]} frames", flush=True)

        manifest.append(
            {
                "uuid": uuid,
                "video": out_mp4.name,
                "latents": out_lat.name,
                "prompt": prompt,
                "n_chunks": N_CHUNKS,
                "seed": SEED,
            }
        )

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"SOURCES-DONE | {len(manifest)} clips -> {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
