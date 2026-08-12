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

"""Calibration sweep: which mid-stream edits land, and at what guidance.

One pipeline load, then RNG-matched rollouts for a bank of edit prompts x
guidance scales against a shared control. The snow prompts include the
scene bundle's own snowstorm phrasing (training-distribution wording) to
separate "snow is OOD" from "my prompt was OOD". Writes per-combo videos,
a per-chunk divergence report, and a comparison grid.

Run from the repo root::

    .venv/bin/python integrations/omnidreams/scripts/sweep_text_edit.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mediapy as media
import numpy as np
import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import DEFAULT_VIDEO_HEIGHT, DEFAULT_VIDEO_WIDTH
from torch import Tensor

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
UUID = os.environ.get("UUID", "23599139-948f-4681-b7f4-74794113086d")
N_CHUNKS = int(os.environ.get("N_CHUNKS", "28"))
SWAP_AT = int(os.environ.get("SWAP_AT", "8"))
SEED = int(os.environ.get("SEED", "42"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/scripts/outputs/edit_sweep")
)

# The scene bundle's own weather phrasings (training-distribution wording),
# lightly de-scene-specified (drop the named parked cars).
SNOW_NATIVE = (
    "A dashcam perspective from inside a vehicle driving down a wide suburban "
    "residential street during a snowstorm. The road is heavily covered in "
    "white snow with visible parallel tire tracks. Vehicles parked along the "
    "curb are coated in a layer of snow. The surrounding houses, lawns, and "
    "large trees are completely blanketed in winter snow. The sky is overcast "
    "and gray with snowflakes visibly falling. In the foreground, the bottom "
    "of the windshield and the car's hood are visible, with snow accumulating "
    "around the windshield wipers."
)
SNOW_MINE = (
    "Driving scene from a front-facing car camera at night in a heavy "
    "snowstorm. Thick snow falling, snow-covered road and buildings, "
    "headlights and streetlights glowing through the snow. Photorealistic "
    "dashcam footage."
)
RAIN_NIGHT_NATIVE = (
    "A deep night sky of dark blue and grey is heavy with persistent, visible "
    "rain streaks. The overall atmosphere is dark and thoroughly wet. An "
    "asphalt road, marked by double yellow center lines, extends into the "
    "distance, its surface completely saturated with sheeting water, creating "
    "a glossy mirror that breaks and complexifies the reflections of multiple "
    "warm-toned overhead streetlights. In the immediate lower foreground, the "
    "car's wet hood is covered with rain droplets and reflecting light."
)
FOG = (
    "A dashcam perspective of a suburban street in extremely dense fog. "
    "Visibility is very low; buildings and trees fade into a uniform white-"
    "gray haze within tens of meters. Faint silhouettes of parked cars line "
    "the curb, headlights diffuse into soft glows. Muted, desaturated colors."
)
NIGHT = (
    "A dashcam perspective of a suburban street late at night. Dark sky, the "
    "road lit by warm streetlights and the car's headlights, parked cars in "
    "shadow along the curb, illuminated house windows, deep shadows under the "
    "trees. Photorealistic night dashcam footage."
)
SUNSET = (
    "A dashcam perspective of a suburban street at golden-hour sunset. Warm "
    "orange low sun ahead near the horizon, long shadows across the road, "
    "golden light on the trees and house facades, glowing warm sky with a few "
    "pink clouds. Photorealistic dashcam footage."
)

# (name, prompt, guidance_scale, guidance_chunks); scale 1.0 = plain swap.
COMBOS: list[tuple[str, str, float, int]] = [
    ("snow_native_plain", SNOW_NATIVE, 1.0, 0),
    ("snow_native_g3", SNOW_NATIVE, 3.0, 6),
    ("snow_native_g5", SNOW_NATIVE, 5.0, 6),
    ("snow_mine_g3", SNOW_MINE, 3.0, 6),
    ("snow_mine_g5", SNOW_MINE, 5.0, 6),
    ("rain_night_g3", RAIN_NIGHT_NATIVE, 3.0, 6),
    ("fog_g3", FOG, 3.0, 6),
    ("night_g3", NIGHT, 3.0, 6),
    ("sunset_g3", SUNSET, 3.0, 6),
]


def _sample_paths(uuid: str) -> tuple[Path, Path, str]:
    hdmaps = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/*_hdmap.mp4"))
    frames = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/first_frame.png"))
    prompts = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/prompt.txt"))
    assert hdmaps and frames and prompts, f"sample {uuid} missing from local HF cache"
    return hdmaps[0], frames[0], prompts[0].read_text().strip()


@torch.no_grad()
def _rollout(
    pipe: OmnidreamsPipeline,
    *,
    hdmap: Tensor,
    first: Tensor,
    base_prompt: str,
    edit: tuple[str, float, int] | None,
) -> Tensor:
    pipe.diffusion_model._rng = torch.Generator(device=pipe.device).manual_seed(SEED)
    cache = pipe.initialize_cache(text=[[base_prompt]], image=first)
    chunks: list[Tensor] = []
    start = 0
    for ar_idx in range(N_CHUNKS):
        if edit is not None and ar_idx == SWAP_AT:
            prompt, scale, guide_chunks = edit
            pipe.replace_text(
                cache,
                [[prompt]],
                guidance_scale=scale,
                guidance_chunks=guide_chunks,
            )
        num_frames = pipe.get_num_frames(ar_idx)
        chunk = pipe.generate(
            ar_idx, cache, hdmap=hdmap[:, :, start : start + num_frames]
        )
        pipe.finalize(ar_idx, cache)
        chunks.append(chunk[0, 0].float().cpu())
        start += num_frames
    del cache
    torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def _per_chunk_gap(a: Tensor, b: Tensor) -> list[float]:
    gaps, start = [], 0
    for ar_idx in range(N_CHUNKS):
        n = 5 if ar_idx == 0 else 8
        gaps.append(
            float((a[start : start + n] - b[start : start + n]).abs().mean() * 127.5)
        )
        start += n
    return gaps


def main() -> None:
    hdmap_path, frame_path, clip_prompt = _sample_paths(UUID)
    total_frames = 5 + (N_CHUNKS - 1) * 8
    device = torch.device("cuda")
    hdmap = load_video_tensor(
        hdmap_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )[:total_frames][None, None]
    first = load_first_frame_tensor(
        frame_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )[None, None]

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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"clip {UUID}: {clip_prompt[:100]}...")
    print("rolling out control ...", flush=True)
    control = _rollout(
        pipe, hdmap=hdmap, first=first, base_prompt=clip_prompt, edit=None
    )
    write_video_tensor(control, OUT_DIR / "control.mp4", fps=30, layout="tchw")

    report: dict[str, dict] = {}
    videos: dict[str, Tensor] = {"control": control}
    for name, prompt, scale, guide_chunks in COMBOS:
        print(f"rolling out {name} ...", flush=True)
        video = _rollout(
            pipe,
            hdmap=hdmap,
            first=first,
            base_prompt=clip_prompt,
            edit=(prompt, scale, guide_chunks),
        )
        videos[name] = video
        write_video_tensor(video, OUT_DIR / f"{name}.mp4", fps=30, layout="tchw")
        gaps = _per_chunk_gap(video, control)
        report[name] = {
            "prompt": prompt,
            "guidance_scale": scale,
            "guidance_chunks": guide_chunks,
            "pre_swap_max_gap": max(gaps[:SWAP_AT]),
            "post_swap_gaps": gaps[SWAP_AT:],
        }
        post = gaps[SWAP_AT:]
        print(
            f"{name:>18}: pre {max(gaps[:SWAP_AT]):5.3f}  "
            f"post first/mid/last {post[0]:6.2f} {post[len(post) // 2]:6.2f} {post[-1]:6.2f}"
        )

    # Grid: rows = [control, *combos], cols = pre-swap / +6 / +12 / last.
    frame_cols = [SWAP_AT * 8 - 8, SWAP_AT * 8 + 45, SWAP_AT * 8 + 93, total_frames - 1]
    row_names = ["control", *(name for name, *_ in COMBOS)]
    rows = []
    for name in row_names:
        arr = ((videos[name].numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
        rows.append(
            np.concatenate([arr[c].transpose(1, 2, 0) for c in frame_cols], axis=1)
        )
    grid = np.concatenate(rows, axis=0)[::2, ::2]
    media.write_image(OUT_DIR / "grid.png", grid)

    meta = {
        "uuid": UUID,
        "clip_prompt": clip_prompt,
        "n_chunks": N_CHUNKS,
        "swap_at": SWAP_AT,
        "seed": SEED,
        "grid_row_order": row_names,
        "grid_frame_cols": frame_cols,
        "combos": report,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(meta, indent=2))
    print(f"done -> {OUT_DIR}/ (grid rows: {', '.join(row_names)})")


if __name__ == "__main__":
    main()
