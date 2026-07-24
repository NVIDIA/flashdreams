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

"""Tiled-HDMap loop rollouts: drifted-vs-clean pair clips for Omnidreams (v2).

The Omnidreams action grammar is the HDMap conditioning video, so the
strafe-loop trick ports as *HDMap tiling*: the rollout conditioning is
``clip[:5] + tile(clip[5 : 5 + 8 * LAP_CHUNKS], LAPS)`` — every lap's chunk
``i`` consumes pixel-identical HDMap frames, so a late (drifted) chunk's
KV-window content can be swapped for its lap-aligned clean counterparts at
the same conditioning and the same RoPE positions. The lap boundary is a
conditioning teleport; probes/training only use chunks whose surviving
window lies inside one lap (gate) or map counterfactuals by lap position
(training). Rollouts run ~21.5 s (81 chunks) because this host's residual
drift shows late.

v2 (régime fix 2026-07-22): pairs v1 seeded rollouts with the local
benchmarking corpus, whose "first frames" are HDMap renders — the rollouts
stayed in a render-adjacent régime and collapsed to one scene. v2 sources
the gated HF sample set (``nvidia/omni-dreams-samples``): 32 clips with
REAL dashcam first frames, per-clip prompts, and 80 s authentic HDMaps.
Requires an authenticated HF token.

v3 (v3 layout, 2026-07-23): run with ``LAP_MIX=4,5,6`` — mixed lap
lengths rotated per clip break the fixed 40-frame revisit period that let
the v2-trained corrector learn a repeat prior (see README known issues). Default
``LAP_MIX=5`` reproduces the v2 layout exactly.

Run from the flashdreams repo root::

    HF_TOKEN=$(cat ~/.cache/huggingface/token) \
        .venv/bin/python integrations/omnidreams/drift_correction/build_pairs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline, capture_rollout, save_clip
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
    _load_video,
)

from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    write_video_tensor,
)

## Pair-build configuration

OUT_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/omnidreams/drift_correction/outputs/pairs_v2"
    )
)
"""Clip files consumed by ``gate_faithful.py`` / ``train_v1.py``."""

N_CLIPS = int(os.environ.get("N_CLIPS", "6"))
"""Sample clips to roll out (first ``N_CLIPS`` of the dataset, sorted)."""

LAP_MIX = tuple(int(x) for x in os.environ.get("LAP_MIX", "5").split(","))
"""AR chunks per lap, rotated per clip (clip ``c`` gets
``LAP_MIX[c % len(LAP_MIX)]``). Pairs v1/v2 used a single ``5`` (40 decoded
frames = 1.33 s). Pairs v3 uses ``4,5,6`` (32/40/48 frames): a FIXED lap
period lets the corrector learn "content recurs at lag 40" as a shortcut —
the HY v2 repeat-prior failure, reproduced here on the scen6 deploy sweep
(hallucinated repeated trailers + suppressed conditioned crosswalk;
see README known issues) — and mixing the lengths breaks the fixed
revisit period."""

TARGET_LAP_CHUNKS = 80
"""Tiled chunks per rollout: laps per clip = ``round(80 / lap_chunks)``,
so every clip runs ~81 chunks = ~21.5 s regardless of its lap length."""


def clip_layout(c: int) -> tuple[int, int, int]:
    """Per-clip ``(lap_chunks, laps, num_chunk)`` under the LAP_MIX rotation.

    ``num_chunk`` = chunk 0 (image-anchored, 5 frames) + the tiled laps.
    """
    lap_chunks = LAP_MIX[c % len(LAP_MIX)]
    laps = round(TARGET_LAP_CHUNKS / lap_chunks)
    return lap_chunks, laps, 1 + lap_chunks * laps


NOISE_SEED = int(os.environ.get("NOISE_SEED", "5042"))
"""Diffusion RNG seed per rollout (offset by clip index)."""

CORRECTOR_LORA = os.environ.get("CORRECTOR_LORA", "")
"""Optional v1 checkpoint; when set, rollouts run with the corrector at the
deploy dial (per-step ``alpha*(t) x 0.5``) so the DAgger pool reflects the
states the deployed corrector actually visits."""


def _list_sample_uuids(n: int) -> list[str]:
    """Return the first ``n`` single-view sample UUIDs, alphabetically.

    ``SAMPLE_UUIDS`` (comma list) bypasses the HF listing API — the shared
    IP rate limit killed three pipeline stages on 2026-07-24; every consumer
    already has the files in the local HF cache, only this listing needed
    the network.
    """
    if os.environ.get("SAMPLE_UUIDS"):
        uuids = os.environ["SAMPLE_UUIDS"].split(",")
        assert len(uuids) >= n, f"SAMPLE_UUIDS lists only {len(uuids)} clips"
        return uuids[:n]
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFolder

    entries = HfApi().list_repo_tree(
        repo_id="nvidia/omni-dreams-samples",
        repo_type="dataset",
        path_in_repo="data/single_view",
        recursive=False,
    )
    uuids = sorted(
        e.path.rsplit("/", 1)[-1] for e in entries if isinstance(e, RepoFolder)
    )
    assert len(uuids) >= n, f"dataset lists only {len(uuids)} single-view clips"
    return uuids[:n]


def _sample_files(uuid: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Cache-first ``((hdmap,), (first_frame,))`` resolution for a sample.

    The runner's ``_ensure_hf_single_view_example_data_synced`` lists the HF
    repo per clip even when every file is already local; the shared IP rate
    limit killed three pipeline stages on 2026-07-24. Fall back to the
    network path only on a cache miss.
    """
    root = (
        Path.home()
        / ".cache/huggingface/hub/datasets--nvidia--omni-dreams-samples/snapshots"
    )
    hits = sorted(root.glob(f"*/data/single_view/{uuid}/*_hdmap.mp4"))
    for h in hits:
        frame = h.parent / "first_frame.png"
        if frame.exists():
            return (h,), (frame,)
    return _ensure_hf_single_view_example_data_synced(uuid)


def _clip_prompt(uuid: str) -> str:
    """Fetch the clip's own prompt from the dataset (cache-first)."""
    root = (
        Path.home()
        / ".cache/huggingface/hub/datasets--nvidia--omni-dreams-samples/snapshots"
    )
    hits = sorted(root.glob(f"*/data/single_view/{uuid}/prompt.txt"))
    if hits:
        return hits[0].read_text().strip()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="nvidia/omni-dreams-samples",
        repo_type="dataset",
        filename=f"data/single_view/{uuid}/prompt.txt",
    )
    return Path(path).read_text().strip()


def tile_hdmap(hdmap: torch.Tensor, lap_chunks: int, laps: int) -> torch.Tensor:
    """Tile a ``[T, C, H, W]`` HDMap clip into the loop conditioning.

    Layout: the first 5 frames (chunk 0) then ``laps`` copies of the next
    ``8 * lap_chunks`` frames, so chunks ``1 + j * lap_chunks + i`` consume
    identical pixels for every lap ``j``.
    """
    lap_frames = 8 * lap_chunks
    assert hdmap.shape[0] >= 5 + lap_frames, (
        f"clip has {hdmap.shape[0]} frames; need {5 + lap_frames}."
    )
    lap = hdmap[5 : 5 + lap_frames]
    return torch.cat([hdmap[:5], lap.repeat(laps, 1, 1, 1)], dim=0)


def main() -> None:
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16

    # Load every clip's inputs BEFORE any model/rollout work: video decode
    # forks ffmpeg, which fails silently (empty read) once this process has
    # grown to rollout size -- observed 3x on this box, always on the
    # second decode of a job.
    inputs: list[tuple[str, str, torch.Tensor, torch.Tensor] | None] = []
    for c, uuid in enumerate(_list_sample_uuids(N_CLIPS)):
        if (OUT_DIR / f"clip_{c:02d}.pt").exists():
            inputs.append(None)
            continue
        (hdmap_path,), (frame_path,) = _sample_files(uuid)
        hdmap = _load_video(
            hdmap_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",  # ty: ignore[invalid-argument-type]
            dtype=dtype,
        )
        lap_chunks, laps, _ = clip_layout(c)
        hdmap = tile_hdmap(hdmap, lap_chunks, laps)[None, None]  # [1, 1, T, C, H, W]
        first = load_first_frame_tensor(
            frame_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",  # ty: ignore[invalid-argument-type]
            dtype=dtype,
        )[None, :, None]  # [1, V=1, 1, C, H, W]
        inputs.append((uuid, _clip_prompt(uuid), hdmap, first))
        print(f"loaded inputs for clip {c} ({uuid})", flush=True)

    pipe = build_pipeline(with_oneshot_encoders=True)
    device = pipe.device

    if CORRECTOR_LORA:
        from _lora import apply_lora, load_lora, unwrap_compiled
        from eval_rollouts import install_alpha_gate

        transformer = pipe.diffusion_model.transformer
        network = unwrap_compiled(transformer.network)
        apply_lora(network)
        load_lora(network, CORRECTOR_LORA)
        install_alpha_gate(transformer, network, {"gain": ("gate", 0.5)})
        print(f"corrector active at gate x 0.5: {CORRECTOR_LORA}", flush=True)

    for c, item in enumerate(inputs):
        out = OUT_DIR / f"clip_{c:02d}.pt"
        if item is None:
            print(f"SKIP clip {c}: {out} exists", flush=True)
            continue
        uuid, prompt, hdmap, first = item
        lap_chunks, laps, num_chunk = clip_layout(c)
        hdmap = hdmap.to(device)
        first = first.to(device)

        embeddings = pipe.precompute_embeddings(text=[[prompt]], image=first)
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=embeddings["text_embeddings"],  # ty: ignore[invalid-argument-type]
            image_embeddings=embeddings["image_embeddings"],  # ty: ignore[invalid-argument-type]
        )
        snaps, video = capture_rollout(
            pipe,
            cache,
            hdmap_video=hdmap,
            num_chunk=num_chunk,
            noise_seed=NOISE_SEED + c,
        )
        save_clip(
            out,
            snaps=snaps,
            embeddings=embeddings,
            meta={
                "uuid": uuid,
                "prompt": prompt,
                "num_chunk": num_chunk,
                "lap_chunks": lap_chunks,
                "laps": laps,
                "noise_seed": NOISE_SEED + c,
            },
        )
        mp4 = OUT_DIR / f"clip_{c:02d}_loop.mp4"
        write_video_tensor(video[0, 0].permute(0, 2, 3, 1), mp4, fps=30, layout="thwc")
        print(f"clip {c} ({uuid}): saved {out} + {mp4}", flush=True)

    print("PAIRS-DONE", flush=True)


if __name__ == "__main__":
    main()
