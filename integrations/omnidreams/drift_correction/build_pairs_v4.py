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

"""Non-looping drift pairs with re-anchored fork counterparts (pairs v4).

Every loop-pair recipe (v1-v3) revisits its own content — the corrector
learns a *loop prior* (self-copying / novel-content suppression; project
history). v4 removes revisits entirely:

- **Conditioning**: each rollout consumes its clip's OWN continuous 645-frame
  HDMap (the HF samples ship ~80 s) — no tiling, no stitching, no teleports;
  content never recurs by construction.
- **Clean counterpart**: every :data:`SEG_CHUNKS` chunks the build forks a
  fresh IMAGE-ANCHORED rollout B_s seeded by re-encoding the drifted rollout
  A's own decoded frame at the fork, rolled over the same upcoming HDMap
  frames. B is content-matched at the fork and early-horizon-clean (drift
  shows late; anchored chunks regenerate crisp structure), so the pair
  isolates the *structure/sharpness* share of drift. Known limitation
  (pre-stated): the anchor inherits A's low-frequency color drift, so that
  share is NOT in the target — the alpha*(t) gate on these pairs
  (``gate_faithful.py`` with ``PAIR_SCHEME=fork``) must clear its bar before
  any training (see README).

Frame alignment: fork s anchors at decoded frame ``f0 - 5`` and consumes
HDMap ``[f0-5, f0+160)`` — its 5-frame chunk 0 covers the tail of A's chunk
``c_s - 1``, so B chunk ``1 + m`` consumes exactly A chunk ``c_s + m``'s
8-frame window (no conditioning lag).

Per-clip output adds ``fork_starts`` / ``fork_latents`` (B chunks 1..20 per
fork; chunk 0 is the anchor and is never a counterpart) to the pairs-v2/v3
format, so v3-format consumers keep working (they ignore the extra keys).

Run from the flashdreams repo root::

    HF_TOKEN=... .venv/bin/python integrations/omnidreams/drift_correction/build_pairs_v4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline, capture_rollout, save_clip
from build_pairs import _clip_prompt, _list_sample_uuids, _sample_files
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_video,
)

from flashdreams.infra.runner_io import (
    load_first_frame_tensor,
    write_video_tensor,
)

## Pair-build configuration

OUT_DIR = Path(
    os.environ.get(
        "PAIRS_DIR", "integrations/omnidreams/drift_correction/outputs/pairs_v4"
    )
)

N_CLIPS = int(os.environ.get("N_CLIPS", "6"))
"""Sample clips (first N of the dataset, sorted) — same roster as v2/v3."""

NUM_CHUNK = int(os.environ.get("NUM_CHUNK", "81"))
"""Drifted rollout A: chunk 0 + 80 chunks = 645 frames = 21.5 s."""

SEG_CHUNKS = int(os.environ.get("SEG_CHUNKS", "20"))
"""Fork cadence: B_s counterparts cover chunks ``c_s .. c_s + 19`` for
``c_s in {1, 21, 41, 61}`` (defaults; env overrides are for smoke runs)."""

FORK_STARTS = tuple(range(1, NUM_CHUNK, SEG_CHUNKS))

NOISE_SEED = int(os.environ.get("NOISE_SEED", "5042"))

CORRECTOR_LORA = os.environ.get("CORRECTOR_LORA", "")
"""Optional checkpoint for DAgger rounds: rollouts (A and forks) run with
the corrector at the deploy dial (``alpha*(t) x 0.5``)."""


def main() -> None:
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16

    # Load all inputs before any model work (ffmpeg fails silently once the
    # process reaches rollout size on this box).
    n_frames = 5 + (NUM_CHUNK - 1) * 8
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
        assert hdmap.shape[0] >= n_frames, (uuid, hdmap.shape)
        first = load_first_frame_tensor(
            frame_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",  # ty: ignore[invalid-argument-type]
            dtype=dtype,
        )[None, :, None]
        inputs.append((uuid, _clip_prompt(uuid), hdmap[:n_frames][None, None], first))
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
        hdmap = hdmap.to(device)
        first = first.to(device)

        # Drifted rollout A over the clip's own continuous HDMap.
        embeddings = pipe.precompute_embeddings(text=[[prompt]], image=first)
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=embeddings["text_embeddings"],  # ty: ignore[invalid-argument-type]
            image_embeddings=embeddings["image_embeddings"],  # ty: ignore[invalid-argument-type]
        )
        snaps, video = capture_rollout(
            pipe,
            cache,
            hdmap_video=hdmap,
            num_chunk=NUM_CHUNK,
            noise_seed=NOISE_SEED + c,
        )

        # Re-anchored fork counterparts B_s.
        fork_latents: list[list[torch.Tensor]] = []
        for s, c_s in enumerate(FORK_STARTS):
            f0 = 5 + 8 * (c_s - 1)
            anchor = video[0, 0, f0 - 5].to(device, dtype)[None, None, None]
            emb_s = pipe.precompute_embeddings(text=[[prompt]], image=anchor)
            cache_s = pipe.initialize_cache_from_embeddings(
                text_embeddings=emb_s["text_embeddings"],  # ty: ignore[invalid-argument-type]
                image_embeddings=emb_s["image_embeddings"],  # ty: ignore[invalid-argument-type]
            )
            snaps_s, video_s = capture_rollout(
                pipe,
                cache_s,
                hdmap_video=hdmap[:, :, f0 - 5 : f0 - 5 + 5 + 8 * SEG_CHUNKS],
                num_chunk=1 + SEG_CHUNKS,
                noise_seed=NOISE_SEED + 100 * (s + 1) + c,
            )
            fork_latents.append([sn.clean_latent.detach().cpu() for sn in snaps_s[1:]])
            if c == 0:
                mp4 = OUT_DIR / f"clip_{c:02d}_fork{s}.mp4"
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                write_video_tensor(
                    video_s[0, 0].permute(0, 2, 3, 1), mp4, fps=30, layout="thwc"
                )
            del snaps_s, video_s, cache_s

        save_clip(
            out,
            snaps=snaps,
            embeddings=embeddings,
            meta={
                "uuid": uuid,
                "prompt": prompt,
                "num_chunk": NUM_CHUNK,
                # lap fields kept for shape-compat; fork scheme ignores them.
                "lap_chunks": SEG_CHUNKS,
                "laps": (NUM_CHUNK - 1) // SEG_CHUNKS,
                "noise_seed": NOISE_SEED + c,
                "pair_scheme": "fork",
                "fork_starts": list(FORK_STARTS),
                "fork_latents": fork_latents,
            },
        )
        mp4 = OUT_DIR / f"clip_{c:02d}_drifted.mp4"
        write_video_tensor(video[0, 0].permute(0, 2, 3, 1), mp4, fps=30, layout="thwc")
        print(f"clip {c} ({uuid}): saved {out} + {mp4}", flush=True)


if __name__ == "__main__":
    main()
