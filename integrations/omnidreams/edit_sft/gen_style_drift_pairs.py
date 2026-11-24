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

"""Style-drift pair generation for the style-drift corrector (GPU, one shot).

The style-skin LoRA (``train_style_sft.py``) compounds through the KV
history on long holds: the first ~1-4 chunks after a swap are the clean
styled manifold, but by +8..+20 chunks the re-styled history has blurred
the world out (``issues_and_fixes.md`` Issue 1). This script rolls the
branch corpus the corrector trains on:

- The style LoRA is deployed exactly as in serving: pre-merged
  :class:`~omnidreams._edit_lora.TextEditLoRA` weights plus a ``use_lora``
  edit window held open to the end of the rollout (the
  ``scripts/smoke_text_edit.py`` ``EDIT_LORA`` + swap machinery).
- Per source clip (``generate_sources.py`` manifest) and per style slug,
  one branch per swap offset in :data:`SWAP_OFFSETS`, all under the SAME
  seed. The swap draws no noise on this host, so branches are bit-equal
  before their swap (asserted) and diverge only in style-hold depth.
- Per chunk, the patchified x0 latent is captured exactly as the AR cache
  consumed it (the ``drift_correction/_host.py:capture_rollout`` fields),
  plus the patchified HDMap conditioning once per clip.

Offsets spaced 4 apart make every late-window chunk of one branch (swap
8-20 chunks in the past — drifted) share its absolute index with an
early-window chunk of another branch (swap 1-4 chunks in the past — the
clean styled manifold). ``train_style_corrector.py`` builds counterfactual
probes from those (drifted, clean) branch pairs.

Outputs (under ``OUT_DIR``):

- ``<uuid>_base.pt``: ``{"latents": [n_chunks x [1, 1, L, D]],
  "hdmaps": [...], "n_chunks", "seed"}`` — the unswapped rollout.
- ``<uuid>__<slug>.pt``: ``{"branches": {offset: [latents for chunks
  offset..n_chunks-1]}, "swap_offsets", "n_chunks", "seed",
  "style_lora"}`` — pre-swap chunks live in the base file.

Env knobs: ``STYLE_LORA``, ``STYLES``, ``SWAP_OFFSETS``, ``N_CLIPS``,
``N_CHUNKS``, ``SEED``, ``OUT_DIR``, ``SAVE_MP4``.

Run from the flashdreams repo root (after ``precompute_style.py``; needs
no one-shot encoders — prompts come from ``style_embeddings.pt``)::

    .venv/bin/python integrations/omnidreams/edit_sft/gen_style_drift_pairs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from _host import build_pipeline
from omnidreams._edit_lora import TextEditLoRA
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_hf_single_view_example_data_synced,
)
from style_prompts import STYLE_PROMPTS, clip_key
from torch import Tensor

from flashdreams.infra.runner_io import load_video_tensor, write_video_tensor

## Configuration

BASE = Path("integrations/omnidreams/edit_sft")
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE / "outputs/style_drift_pairs")))
SRC_DIR = BASE / "outputs" / "sources"

STYLE_LORA = Path(
    os.environ.get("STYLE_LORA", str(BASE / "outputs/lora_style_step1600.pt"))
)
"""Style-skin checkpoint to deploy (``train_style_sft.py`` format)."""

STYLES = tuple(
    s for s in os.environ.get("STYLES", "arcade_racer,comic_ink").split(",") if s
)
"""Style slugs to roll branches for (the full-strength training styles)."""

SWAP_OFFSETS = tuple(
    int(x) for x in os.environ.get("SWAP_OFFSETS", "4,8,12,16,20,24").split(",")
)
"""Swap chunks, spaced 4 apart so every probe chunk has both a drifted
(+8..+20) and a clean (+1..+4) branch counterpart at the same index."""

N_CLIPS = int(os.environ.get("N_CLIPS", "0"))
"""Source clips to roll (manifest order); 0 = all."""

SEED = int(os.environ.get("SEED", "42"))
SAVE_MP4 = os.environ.get("SAVE_MP4", "0") == "1"
"""Also write the decoded branch rollouts for eyeballing the drift."""


def _sample_files(uuid: str) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Cache-first ``((hdmap,), (first_frame,))`` resolution for a sample.

    The runner's ``_ensure_hf_single_view_example_data_synced`` lists the
    HF repo per clip even when every file is already local; the shared IP
    rate limit killed three pipeline stages on 2026-07-24. Fall back to
    the network path only on a cache miss.
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


@torch.no_grad()
def rollout(
    pipe: OmnidreamsPipeline,
    *,
    text_emb: Tensor,
    image_emb: Tensor,
    style_emb: Tensor | None,
    hdmap: Tensor,
    n_chunks: int,
    swap_at: int | None,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    """Roll ``n_chunks`` with an optional held-open style window at ``swap_at``.

    Args:
        pipe: Pipeline with the pre-merged style ``TextEditLoRA`` attached.
        text_emb: ``[1, 1, L, D]`` clip-prompt embeddings.
        image_emb: ``[1, 1, 1, Cl, Hl, Wl]`` first-frame embeddings.
        style_emb: ``[1, 1, L, D]`` style-prompt embeddings (``swap_at``
            branches only).
        hdmap: ``[1, 1, T, 3, H, W]`` HDMap pixels covering ``n_chunks``.
        n_chunks: AR chunks to roll.
        swap_at: Chunk index of the style swap; ``None`` = base branch.

    Returns:
        ``(latents, hdmaps, video)`` — per-chunk patchified x0 and HDMap
        conditioning exactly as the AR cache consumed them (CPU), plus the
        decoded rollout ``[T, 3, H, W]`` on CPU.
    """
    device = pipe.device
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)
    cache = pipe.initialize_cache_from_embeddings(
        text_embeddings=text_emb, image_embeddings=image_emb
    )
    latents: list[Tensor] = []
    hdmaps: list[Tensor] = []
    chunks: list[Tensor] = []
    start = 0
    for ar_idx in range(n_chunks):
        if swap_at is not None and ar_idx == swap_at:
            # The deploy swap: guidance_scale is unused on the use_lora
            # path but must be != 1.0 to open the window; the countdown
            # covers every remaining chunk (held-open long-hold regime).
            assert style_emb is not None
            pipe.replace_text_from_embeddings(
                cache,
                style_emb,
                guidance_scale=2.0,
                guidance_chunks=n_chunks - swap_at,
            )
            g = cache.transformer_cache.text_edit_guidance
            assert g is not None and g.use_lora, (
                "style window must run the pre-merged LoRA path "
                "(is the TextEditLoRA attached?)"
            )
        num_frames = pipe.get_num_frames(ar_idx)
        end = start + num_frames
        assert end <= hdmap.shape[2], f"hdmap too short at chunk {ar_idx}"
        chunk = pipe.generate(ar_idx, cache, hdmap=hdmap[:, :, start:end])
        fs = cache.final_state
        assert fs is not None and isinstance(fs.input, Tensor)
        latents.append(fs.clean_latent.detach().to("cpu"))
        hdmaps.append(fs.input.detach().to("cpu"))
        pipe.finalize(ar_idx, cache)
        chunks.append(chunk[0, 0].float().cpu())
        start = end
    del cache
    torch.cuda.empty_cache()
    return latents, hdmaps, torch.cat(chunks, dim=0)


def main() -> None:
    """Roll the base + per-(style, offset) branch corpus and save it."""
    manifest = json.loads((SRC_DIR / "manifest.json").read_text())
    uuids = [e["uuid"] for e in manifest]
    if N_CLIPS:
        uuids = uuids[:N_CLIPS]
    n_chunks = int(os.environ.get("N_CHUNKS", str(manifest[0]["n_chunks"])))
    total_frames = 5 + (n_chunks - 1) * 8
    assert max(SWAP_OFFSETS) < n_chunks, (
        f"swap offset {max(SWAP_OFFSETS)} outside the {n_chunks}-chunk rollout"
    )

    prompt_emb = torch.load(
        BASE / "outputs/style_embeddings.pt", map_location="cpu", weights_only=False
    )
    assets = torch.load(
        BASE / "outputs/style_clip_assets.pt", map_location="cpu", weights_only=False
    )
    for slug in STYLES:
        assert slug in STYLE_PROMPTS and slug in prompt_emb, (
            f"style {slug!r} missing from STYLE_PROMPTS / style_embeddings.pt"
        )

    pipe = build_pipeline(with_oneshot_encoders=False)
    transformer = pipe.diffusion_model.transformer
    edit_lora = TextEditLoRA(transformer.network, STYLE_LORA)
    transformer.set_text_edit_lora(edit_lora)
    print(f"deployed {edit_lora.describe()} from {STYLE_LORA}", flush=True)
    print(
        f"{len(uuids)} clips x {len(STYLES)} styles x {len(SWAP_OFFSETS)} "
        f"offsets, {n_chunks} chunks each",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for uuid in uuids:
        base_path = OUT_DIR / f"{uuid}_base.pt"
        todo = [s for s in STYLES if not (OUT_DIR / f"{uuid}__{s}.pt").exists()]
        if base_path.exists() and not todo:
            print(f"skip {uuid} (exists)", flush=True)
            continue

        (hdmap_path,), _ = _sample_files(uuid)
        hdmap = load_video_tensor(
            hdmap_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device=pipe.device,
            dtype=torch.bfloat16,
        )
        assert hdmap.shape[0] >= total_frames, (
            f"{uuid}: {hdmap.shape[0]} HDMap frames < {total_frames}"
        )
        hdmap = hdmap[:total_frames][None, None]
        text_emb = prompt_emb[clip_key(uuid)]
        image_emb = assets["image_embeddings"][uuid]

        if base_path.exists():
            base = torch.load(base_path, map_location="cpu", weights_only=False)[
                "latents"
            ]
        else:
            base, hdmaps, _ = rollout(
                pipe,
                text_emb=text_emb,
                image_emb=image_emb,
                style_emb=None,
                hdmap=hdmap,
                n_chunks=n_chunks,
                swap_at=None,
            )
            torch.save(
                {
                    "uuid": uuid,
                    "latents": base,
                    "hdmaps": hdmaps,
                    "n_chunks": n_chunks,
                    "seed": SEED,
                },
                base_path,
            )
            print(f"rolled {uuid} base ({n_chunks} chunks)", flush=True)

        for slug in todo:
            branches: dict[int, list[Tensor]] = {}
            for s in SWAP_OFFSETS:
                latents, _, video = rollout(
                    pipe,
                    text_emb=text_emb,
                    image_emb=image_emb,
                    style_emb=prompt_emb[slug],
                    hdmap=hdmap,
                    n_chunks=n_chunks,
                    swap_at=s,
                )
                # The swap draws no noise -> branches are bit-equal to the
                # base rollout before their swap (verified host property);
                # only the post-swap chunks are stored.
                assert torch.equal(latents[0], base[0]) and torch.equal(
                    latents[s - 1], base[s - 1]
                ), f"{uuid}__{slug} s={s}: pre-swap chunks diverged from base"
                branches[s] = latents[s:]
                if SAVE_MP4:
                    write_video_tensor(
                        video,
                        OUT_DIR / f"{uuid[:8]}__{slug}_s{s:02d}.mp4",
                        fps=30,
                        layout="tchw",
                    )
                print(f"rolled {uuid}__{slug} s={s}", flush=True)
            torch.save(
                {
                    "uuid": uuid,
                    "slug": slug,
                    "branches": branches,
                    "swap_offsets": list(SWAP_OFFSETS),
                    "n_chunks": n_chunks,
                    "seed": SEED,
                    "style_lora": str(STYLE_LORA),
                },
                OUT_DIR / f"{uuid}__{slug}.pt",
            )

    print(
        f"STYLE-DRIFT-PAIRS-DONE | {len(uuids)} clips x {len(STYLES)} styles "
        f"x {len(SWAP_OFFSETS)} offsets -> {OUT_DIR}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
