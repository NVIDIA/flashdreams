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

"""Precompute the style-SFT embeddings and latents (one shot).

Three phases, so ``train_style_sft.py`` never loads the ~14 GB text
encoder or touches ffmpeg (the ``guidance_distill/precompute_embeddings``
pattern, extended with video encoding):

A. Text + first-frame embeddings — every style-slug prompt
   (:data:`~style_prompts.STYLE_PROMPTS`) and every source clip's own
   prompt / first frame through the resident one-shot encoders.
B. Per-clip HDMap latents — the clip's HDMap video stream-encoded with the
   pipeline's per-AR-step Wan-VAE encoder in the AR chunk schedule (5
   frames -> 2 latent frames, then 8 -> 2), patchified per chunk.
C. Styled-target latents — every ``style_pairs/<uuid>__<slug>.mp4``
   through the SAME encoder/schedule, so the flow-matching targets live in
   the exact latent space the pipeline itself maps RGB into (the encoder
   checkpoint is the one the first-frame I2V injection uses).

All styled pairs found on disk are encoded; the VLM gate
(``style_pairs/filter_report.json``, possibly still being written) is
applied at training time, not here.

Outputs (under ``edit_sft/outputs/``):

- ``style_embeddings.pt``: ``{slug | "clip:<uuid>": [1, 1, L, D] bf16}``.
- ``style_clip_assets.pt``: ``{"uuids", "prompts",
  "image_embeddings": {uuid: [1, 1, 1, Cl, Hl, Wl] bf16}}``.
- ``latents/<uuid>_hdmap_latents.pt``: ``{"hdmaps": [n_chunks x
  [1, 1, L, D] bf16]}`` (patchified, ``transformer.latent_shape``).
- ``latents/<uuid>__<slug>_latents.pt``: ``{"latents": [...]}``, same
  layout.

Latent files are skipped when present, so the script is resumable — the
frame-count assert catches the silent-empty ffmpeg reads observed on this
box (``build_pairs.py`` note; the one-shot encoders are released before
the first decode to shrink the process).

Run from the flashdreams repo root (~20 GB VRAM during phase A)::

    .venv/bin/python integrations/omnidreams/edit_sft/precompute_style.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drift_correction"))

import torch
from _host import build_pipeline
from build_pairs import _sample_files
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_first_frame,
    _load_video,
)
from style_prompts import STYLE_PROMPTS, clip_key
from torch import Tensor

## Configuration

BASE = Path("integrations/omnidreams/edit_sft")
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE / "outputs")))
SRC_DIR = OUT_DIR / "sources"
STYLE_DIR = Path(os.environ.get("STYLE_DIR", str(OUT_DIR / "style_pairs")))
"""Styled-target videos; point at ``style_pairs_teacher`` for the
teacher-regenerated corpus (``gen_teacher_styled_targets.py``)."""
LAT_DIR = Path(os.environ.get("LAT_DIR", str(OUT_DIR / "latents")))
"""Latent output dir; use a separate dir per styled corpus (don't clobber)."""


def _styled_videos(known_uuids: set[str]) -> list[tuple[str, str, Path]]:
    """``(uuid, slug, path)`` for every recognized styled pair on disk."""
    out: list[tuple[str, str, Path]] = []
    for path in sorted(STYLE_DIR.glob("*__*.mp4")):
        uuid, slug = path.stem.split("__", 1)
        if uuid not in known_uuids:
            print(f"skip {path.name}: uuid not in the sources manifest", flush=True)
            continue
        if slug not in STYLE_PROMPTS:
            print(f"skip {path.name}: slug {slug!r} not in STYLE_PROMPTS", flush=True)
            continue
        out.append((uuid, slug, path))
    return out


def encode_video_chunks(
    pipe: OmnidreamsPipeline, video: Tensor, n_chunks: int
) -> list[Tensor]:
    """Stream-encode an RGB video in the AR chunk schedule.

    Args:
        pipe: Pipeline whose per-AR-step Wan-VAE encoder does the work
            (fresh streaming cache per video).
        video: ``[T, 3, H, W]`` pixels in ``[-1, 1]`` on CPU; ``T`` must
            cover ``n_chunks`` chunks (5 + 8 * (n_chunks - 1) frames).
        n_chunks: AR chunks to encode.

    Returns:
        Per-chunk patchified latents (``transformer.latent_shape`` layout,
        the space ``generate_sources.py`` saved the source latents in),
        bf16 on CPU.
    """
    transformer = pipe.diffusion_model.transformer
    assert pipe.encoder is not None
    enc_cache = pipe.encoder.initialize_autoregressive_cache()
    chunks: list[Tensor] = []
    start = 0
    for ar_idx in range(n_chunks):
        num_frames = pipe.get_num_frames(ar_idx)
        pixels = video[start : start + num_frames][None, None].to(pipe.device)
        z = pipe.encoder(input=pixels, autoregressive_index=ar_idx, cache=enc_cache)
        chunks.append(transformer.patchify_and_maybe_split_cp(z).cpu())
        start += num_frames
    return chunks


@torch.no_grad()
def main() -> None:
    """Encode embeddings (phase A), HDMap latents (B), styled latents (C)."""
    dtype = torch.bfloat16
    manifest = json.loads((SRC_DIR / "manifest.json").read_text())
    uuids: list[str] = [e["uuid"] for e in manifest]
    prompts: dict[str, str] = {e["uuid"]: e["prompt"] for e in manifest}
    n_chunks = int(manifest[0]["n_chunks"])
    total_frames = 5 + (n_chunks - 1) * 8
    styled = _styled_videos(set(uuids))
    assert styled, f"no recognized styled pairs under {STYLE_DIR}"

    # Phase A is skippable when the embedding files already cover every
    # needed key (reruns against a new STYLE_DIR/LAT_DIR corpus): the
    # ~14 GB text encoder never loads.
    emb_path = OUT_DIR / "style_embeddings.pt"
    assets_path = OUT_DIR / "style_clip_assets.pt"
    skip_phase_a = False
    if emb_path.exists() and assets_path.exists():
        prev_emb = torch.load(emb_path, map_location="cpu", weights_only=False)
        prev_assets = torch.load(assets_path, map_location="cpu", weights_only=False)
        skip_phase_a = all(slug in prev_emb for _, slug, _ in styled) and all(
            clip_key(u) in prev_emb and u in prev_assets["image_embeddings"]
            for u in uuids
        )
        del prev_emb, prev_assets

    if skip_phase_a:
        print("phase A: existing embedding files cover all keys — skipped", flush=True)
        pipe = build_pipeline(with_oneshot_encoders=False)
    else:
        # First frames are PNGs (cv2, no ffmpeg fork) — safe to load lazily,
        # but front-load them anyway so phase A never touches the filesystem
        # mid-encoding.
        firsts: dict[str, Tensor] = {}
        for uuid in uuids:
            _, (frame_path,) = _sample_files(uuid)
            firsts[uuid] = _load_first_frame(
                frame_path,
                pixel_height=DEFAULT_VIDEO_HEIGHT,
                pixel_width=DEFAULT_VIDEO_WIDTH,
                device="cpu",
                dtype=dtype,
            )[None, :, None]  # [1, V=1, 1, C, H, W]

        pipe = build_pipeline(with_oneshot_encoders=True)
        device = pipe.device
        assert pipe.text_encoder is not None  # with_oneshot_encoders=True

        # Phase A: style + clip prompt embeddings and first-frame latents.
        emb_by_text: dict[str, Tensor] = {}
        prompt_embeddings: dict[str, Tensor] = {}
        for slug in sorted({slug for _, slug, _ in styled}):
            text = STYLE_PROMPTS[slug]
            if text not in emb_by_text:  # _v2 slugs alias their base prompt
                emb = torch.stack([pipe.text_encoder([text])], dim=0)  # [1, 1, L, D]
                emb_by_text[text] = emb.to("cpu", dtype)
            prompt_embeddings[slug] = emb_by_text[text]
            print(
                f"encoded style prompt {slug}: {tuple(prompt_embeddings[slug].shape)}",
                flush=True,
            )
        image_embeddings: dict[str, Tensor] = {}
        for uuid in uuids:
            emb = pipe.precompute_embeddings(
                text=[[prompts[uuid]]], image=firsts[uuid].to(device)
            )
            text_emb = emb["text_embeddings"]
            image_emb = emb["image_embeddings"]
            assert text_emb is not None and image_emb is not None
            prompt_embeddings[clip_key(uuid)] = text_emb.to("cpu", dtype)
            image_embeddings[uuid] = image_emb.to("cpu", dtype)
            print(f"encoded clip {uuid}", flush=True)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(prompt_embeddings, emb_path)
        torch.save(
            {"uuids": uuids, "prompts": prompts, "image_embeddings": image_embeddings},
            assets_path,
        )
        # Shrink the process before the ffmpeg decodes below (fork hazard).
        pipe.release_oneshot_encoders()

    def encode_to(
        path: Path, video_path: Path, key: str, allow_short: bool = False
    ) -> None:
        video = _load_video(
            video_path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device="cpu",
            dtype=dtype,
        )
        if allow_short and video.shape[0] < total_frames:
            # Teacher-regenerated targets cover one bidirectional chunk48
            # window (189 frames = student chunks 0..23); encode the full
            # chunks available. Still assert non-empty (silent ffmpeg reads).
            avail = (int(video.shape[0]) - 5) // 8 + 1
            assert avail >= 1, (
                f"{video_path.name}: {video.shape[0]} frames < one chunk "
                "(silent-empty ffmpeg read? re-run to resume)."
            )
        else:
            assert video.shape[0] >= total_frames, (
                f"{video_path.name}: {video.shape[0]} frames < {total_frames} "
                "(short clip, or a silent-empty ffmpeg read — re-run to resume)."
            )
            avail = n_chunks
        frames_used = 5 + (avail - 1) * 8
        chunks = encode_video_chunks(pipe, video[:frames_used], avail)
        torch.save({key: chunks, "n_chunks": avail}, path)
        print(f"encoded {path.name}: {avail} chunks", flush=True)

    LAT_DIR.mkdir(parents=True, exist_ok=True)
    # Phase B: per-clip HDMap latents (conditioning for replay + training).
    for uuid in uuids:
        out = LAT_DIR / f"{uuid}_hdmap_latents.pt"
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        (hdmap_path,), _ = _sample_files(uuid)
        encode_to(out, hdmap_path, "hdmaps")

    # Phase C: styled-target latents.
    for uuid, slug, path in styled:
        out = LAT_DIR / f"{uuid}__{slug}_latents.pt"
        if out.exists():
            print(f"skip {out.name} (exists)", flush=True)
            continue
        encode_to(out, path, "latents", allow_short=True)

    phase_a = "reused" if skip_phase_a else "encoded"
    print(
        f"PRECOMPUTE-STYLE-DONE | embeddings {phase_a}, {len(uuids)} clips, "
        f"{len(styled)} styled pairs -> {LAT_DIR}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
