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

"""Regenerate the JoyAI styled targets through the 35-step bidirectional teacher.

Issue 1's final quality lever: every style-LoRA version was capped by SOFT
training targets — the JoyAI RGB restyles re-encoded through the VAE. This
script produces SHARP, temporally consistent styled targets by SDEdit-style
renoise-refine through the TEACHER (which is photoreal-trained and will not
restyle from a prompt alone — the low-noise styled latent carries the style
and layout, the teacher re-renders them at its native sharpness):

1. Encode a filter-passed JoyAI styled video
   (``style_pairs/<uuid>__<slug>.mp4``) with the teacher's Wan-VAE encoder
   into ONE bidirectional chunk48 window. ``len_t = 48`` LATENT frames =
   189 pixel frames (``1 + 47 * 4``) — exactly student chunks 0..23, which
   covers the trainer's maximum supervised chunk
   (``SWAP_MAX + SPAN - 1 = 23``), so a single window per clip suffices.
2. Renoise the styled latent at a mid sigma of the teacher's 35-step
   shift-5 UniPC schedule and run the REMAINDER of the schedule (partial
   UniPC restart: order-1 warmup at the entry step, exactly the baked
   coefficients afterwards), CFG 3.0 with the style prompt
   (``style_prompts.py`` declarative phrasing) + the clip's HDMap
   conditioning.
3. Decode through the light TAE decoder (mandatory — the full Wan-VAE
   decode OOMs on 48-latent-frame chunks) and write
   ``<uuid>__<slug>.mp4`` + a styled-vs-teacher contact sheet.

The teacher pipeline setup (chunk48 config, checkpoint key filter, light
TAE decoder) mirrors ``spawn_distill/gen_teacher_pairs.py``; the
renoise-refine mechanics mirror ``scripts/probe_composite_refine.py``.

Env knobs: ``SIGMAS`` (comma list; >1 value = sigma sweep, suffixing
outputs with ``__s<sigma>``), ``STYLES`` (default
``arcade_racer,comic_ink``), ``LIMIT`` (max pairs, 0 = all), ``ONLY``
(comma list of ``<uuid>__<slug>`` stems), ``OUT_DIR`` (default
``outputs/style_pairs_teacher``), ``SEED``, ``MAX_USED_GB`` (VRAM
co-tenant gate, default 100).

Run from the repo root (sweep first, then batch)::

    SIGMAS=0.35,0.5,0.65 LIMIT=1 OUT_DIR=.../style_pairs_teacher_sweep \\
        .venv/bin/python integrations/omnidreams/edit_sft/gen_teacher_styled_targets.py
    SIGMAS=<chosen> \\
        .venv/bin/python integrations/omnidreams/edit_sft/gen_teacher_styled_targets.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from omnidreams.config import (
    SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
)
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.transformer.impl.network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkConfig,
)
from style_prompts import STYLE_PROMPTS
from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner_io import read_video_rgb, write_video_tensor

DEFAULT_VIDEO_HEIGHT = 704
DEFAULT_VIDEO_WIDTH = 1280


def _load_video(
    path: Path,
    *,
    pixel_height: int,
    pixel_width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Load + resize a video to ``[T, C, H, W]`` in ``[-1, 1]``."""
    import cv2  # noqa: PLC0415

    video_np = read_video_rgb(path)
    if video_np.shape[1:3] != (pixel_height, pixel_width):
        video_np = np.stack(
            [cv2.resize(f, (pixel_width, pixel_height)) for f in video_np], axis=0
        )
    tensor = torch.from_numpy(video_np).to(dtype=dtype, device=device) / 127.5 - 1.0
    return tensor.permute(0, 3, 1, 2).contiguous()


## Configuration

BASE = Path("integrations/omnidreams/edit_sft")
STYLE_DIR = Path(os.environ.get("STYLE_DIR", str(BASE / "outputs/style_pairs")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(BASE / "outputs/style_pairs_teacher")))

SIGMAS = [float(s) for s in os.environ.get("SIGMAS", "0.35,0.5,0.65").split(",")]
STYLES = frozenset(os.environ.get("STYLES", "arcade_racer,comic_ink").split(","))
LIMIT = int(os.environ.get("LIMIT", "0"))
ONLY = frozenset(s for s in os.environ.get("ONLY", "").split(",") if s)
SEED = int(os.environ.get("SEED", "42"))
MAX_USED_GB = float(os.environ.get("MAX_USED_GB", "100"))

TEACHER_FRAMES = 189
"""One bidirectional chunk48 window: 48 latent frames = 1 + 47 * 4 pixels."""

CONTACT_FRAMES = (0, 40, 90, 140, 188)

_HF_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--omni-dreams-models/snapshots"
    / "253701787e2f99efec31aaab665d0d9e0cc1eb4a/single_view"
)
_TEACHER_CKPT = _HF_SNAPSHOT / "teacher/3b4c21d0-7b77-4694-9d9d-6ac9b6dbba51_model.pt"


def _network_key_filter_transform():
    """``net.``-prefix strip + drop training-stack extras (teacher probe)."""
    with torch.device("meta"):
        reference = CosmosDiTNetwork(CosmosDiTNetworkConfig(additional_concat_ch=16))
    valid = set(reference.state_dict().keys())
    del reference

    def transform(state_dict):
        out = {}
        for key, value in state_dict.items():
            key = key[len("net.") :] if key.startswith("net.") else key
            if key in valid:
                out[key] = value
        assert len(out) == len(valid), (
            f"teacher checkpoint covers {len(out)}/{len(valid)} network keys"
        )
        return out

    return transform


def _wait_for_vram(max_used_gb: float) -> None:
    """Block until the co-tenant leaves enough VRAM for the teacher (~150 GB)."""
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        used_gb = max(float(line) for line in out.strip().splitlines()) / 1024.0
        if used_gb <= max_used_gb:
            return
        print(f"GPU busy ({used_gb:.0f} GB used > {max_used_gb:.0f}); waiting 60 s")
        time.sleep(60)


def _build_pipeline() -> OmnidreamsPipeline:
    cfg = derive_config(
        SV_35STEPS_CHUNK48_LOC48_COSMOS2_2B_RES720P_30FPS_HDMAP_VAE_MADS1M,
        name="omnidreams-sv-teacher-styled-targets-chunk48",
        enable_sync_and_profile=False,
        # Full Wan-VAE decode OOMs on 48-latent-frame chunks; the light TAE
        # decoder shares the latent space and decode is output-only.
        decoder=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE.decoder,
        diffusion_model=dict(
            seed=SEED,
            transformer=dict(
                checkpoint_path=str(_TEACHER_CKPT),
                state_dict_transform=_network_key_filter_transform(),
                compile_network=False,
                use_cuda_graph=False,
            ),
        ),
    )
    pipe = cfg.setup()
    assert isinstance(pipe, OmnidreamsPipeline)
    return pipe.to("cuda")


def _hdmap_path(uuid: str) -> Path:
    root = (
        Path.home()
        / ".cache/huggingface/hub/datasets--nvidia--omni-dreams-samples/snapshots"
    )
    hits = sorted(root.glob(f"*/data/single_view/{uuid}/*_hdmap.mp4"))
    assert hits, f"sample {uuid} hdmap not in the local HF cache"
    return hits[0]


def _passed_pairs() -> list[tuple[str, str, Path]]:
    """Filter-passed ``(uuid, slug, path)`` styled pairs in ``STYLES``."""
    report = json.loads((STYLE_DIR / "filter_report.json").read_text())
    passed = {e["output"] for e in report if e.get("passed")}
    out: list[tuple[str, str, Path]] = []
    for path in sorted(STYLE_DIR.glob("*__*.mp4")):
        uuid, slug = path.stem.split("__", 1)
        if path.name not in passed or slug not in STYLES:
            continue
        if ONLY and path.stem not in ONLY:
            continue
        out.append((uuid, slug, path))
    return out


@torch.no_grad()
def _encode_window(pipe: OmnidreamsPipeline, video: Tensor) -> Tensor:
    """Encode ``[189, 3, H, W]`` pixels into the patchified chunk-0 latent."""
    transformer = pipe.diffusion_model.transformer
    assert pipe.encoder is not None
    enc_cache = pipe.encoder.initialize_autoregressive_cache()
    pixels = video[None, None].to(pipe.device)
    z = pipe.encoder(input=pixels, autoregressive_index=0, cache=enc_cache)
    return transformer.patchify_and_maybe_split_cp(z)


@torch.no_grad()
def _sdedit_window(
    pipe: OmnidreamsPipeline,
    cache,
    *,
    styled_latent: Tensor,
    hdmap_latent: Tensor,
    sigma_target: float,
    rng: torch.Generator,
) -> tuple[Tensor, dict]:
    """Renoise-refine one chunk48 window from ``sigma_target``.

    Partial restart of the scheduler's own order-2 UniPC loop: enter the
    schedule at the step whose sigma is nearest ``sigma_target``, renoise
    the styled latent there, and run the remaining steps with the baked
    per-step coefficients (order-1 warmup at the entry step — no
    predictor/corrector history exists yet — matching ``sample()``'s own
    step-0 behavior). Returns the decoded ``[T, 3, H, W]`` video on CPU.
    """
    transformer = pipe.diffusion_model.transformer
    sched = pipe.diffusion_model.scheduler
    tc = cache.transformer_cache
    assert pipe.decoder is not None and cache.decoder_cache is not None
    tc.start(0)

    n_steps = int(sched.timesteps.shape[0])
    i0 = int(torch.argmin((sched.sigmas - sigma_target).abs()).item())
    sigma0 = float(sched.sigmas[i0].item())
    dtype = styled_latent.dtype

    noise = torch.randn(
        styled_latent.shape, device=styled_latent.device, dtype=dtype, generator=rng
    )
    sample = ((1.0 - sigma0) * styled_latent.float() + sigma0 * noise.float()).to(dtype)

    m_prev: Tensor | None = None
    m_prev_prev: Tensor | None = None
    last_sample: Tensor | None = None
    t_start = time.perf_counter()
    for i in range(i0, n_steps):
        timestep = sched.timesteps[i].to(dtype=dtype)
        flow = transformer.predict_flow(
            noisy_latent=sample, timestep=timestep, cache=tc, input=hdmap_latent
        )
        m_curr = sample.to(torch.float32) - sched.sigmas[i] * flow.to(torch.float32)
        if i > i0:
            assert last_sample is not None and m_prev is not None
            m_pp = m_prev_prev if m_prev_prev is not None else m_prev
            sample = (
                sched.a_corr[i] * last_sample.to(torch.float32)
                + sched.b_corr_m0[i] * m_prev
                + sched.b_corr_dprev[i] * (m_pp - m_prev)
                + sched.b_corr_dt[i] * (m_curr - m_prev)
            ).to(dtype)
        last_sample = sample
        m_p = m_prev if m_prev is not None else m_curr
        sample = (
            sched.a_pred[i] * sample.to(torch.float32)
            + sched.b_pred_m0[i] * m_curr
            + sched.b_pred_dprev[i] * (m_p - m_curr)
        ).to(dtype)
        m_prev_prev, m_prev = m_prev, m_curr

    clean = transformer.postprocess_clean_latent(
        clean_latent=sample, cache=tc, input=hdmap_latent
    )
    decoded = pipe.decoder(
        input=transformer.unpatchify_and_maybe_gather_cp(clean),
        autoregressive_index=0,
        cache=cache.decoder_cache,
    )
    info = {
        "sigma_target": sigma_target,
        "sigma_snapped": sigma0,
        "start_step": i0,
        "steps_run": n_steps - i0,
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    return decoded[0, 0].float().cpu(), info


def _to_uint8(frame: Tensor) -> np.ndarray:
    return (
        ((frame.permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    )


def _write_contact_sheet(path: Path, rows: dict[str, Tensor]) -> None:
    """Grid PNG: rows = named videos, columns = ``CONTACT_FRAMES``."""
    from PIL import Image, ImageDraw

    stacked = []
    for name, video in rows.items():
        row = np.concatenate(
            [_to_uint8(video[min(f, video.shape[0] - 1)]) for f in CONTACT_FRAMES],
            axis=1,
        )
        img = Image.fromarray(row)
        ImageDraw.Draw(img).text((8, 8), name, fill=(255, 255, 0))
        stacked.append(np.asarray(img))
    Image.fromarray(np.concatenate(stacked, axis=0)).save(path)


@torch.no_grad()
def main() -> None:
    pairs = _passed_pairs()
    if LIMIT:
        pairs = pairs[:LIMIT]
    assert pairs, f"no filter-passed pairs for styles {sorted(STYLES)} in {STYLE_DIR}"
    print(
        f"{len(pairs)} pairs x sigmas {SIGMAS} -> {OUT_DIR}",
        flush=True,
    )
    _wait_for_vram(MAX_USED_GB)

    pipe = _build_pipeline()
    dtype = torch.bfloat16
    device = pipe.device

    # Phase A: one-shot embeddings for every pair (style prompt + negative +
    # styled first frame), then drop the ~14 GB text encoder.
    styled_videos: dict[str, Tensor] = {}
    emb: dict[str, dict] = {}
    emb_by_prompt: dict[str, Tensor] = {}
    for uuid, slug, path in pairs:
        video = _load_video(
            path,
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        assert video.shape[0] >= TEACHER_FRAMES, (
            f"{path.name}: {video.shape[0]} frames < {TEACHER_FRAMES}"
        )
        styled_videos[path.stem] = video[:TEACHER_FRAMES].contiguous()
        first = styled_videos[path.stem][:1][None, None].to(device)
        e = pipe.precompute_embeddings(text=[[STYLE_PROMPTS[slug]]], image=first)
        # Text/negative embeddings only depend on the slug; share them.
        if STYLE_PROMPTS[slug] not in emb_by_prompt:
            emb_by_prompt[STYLE_PROMPTS[slug]] = e["text_embeddings"]
        emb[path.stem] = {
            "text": emb_by_prompt[STYLE_PROMPTS[slug]],
            "negative": e["negative_text_embeddings"],
            "image": e["image_embeddings"],
        }
        print(f"embedded {path.stem}", flush=True)
    pipe.release_oneshot_encoders()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    report_path = OUT_DIR / "regen_report.json"
    for uuid, slug, path in pairs:
        stem = path.stem
        styled = styled_videos[stem]
        outputs = {
            (f"{stem}__s{sigma:g}" if len(SIGMAS) > 1 else stem): sigma
            for sigma in SIGMAS
        }
        if all((OUT_DIR / f"{name}.mp4").exists() for name in outputs):
            print(f"{stem}: exists, skipped", flush=True)
            continue

        styled_latent = _encode_window(pipe, styled)
        hdmap = _load_video(
            _hdmap_path(uuid),
            pixel_height=DEFAULT_VIDEO_HEIGHT,
            pixel_width=DEFAULT_VIDEO_WIDTH,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        assert hdmap.shape[0] >= TEACHER_FRAMES, f"{uuid}: hdmap too short"
        hdmap_latent = _encode_window(pipe, hdmap[:TEACHER_FRAMES].contiguous())

        sheet_rows: dict[str, Tensor] = {"styled (JoyAI)": styled.float()}
        for name, sigma in outputs.items():
            rng = torch.Generator(device=device).manual_seed(
                SEED + zlib.crc32(name.encode()) % 100_000
            )
            cache = pipe.initialize_cache_from_embeddings(
                text_embeddings=emb[stem]["text"],
                image_embeddings=emb[stem]["image"],
                negative_text_embeddings=emb[stem]["negative"],
            )
            refined, info = _sdedit_window(
                pipe,
                cache,
                styled_latent=styled_latent,
                hdmap_latent=hdmap_latent,
                sigma_target=sigma,
                rng=rng,
            )
            del cache
            torch.cuda.empty_cache()
            write_video_tensor(refined, OUT_DIR / f"{name}.mp4", fps=30, layout="tchw")
            sheet_rows[f"teacher s={info['sigma_snapped']:.3f}"] = refined
            report[name] = info
            print(f"{name}: {info}", flush=True)

        _write_contact_sheet(OUT_DIR / f"{stem}_contact.png", sheet_rows)
        report_path.write_text(json.dumps(report, indent=2))
        del styled_latent, hdmap_latent
        torch.cuda.empty_cache()

    print(f"TEACHER-STYLED-TARGETS-DONE -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
