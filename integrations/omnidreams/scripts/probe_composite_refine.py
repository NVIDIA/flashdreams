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

"""Probe: composite-then-refine (SDEdit-style) through the 2-step student.

The billboard-sprite composites (``composite_assets.py``) look like 2D
paper cutouts. This probe asks whether a light renoise -> redenoise pass
through the distilled student "re-renders" the pasted objects into
scene-consistent photoreal ones. Per chunk, instead of sampling from the
model's own noise, the "clean latent" comes from renoising the COMPOSITED
chunk latent at strength S and denoising from that state; the REFINED
latent is committed to the KV cache, so temporal consistency builds on
refined history. Conditioning is the boxed HDMap (the sprites sit exactly
at those boxes, so conditioning and pixels agree).

Arms (FlowMatchScheduler, shift=5, denoising_timesteps=[1000, 450]):

- ``light``: start at the schedule's second step — z = add_noise(c, 803.57)
  (sigma 0.8036, i.e. keep 19.6% composite signal) and run just the final
  Euler/x0 step.
- ``mid``: off-schedule single step at raw t=250 -> warped 625.0
  (sigma 0.625, keep 37.5% composite signal); the [1000, 250] PERF
  experiment shows the student tolerates queries there.
- ``heavy``: full 2-step regeneration from sigma 1.0 (the composite
  contributes nothing at step 0 — pure conditioning-driven arm).

The manifold pull is the experiment: it may integrate the object (win) or
erase/deform it (fail); the renoise strength trades one against the other.

Honest-metric guardrail (project evidence standard): per arm we report the
mask-anchored in-box |refined - composited| (how much the pasted objects
changed) and out-box |refined - composited| (collateral scene damage),
masks recovered from |boxed - baseline| HDMap diff as in
``composite_assets.py``.

Env knobs: ``N_CHUNKS`` (default 26 = full clip), ``SEED``, ``ARMS``
(comma list, default ``light,mid,heavy``), ``OUT_BASE``, ``COMPOSITE_SBS``
(side-by-side composited input video; right half is refined),
``BOXED_HDMAP`` / ``BASELINE_HDMAP`` (conditioning + box-mask renders,
which must match the composite's source rollout).

Run from the repo root::

    .venv/bin/python integrations/omnidreams/scripts/probe_composite_refine.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Must land before the first CUDA allocation (co-tenant VRAM share).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import OmnidreamsPipeline, OmnidreamsPipelineCache
from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner_io import read_video_rgb, write_video_tensor

from flashdreams.infra.diffusion.model import DiffusionModel

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


SCRIPTS = Path("integrations/omnidreams/scripts")
OUTPUTS = SCRIPTS / "outputs"

COMPOSITE_SBS = Path(
    os.environ.get(
        "COMPOSITE_SBS", str(OUTPUTS / "pr_videos/composite_crowd_midroad_real.mp4")
    )
)
BOXED_HDMAP = Path(
    os.environ.get("BOXED_HDMAP", str(OUTPUTS / "ped20_hdmap/hdmap.mp4"))
)
BASELINE_HDMAP = Path(
    os.environ.get("BASELINE_HDMAP", str(OUTPUTS / "baseline_hdmap/hdmap.mp4"))
)
SCENE_PROMPT = (
    Path.home()
    / ".cache/flashdreams/omnidreams-scenes/0d404ff7-2b66-498c-b047-1ed8cded60d4"
    / "clipgt/prompt1.txt"
)

N_CHUNKS = int(os.environ.get("N_CHUNKS", "26"))
SEED = int(os.environ.get("SEED", "42"))
ARM_NAMES = os.environ.get("ARMS", "light,mid,heavy").split(",")
OUT_BASE = Path(os.environ.get("OUT_BASE", str(OUTPUTS)))

DIFF_THRESHOLD = 40  # composite_assets.py box-mask convention
PNG_FRAMES = (40, 104, 180)


def _total_frames(n_chunks: int) -> int:
    return 5 + (n_chunks - 1) * 8


def _build_pipeline() -> OmnidreamsPipeline:
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
    return pipe.to("cuda")


@torch.no_grad()
def _encode_chunks(
    pipe: OmnidreamsPipeline, video: Tensor, n_chunks: int
) -> list[Tensor]:
    """Stream-encode ``[T, 3, H, W]`` pixels into per-chunk patchified latents.

    Same schedule/space as ``edit_sft/precompute_style.py``: fresh streaming
    encoder cache, AR chunk schedule (5 frames then 8), patchified via the
    transformer so the latents live exactly where ``predict_flow`` consumes
    them. Returned on CPU (bf16).
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


def _sigma_for(pipe: OmnidreamsPipeline, warped_timestep: float) -> float:
    """Exact sigma the scheduler's ``add_noise`` will snap this timestep to."""
    sched = pipe.diffusion_model.scheduler
    full_t = sched._full_timesteps
    idx = int(torch.argmin((full_t - warped_timestep).abs()).item())
    return float(sched._full_sigmas[idx].item())


@torch.no_grad()
def _refine_rollout(
    pipe: OmnidreamsPipeline,
    cache: OmnidreamsPipelineCache,
    *,
    comp_latents: list[Tensor],
    hdmap_latents: list[Tensor],
    start_timestep: float | None,
) -> Tensor:
    """Teacher-forced refinement rollout; returns decoded ``[T, 3, H, W]`` on CPU.

    Per chunk: renoise the composited latent to ``start_timestep`` (warped
    units; ``None`` = full-schedule regeneration from pure noise), denoise
    from that state, inject the I2V image latent (chunk 0), commit the
    REFINED latent to the KV cache through the production ``finalize`` path
    (context re-noise at t=128), and decode.
    """
    device = pipe.device
    dm = pipe.diffusion_model
    transformer = dm.transformer
    sched = dm.scheduler
    dm._rng = torch.Generator(device=device).manual_seed(SEED)
    tc = cache.transformer_cache
    assert pipe.decoder is not None and cache.decoder_cache is not None

    frames: list[Tensor] = []
    for ar_idx in range(len(comp_latents)):
        hd = hdmap_latents[ar_idx].to(device)
        comp = comp_latents[ar_idx].to(device)
        tc.start(ar_idx)

        def predict_flow(z: Tensor, t: Tensor) -> Tensor:
            return transformer.predict_flow(
                noisy_latent=z, timestep=t, cache=tc, input=hd
            )

        if start_timestep is None:
            # Full distilled schedule from sigma=1: identical to a normal
            # generate() except history/conditioning-driven (no composite).
            noise = torch.randn(
                comp.shape, device=device, dtype=comp.dtype, generator=dm._rng
            )
            clean = sched.sample(
                initial_noise=noise,
                predict_flow=predict_flow,  # ty: ignore[invalid-argument-type]
                rng=dm._rng,
            )
        else:
            sigma = _sigma_for(pipe, start_timestep)
            t = torch.tensor(start_timestep, device=device)
            z = sched.add_noise(clean_input=comp, timestep=t, rng=dm._rng)
            # Cast as the scheduler's sample() does: network timesteps run
            # at the latent dtype.
            flow = predict_flow(z, t.to(dtype=comp.dtype))
            clean = z - sigma * flow

        clean = transformer.postprocess_clean_latent(
            clean_latent=clean, cache=tc, input=hd
        )
        # Commit the REFINED latent to the KV history via the production
        # path (re-noise at context t=128 + finalize_kv_cache + tc.finalize).
        dm.finalize(
            DiffusionModel.FinalState(
                clean_latent=clean, autoregressive_index=ar_idx, cache=tc, input=hd
            )
        )
        decoded = pipe.decoder(
            input=transformer.unpatchify_and_maybe_gather_cp(clean),
            autoregressive_index=ar_idx,
            cache=cache.decoder_cache,
        )
        frames.append(decoded[0, 0].float().cpu())
    return torch.cat(frames, dim=0)


def _box_masks(total_frames: int) -> Tensor:
    """Per-frame injected-box masks ``[T, H, W]`` bool (boxed vs baseline diff)."""
    boxed = _load_video(
        BOXED_HDMAP,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[:total_frames]
    baseline = _load_video(
        BASELINE_HDMAP,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[:total_frames]
    assert baseline.shape[0] >= total_frames, "baseline hdmap shorter than clip"
    diff = (boxed - baseline).abs().amax(dim=1) * 127.5  # [T, H, W] uint8 units
    return diff > DIFF_THRESHOLD


def _mask_report(
    refined: Tensor, composited: Tensor, masks: Tensor
) -> dict[str, float]:
    """In-box / out-box mean |refined - composited| in uint8 units."""
    diff = (refined - composited).abs().mean(dim=1) * 127.5  # [T, H, W]
    in_vals = diff[masks]
    out_vals = diff[~masks]
    return {
        "in_box_mean": float(in_vals.mean()) if in_vals.numel() else float("nan"),
        "out_box_mean": float(out_vals.mean()),
        "box_pixel_fraction": float(masks.float().mean()),
    }


def _write_comparison_png(
    path: Path, videos: dict[str, Tensor], frame_indices: tuple[int, ...]
) -> None:
    """Grid PNG: rows = frames, columns = (composited, *arms)."""
    from PIL import Image

    rows = []
    for f in frame_indices:
        row = [
            ((videos[name][f].permute(1, 2, 0).numpy() + 1.0) * 127.5)
            .clip(0, 255)
            .astype(np.uint8)
            for name in videos
        ]
        rows.append(np.concatenate(row, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main() -> None:
    total_frames = _total_frames(N_CHUNKS)
    prompt = SCENE_PROMPT.read_text().strip()
    print(f"chunks={N_CHUNKS} frames={total_frames} arms={ARM_NAMES}")

    # The composite PR video is side-by-side (original | composited); take
    # the RIGHT half. Loading at the native sbs width avoids any resize.
    sbs = _load_video(
        COMPOSITE_SBS,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=2 * DEFAULT_VIDEO_WIDTH,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert sbs.shape[0] >= total_frames, f"composite clip too short: {sbs.shape[0]}"
    composited = sbs[:total_frames, :, :, DEFAULT_VIDEO_WIDTH:].contiguous()
    del sbs
    hdmap = _load_video(
        BOXED_HDMAP,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )[:total_frames]

    pipe = _build_pipeline()

    # One-shot embeddings, then drop the ~14 GB text encoder: each arm
    # rebuilds its rollout cache from these.
    first = composited[:1][None, None].to(pipe.device)  # [1, 1, 1, 3, H, W]
    emb = pipe.precompute_embeddings(text=[[prompt]], image=first)
    pipe.release_oneshot_encoders()

    print("encoding composited + hdmap chunks ...", flush=True)
    comp_latents = _encode_chunks(pipe, composited, N_CHUNKS)
    hdmap_latents = _encode_chunks(pipe, hdmap, N_CHUNKS)

    sched = pipe.diffusion_model.scheduler
    arms: dict[str, float | None] = {
        # Warped timestep to renoise the composite to; None = full schedule.
        "light": float(sched.denoising_step_list[1].item()),  # 803.57, sigma .8036
        "mid": 625.0,  # raw t=250 warped, sigma 0.625
        "soft": 468.75,  # raw t=150 warped, sigma 0.4688
        "heavy": None,  # sigma 1.0 then the 803.57 step
    }
    arms = {k: arms[k] for k in ARM_NAMES}

    videos: dict[str, Tensor] = {"composited": composited.float()}
    report: dict[str, dict] = {}
    for name, start_t in arms.items():
        sigma = None if start_t is None else _sigma_for(pipe, start_t)
        print(f"arm {name}: start_timestep={start_t} sigma={sigma}", flush=True)
        cache = pipe.initialize_cache_from_embeddings(
            text_embeddings=emb["text_embeddings"],
            image_embeddings=emb["image_embeddings"],
            negative_text_embeddings=emb["negative_text_embeddings"],
        )
        refined = _refine_rollout(
            pipe,
            cache,
            comp_latents=comp_latents,
            hdmap_latents=hdmap_latents,
            start_timestep=start_t,
        )
        del cache
        torch.cuda.empty_cache()
        videos[name] = refined
        out_dir = OUT_BASE / f"composite_refine_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        write_video_tensor(refined, out_dir / "drive.mp4", fps=30, layout="tchw")
        report[name] = {"start_timestep": start_t, "sigma": sigma}
        print(f"wrote {out_dir / 'drive.mp4'}", flush=True)

    masks = _box_masks(total_frames)
    comp_f = videos["composited"]
    for name in arms:
        report[name].update(_mask_report(videos[name], comp_f, masks))
        print(f"{name}: {report[name]}", flush=True)

    report_dir = OUT_BASE / f"composite_refine_{ARM_NAMES[0]}"
    png_path = report_dir / "comparison.png"
    frame_indices = tuple(min(f, total_frames - 1) for f in PNG_FRAMES)
    _write_comparison_png(png_path, videos, frame_indices)
    (report_dir / "report.json").write_text(
        json.dumps(
            {
                "n_chunks": N_CHUNKS,
                "seed": SEED,
                "prompt_file": str(SCENE_PROMPT),
                "context_noise": pipe.diffusion_model.config.context_noise,
                "arms": report,
            },
            indent=2,
        )
    )
    print(f"comparison PNG: {png_path}", flush=True)


if __name__ == "__main__":
    main()
