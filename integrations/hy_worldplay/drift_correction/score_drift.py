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

"""Drift + guard scoring for paired eval rollouts (Delta-MUSIQ, dynamics, strips).

Per MP4: per-frame MUSIQ (pyiqa) at a fixed frame stride, Delta-drift =
MUSIQ(first 20%) - MUSIQ(all) (BAgger's metric), a RAFT mean-flow dynamic
degree (guard: a corrector that freezes motion "wins" every consistency
metric), the reference protocol metrics (DINO latesim anchoring, lag-2s
identity, full-rate cut count), and a 10-frame contact strip PNG.
Aggregates base-vs-corr into ``outputs/eval/scores.json``.

pyiqa / timm are not part of the project deps; run with an ephemeral overlay::

    uv run --with pyiqa --with timm python integrations/hy_worldplay/drift_correction/score_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import os

EVAL_DIR = Path(
    os.environ.get(
        "EVAL_OUT", "integrations/hy_worldplay/drift_correction/outputs/eval"
    )
)
FRAME_STRIDE = 13
"""Score every 13th frame (one per AR chunk at the decoded rate)."""

DINO_FPS = 1.0
"""DINO feature sampling rate — the reference host's ``posthoc_metrics.py``
samples every 16th frame at 16 fps, so 1 fps keeps latesim / lag-2s on the
cross-host-comparable scale (SF 0.17 / av1s 0.20 / av2s 0.50)."""


def read_frames(path: Path, stride: int) -> np.ndarray:
    """Decode every ``stride``-th frame of an MP4 to ``[N, H, W, 3]`` uint8."""
    import imageio.v3 as iio

    frames = iio.imread(path, plugin="pyav")
    return frames[::stride]


def musiq_curve(frames: np.ndarray, model, device) -> list[float]:
    """Per-frame MUSIQ scores."""
    scores = []
    for f in frames:
        t = torch.from_numpy(f.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        scores.append(float(model(t)))
    return scores


def sharpness_curve(frames: np.ndarray) -> list[float]:
    """Per-frame variance-of-Laplacian (grayscale) — catches blur/structure
    loss that MUSIQ under-penalizes (observed on v1-pilot rollouts)."""
    from scipy.ndimage import laplace

    out = []
    for f in frames:
        gray = f.astype(np.float32).mean(axis=-1)
        out.append(float(laplace(gray).var()))
    return out


def sim_to_start(frames: np.ndarray) -> float:
    """Mean last-20% frame-correlation to the first-2s mean frame.

    The reference repo's progression-bias probe (its
    ``TODO_progression_bias.md`` convention, cross-host comparable scale:
    SF 0.17 / av1s 0.20 / av2s 0.50). Lower = healthy progression, but
    near-zero can be collapse — read alongside the quality guards. On a
    forward trajectory the view should decorrelate from the start; a
    corrector with a learned repeat prior plateaus high instead."""
    g = frames.astype(np.float32).mean(axis=-1)[:, ::4, ::4]
    n2s = max(1, min(len(g), 3))  # first ~2s at the chunk-stride sampling
    anchor = g[:n2s].mean(axis=0)
    anchor = anchor - anchor.mean()
    sims = []
    for f in g[-max(1, len(g) // 5) :]:
        fc = f - f.mean()
        sims.append(
            float(
                (anchor * fc).sum()
                / (np.linalg.norm(anchor) * np.linalg.norm(fc) + 1e-8)
            )
        )
    return float(np.mean(sims))


def sat_drift(frames: np.ndarray) -> float:
    """Mean |saturation - saturation(frame 0)| (the reference's in-domain
    drift metric); catches the color-drift axis MUSIQ tolerates."""
    f = frames.astype(np.float32) / 255
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0).mean(axis=(1, 2))
    return float(np.abs(sat - sat[0]).mean())


def dino_features(frames: np.ndarray, model, mean, std, device) -> torch.Tensor:
    """L2-normalized DINO features for ``[N, H, W, 3]`` uint8 frames."""
    import torch.nn.functional as F

    feats = []
    for f in frames:
        x = torch.from_numpy(f.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        feats.append(F.normalize(model((x - mean) / std).float(), dim=-1))
    return torch.cat(feats)


def latesim(feats: torch.Tensor) -> float:
    """Mean DINO similarity of the last fifth to the opening (first-3 mean).

    The reference protocol's anchoring metric (``posthoc_metrics.py``): a
    corrector with an anchoring pull plateaus high while the base
    decorrelates. Semantic-feature complement to the pixel-space
    ``sim_to_start``; read alongside the dynamics guard.
    """
    import torch.nn.functional as F

    ref = F.normalize(feats[:3].mean(0, keepdim=True), dim=-1)
    sims = (feats @ ref.T).squeeze(-1).cpu().numpy()
    return float(np.mean(sims[-max(1, len(sims) // 5) :]))


def lag2s_identity(feats: torch.Tensor, lag: int = 2) -> float:
    """Mean DINO similarity between frames ``lag`` seconds apart.

    Identity-persistence probe (subjects morphing lowers it); ``lag`` is in
    ``DINO_FPS`` samples. ``nan`` when the clip is shorter than the lag.
    """
    if len(feats) <= lag:
        return float("nan")
    return float((feats[:-lag] * feats[lag:]).sum(-1).mean())


CHUNK_FRAMES = 16
"""Decoded frames per AR chunk after the first: Wan's 4x temporal VAE maps
the 4-latent seed chunk to 13 frames, then 16 per chunk (381 = 13 + 23*16
at 24 chunks), so chunk boundaries sit at frame ``13 + 16k`` (per-frame
motion autocorrelation peaks at lag 16)."""

FIRST_CHUNK_FRAMES = 13
"""Decoded frames in the seed chunk (excluded from seam phase alignment)."""


def seam_motion_ratio(frames: np.ndarray) -> float:
    """Motion at chunk-boundary transitions relative to chunk-interior motion.

    Phase-aligns full-rate adjacent-frame diffs to the decoded chunk cadence
    (boundaries at frame ``13 + 16k``; seed chunk excluded) and returns
    mean(boundary phases 0-1) / mean(interior phases 4-11). ~1.0 = motion
    flows through the seams; above 1 = a boundary jump (anchoring kick /
    statistics snap); below 1 = motion stalling at the seam.
    """
    g = frames.astype(np.float32).mean(axis=-1)[:, ::2, ::2]
    mot = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    seg = mot[FIRST_CHUNK_FRAMES - 1 :]  # phase 0 = transition into chunk 1
    n = (len(seg) // CHUNK_FRAMES) * CHUNK_FRAMES
    if n < 2 * CHUNK_FRAMES:
        return float("nan")
    phases = seg[:n].reshape(-1, CHUNK_FRAMES).mean(axis=0)
    return float(phases[:2].mean() / (phases[4:12].mean() + 1e-9))


def seam_sharpness_ratio(frames: np.ndarray) -> float:
    """Sharpness of chunk-initial frames relative to chunk-interior frames.

    Variance-of-Laplacian per frame, phase-aligned to the decoded chunk
    cadence (chunk-initial frames at ``13 + 16k``; seed chunk excluded):
    mean(phases 0-1) / mean(phases 6-13). Well below 1 = post-boundary blur
    (structural detail loss on each chunk's first frames).
    """
    from scipy.ndimage import laplace

    g = frames.astype(np.float32).mean(axis=-1)[:, ::2, ::2]
    sharp = np.array([float(laplace(f).var()) for f in g[FIRST_CHUNK_FRAMES:]])
    n = (len(sharp) // CHUNK_FRAMES) * CHUNK_FRAMES
    if n < 2 * CHUNK_FRAMES:
        return float("nan")
    phases = sharp[:n].reshape(-1, CHUNK_FRAMES).mean(axis=0)
    return float(phases[:2].mean() / (phases[6:14].mean() + 1e-9))


def cut_count(frames: np.ndarray) -> int:
    """Count hard cuts: adjacent full-rate mean-abs-diff above
    ``max(0.10, mu + 5*sigma)`` (reference convention) — catches HY's
    content-level jump-cut at the context/memory handoff."""
    diffs = np.empty(len(frames) - 1, dtype=np.float32)
    for i in range(len(frames) - 1):
        a = frames[i].astype(np.float32)
        b = frames[i + 1].astype(np.float32)
        diffs[i] = float(np.abs(b - a).mean() / 255.0)
    thr = max(0.10, float(diffs.mean() + 5 * diffs.std()))
    return int((diffs > thr).sum())


def dynamic_degree(frames: np.ndarray, raft, device) -> float:
    """Mean RAFT flow magnitude between scored frames (motion guard)."""
    import torch.nn.functional as F

    mags = []
    for a, b in zip(frames[:-1], frames[1:]):
        ta = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].float().to(device)
        tb = torch.from_numpy(b.copy()).permute(2, 0, 1)[None].float().to(device)
        ta = F.interpolate(ta, size=(352, 640), mode="bilinear") / 127.5 - 1
        tb = F.interpolate(tb, size=(352, 640), mode="bilinear") / 127.5 - 1
        flow = raft(ta, tb)[-1]
        mags.append(float(flow.square().sum(1).sqrt().mean()))
    return float(np.mean(mags))


def contact_strip(frames: np.ndarray, out_path: Path, n: int = 10) -> None:
    """Write an n-frame horizontal strip PNG for eyeballing."""
    from PIL import Image

    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    strip = np.concatenate([frames[i] for i in idx], axis=1)
    Image.fromarray(strip).save(out_path)


def main() -> None:
    import imageio.v3 as iio
    import pyiqa
    import timm
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

    device = "cuda" if torch.cuda.is_available() else "cpu"
    musiq = pyiqa.create_metric("musiq", device=device)
    raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
    dino = (
        timm.create_model("vit_base_patch16_224.dino", pretrained=True, num_classes=0)
        .to(device)
        .eval()
    )
    dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    results: dict[str, dict] = {}
    configs = [d.name for d in sorted(EVAL_DIR.iterdir()) if d.is_dir()]
    for config in configs:
        cfg_dir = EVAL_DIR / config
        per_video = {}
        for mp4 in sorted(cfg_dir.glob("*.mp4")):
            if mp4.stem.startswith("sbs_"):
                continue
            # One full-rate decode per video; strided views feed the
            # per-chunk metrics, ~1 fps feeds DINO, full rate feeds cuts.
            frames_full = read_frames(mp4, 1)
            fps = float(iio.immeta(mp4, plugin="pyav").get("fps", 16.0))
            frames = frames_full[::FRAME_STRIDE]
            dino_stride = max(1, round(fps / DINO_FPS))
            with torch.no_grad():
                feats = dino_features(
                    frames_full[::dino_stride], dino, dino_mean, dino_std, device
                )
            curve = musiq_curve(frames, musiq, device)
            sharp = sharpness_curve(frames)
            n20 = max(1, len(curve) // 5)
            delta = float(np.mean(curve[:n20]) - np.mean(curve))
            # Late-vs-early sharpness ratio: ~1.0 = detail preserved; well
            # below 1 = the compounding-blur failure mode.
            sharp_ratio = float(np.mean(sharp[-n20:]) / (np.mean(sharp[:n20]) + 1e-9))
            with torch.no_grad():
                dyn = dynamic_degree(frames, raft, device)
            contact_strip(frames, mp4.with_suffix(".strip.png"))
            per_video[mp4.stem] = {
                "musiq_overall": float(np.mean(curve)),
                "musiq_late": float(np.mean(curve[-n20:])),
                "delta_drift": delta,
                "dynamic_degree": dyn,
                "sharpness_ratio": sharp_ratio,
                "sat_drift": sat_drift(frames),
                "sim_to_start": sim_to_start(frames),
                "latesim": latesim(feats),
                "lag2s": lag2s_identity(feats),
                "cuts": cut_count(frames_full),
                "seam_motion_ratio": seam_motion_ratio(frames_full),
                "seam_sharpness_ratio": seam_sharpness_ratio(frames_full),
                "curve": curve,
                "sharpness": sharp,
            }
            v = per_video[mp4.stem]
            print(
                f"{config}/{mp4.stem}: MUSIQ {np.mean(curve):.1f} "
                f"(late {np.mean(curve[-n20:]):.1f}) | Delta {delta:+.2f} | "
                f"dyn {dyn:.1f} | sharp-ratio {sharp_ratio:.2f} | "
                f"latesim {v['latesim']:.3f} | lag2s {v['lag2s']:.3f} | "
                f"cuts {v['cuts']}",
                flush=True,
            )
        agg = {
            key: float(np.mean([v[key] for v in per_video.values()]))
            for key in (
                "musiq_overall",
                "musiq_late",
                "delta_drift",
                "dynamic_degree",
                "sharpness_ratio",
                "sat_drift",
                "sim_to_start",
                "latesim",
                "lag2s",
                "cuts",
                "seam_motion_ratio",
                "seam_sharpness_ratio",
            )
        }
        results[config] = {"videos": per_video, "aggregate": agg}

    (EVAL_DIR / "scores.json").write_text(json.dumps(results, indent=2))
    print("\n================ closed-loop drift eval ================")
    print(
        f"{'':10s} {'MUSIQ':>7s} {'late':>7s} {'Delta':>7s} {'dyn':>6s} "
        f"{'sharp':>6s} {'sat':>7s} {'sim':>5s} {'lsim':>6s} {'lag2':>6s} "
        f"{'cuts':>5s} {'seam':>6s} {'sseam':>6s}"
    )
    for name in configs:
        a = results[name]["aggregate"]
        print(
            f"{name:10s} {a['musiq_overall']:7.2f} {a['musiq_late']:7.2f} "
            f"{a['delta_drift']:+7.2f} {a['dynamic_degree']:6.1f} "
            f"{a['sharpness_ratio']:6.2f} {a['sat_drift']:7.4f} "
            f"{a['sim_to_start']:5.2f} {a['latesim']:6.3f} {a['lag2s']:6.3f} "
            f"{a['cuts']:5.1f} {a['seam_motion_ratio']:6.3f} "
            f"{a['seam_sharpness_ratio']:6.3f}"
        )
    b = results.get("base", {}).get("aggregate")
    for name in configs:
        if name == "base" or b is None or b["delta_drift"] <= 0:
            continue
        red = (
            100
            * (b["delta_drift"] - results[name]["aggregate"]["delta_drift"])
            / b["delta_drift"]
        )
        print(f"{name}: Delta-drift reduction vs base: {red:.0f}% (target >= 30%)")
    print(
        "guards: dynamic degree within ~20% of base (motion freeze) and "
        "sharpness ratio not far below base (compounding blur)."
    )


if __name__ == "__main__":
    main()
