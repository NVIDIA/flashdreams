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

"""GPU validation for the CUDA-graph-safe fused drift corrector.

One phase per process (the deploy hooks mutate the network irreversibly);
all phases share the same styled-rollout protocol: base clip prompt, hot
prompt-swap to the arcade style at chunk ``SWAP_AT``, style-drift
corrector attached at build with the styled gate profile, fixed seed.

Phases (``PHASE`` env):

* ``parity_unfused`` / ``parity_fused`` — eager pipeline (no compile, no
  graphs), corrector in the named mode. Saves frames + per-chunk stats;
  once both exist, writes ``parity.json`` (max/mean frame diff) and an
  sbs frame stack for eyeballing.
* ``bench_base`` — compile_network + use_cuda_graph ON, no corrector.
* ``bench_fused`` — compile_network + use_cuda_graph ON, fused corrector.
  Asserts the cond CUDA graph actually captured (no silent eager
  fallback) after the rollout.
* ``bench_unfused_eager`` — the pre-existing serving compromise: eager
  pipeline with the unfused corrector.

Bench phases run one untimed warmup rollout (compile/autotune) and
``RUNS`` timed rollouts; per-chunk wall latency is synchronized, and
steady-state stats skip the first ``WARM_CHUNKS`` chunks (graph capture
for this config finishes around chunk 5: capture_ar_idx=3 plus two
warmup forwards). fps = 8 frames per steady chunk / median latency.

Run each phase from the flashdreams repo root, e.g.::

    PHASE=parity_unfused .venv/bin/python \
        integrations/omnidreams/scripts/validate_fused_corrector.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

# The gate profile is read at _drift_corrector import time.
os.environ.setdefault(
    "GATE_ALPHA_JSON",
    str(Path(__file__).resolve().parents[1] / "edit_sft/outputs/gate_style.json"),
)

import numpy as np
import torch
from einops import rearrange
from omnidreams._drift_corrector import GATE_ALPHA, apply_drift_corrector
from omnidreams.config import SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.pipeline import OmnidreamsPipeline
from omnidreams.runner import (
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _load_first_frame,
    _load_video,
    _write_video,
)
from torch import Tensor

from flashdreams.infra.config import derive_config
from flashdreams.infra.cuda_graph import CUDAGraphWrapper

_OMNI = Path(__file__).resolve().parents[1]

PHASES = (
    "parity_unfused",
    "parity_fused",
    "bench_base",
    "bench_fused",
    "bench_unfused_eager",
)
PHASE = os.environ.get("PHASE", "")
SEED = int(os.environ.get("SEED", "42"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "16"))
SWAP_AT = int(os.environ.get("SWAP_AT", "2"))
RUNS = int(os.environ.get("RUNS", "3"))
WARM_CHUNKS = int(os.environ.get("WARM_CHUNKS", "8"))
UUID = os.environ.get("UUID", "23599139-948f-4681-b7f4-74794113086d")
CORRECTOR = Path(
    os.environ.get(
        "CORRECTOR", str(_OMNI / "edit_sft/outputs/lora_style_corrector_valpeak.pt")
    )
)
GAIN = float(os.environ.get("CORRECTOR_GAIN", "0.25"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", str(_OMNI / "scripts/outputs/fused_corrector_validation"))
)
STYLE_PROMPT = (
    "A bright arcade racing game world with exaggerated saturated "
    "colors, clean stylized surfaces, and a cheerful sunny palette."
)

SAMPLES_ROOT = (
    Path.home()
    / ".cache/huggingface/hub/datasets--nvidia--omni-dreams-samples/snapshots"
)


def _sample_paths(uuid: str) -> tuple[Path, Path, str]:
    hdmaps = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/*_hdmap.mp4"))
    frames = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/first_frame.png"))
    prompts = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/prompt.txt"))
    assert hdmaps and frames and prompts, (
        f"sample {uuid} not in the local HF cache under {SAMPLES_ROOT}"
    )
    return hdmaps[0], frames[0], prompts[0].read_text().strip()


def _build_pipeline(*, accel: bool) -> OmnidreamsPipeline:
    cfg = derive_config(
        SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
        enable_sync_and_profile=False,
        diffusion_model=dict(
            seed=SEED,
            transformer=dict(compile_network=accel, use_cuda_graph=accel),
        ),
    )
    pipe = cfg.setup()
    assert isinstance(pipe, OmnidreamsPipeline)
    return pipe.to("cuda")


@torch.no_grad()
def _rollout(
    pipe: OmnidreamsPipeline,
    *,
    hdmap: Tensor,
    first: Tensor,
    base_prompt: str,
    decode: bool = True,
) -> tuple[Tensor | None, list[float]]:
    """Styled rollout; returns (decoded video or None, per-chunk seconds)."""
    device = pipe.device
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)
    cache = pipe.initialize_cache(text=[[base_prompt]], image=first)
    chunks: list[Tensor] = []
    latencies: list[float] = []
    start = 0
    for ar_idx in range(N_CHUNKS):
        if ar_idx == SWAP_AT:
            pipe.replace_text(cache, [[STYLE_PROMPT]])
        num_frames = pipe.get_num_frames(ar_idx)
        end = start + num_frames
        assert end <= hdmap.shape[2], f"hdmap too short at chunk {ar_idx}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = pipe.generate(ar_idx, cache, hdmap=hdmap[:, :, start:end])
        pipe.finalize(ar_idx, cache)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
        if decode:
            chunks.append(chunk[0, 0].float().cpu())
        start = end
    del cache
    torch.cuda.empty_cache()
    video = torch.cat(chunks, dim=0) if decode else None
    return video, latencies


def _chunk_bounds() -> list[tuple[int, int]]:
    bounds, start = [], 0
    for ar_idx in range(N_CHUNKS):
        n = 5 if ar_idx == 0 else 8
        bounds.append((start, start + n))
        start += n
    return bounds


def _to_uint8(video: Tensor) -> np.ndarray:
    arr = rearrange(video, "t c h w -> t h w c").numpy()
    return (((arr + 1) / 2 * 255).clip(0, 255)).astype(np.uint8)


def _compare_parity() -> None:
    frames = {m: np.load(OUT_DIR / f"frames_{m}.npy") for m in ("unfused", "fused")}
    a = frames["fused"].astype(np.int16)
    b = frames["unfused"].astype(np.int16)
    diff = np.abs(a - b)
    per_chunk = [
        {
            "chunk": i,
            "mean_abs": float(diff[s:e].mean()),
            "max_abs": int(diff[s:e].max()),
        }
        for i, (s, e) in enumerate(_chunk_bounds())
    ]
    report = {
        "protocol": {
            "uuid": UUID,
            "seed": SEED,
            "n_chunks": N_CHUNKS,
            "swap_at": SWAP_AT,
            "gain": GAIN,
            "gate_alpha": {str(k): v for k, v in GATE_ALPHA.items()},
        },
        "max_abs_pixel_diff_uint8": int(diff.max()),
        "mean_abs_pixel_diff_uint8": float(diff.mean()),
        "per_chunk": per_chunk,
    }
    (OUT_DIR / "parity.json").write_text(json.dumps(report, indent=2))
    # 5-frame sbs stack (unfused top, fused bottom) for the eyes-on check.
    import mediapy as media

    n = len(frames["unfused"])
    t_idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    rows = [
        np.concatenate([frames[m][t] for t in t_idx], axis=1)
        for m in ("unfused", "fused")
    ]
    media.write_image(str(OUT_DIR / "parity_sbs.png"), np.concatenate(rows, axis=0))
    print(json.dumps({k: v for k, v in report.items() if k != "per_chunk"}, indent=2))


def main() -> None:
    assert PHASE in PHASES, f"PHASE={PHASE!r} not in {PHASES}"
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdmap_path, frame_path, clip_prompt = _sample_paths(UUID)
    total_frames = 5 + (N_CHUNKS - 1) * 8
    device = torch.device("cuda")
    hdmap = _load_video(
        hdmap_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )[:total_frames][None, None]
    first = _load_first_frame(
        frame_path,
        pixel_height=DEFAULT_VIDEO_HEIGHT,
        pixel_width=DEFAULT_VIDEO_WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )[None, None]

    accel = PHASE in ("bench_base", "bench_fused")
    pipe = _build_pipeline(accel=accel)

    corrector_mode = {
        "parity_unfused": "unfused",
        "parity_fused": "fused",
        "bench_fused": "fused",
        "bench_unfused_eager": "unfused",
        "bench_base": None,
    }[PHASE]
    if corrector_mode is not None:
        mode_str = apply_drift_corrector(
            SimpleNamespace(pipeline=pipe), CORRECTOR, GAIN, mode=corrector_mode
        )
        print(f"{PHASE}: {mode_str}", flush=True)

    if PHASE.startswith("parity"):
        video, _ = _rollout(pipe, hdmap=hdmap, first=first, base_prompt=clip_prompt)
        assert video is not None
        arr = _to_uint8(video)
        np.save(OUT_DIR / f"frames_{corrector_mode}.npy", arr)
        _write_video(
            rearrange(video, "t c h w -> t h w c"),
            OUT_DIR / f"video_{corrector_mode}.mp4",
            fps=30,
        )
        print("ROLLOUT-DONE", flush=True)
        if all((OUT_DIR / f"frames_{m}.npy").exists() for m in ("unfused", "fused")):
            _compare_parity()
        return

    # Bench phases: one warmup rollout (compile/autotune/graph shakeout),
    # then RUNS timed rollouts. The video of the last run is kept for the
    # eyes-on style-hold check.
    print(f"{PHASE}: warmup rollout ...", flush=True)
    _rollout(pipe, hdmap=hdmap, first=first, base_prompt=clip_prompt, decode=False)
    all_lat: list[list[float]] = []
    video = None
    for r in range(RUNS):
        video, lat = _rollout(
            pipe,
            hdmap=hdmap,
            first=first,
            base_prompt=clip_prompt,
            decode=r == RUNS - 1,
        )
        all_lat.append(lat)
        print(f"run {r}: per-chunk ms = {[round(x * 1e3, 1) for x in lat]}", flush=True)

    if accel:
        transformer = pipe.diffusion_model.transformer
        wrapper = transformer._network_call
        assert isinstance(wrapper, CUDAGraphWrapper) and wrapper._graph is not None, (
            "cond CUDA graph did not capture — graph break or eager fallback"
        )
        print("cond CUDA graph captured and replayed OK", flush=True)

    steady = np.array([run[WARM_CHUNKS:] for run in all_lat])  # seconds
    med = float(np.median(steady))
    report = {
        "phase": PHASE,
        "accel": accel,
        "corrector_mode": corrector_mode,
        "protocol": {
            "uuid": UUID,
            "seed": SEED,
            "n_chunks": N_CHUNKS,
            "swap_at": SWAP_AT,
            "runs": RUNS,
            "warm_chunks": WARM_CHUNKS,
            "gain": GAIN,
        },
        "steady_chunk_ms": {
            "median": med * 1e3,
            "mean": float(steady.mean()) * 1e3,
            "p90": float(np.quantile(steady, 0.9)) * 1e3,
            "min": float(steady.min()) * 1e3,
            "max": float(steady.max()) * 1e3,
        },
        "fps_at_8_frames_per_chunk": 8.0 / med,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (OUT_DIR / f"bench_{PHASE}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["steady_chunk_ms"], indent=2))
    print(f"fps ~= {report['fps_at_8_frames_per_chunk']:.1f}", flush=True)
    if video is not None:
        _write_video(
            rearrange(video, "t c h w -> t h w c"),
            OUT_DIR / f"video_{PHASE}.mp4",
            fps=30,
        )


if __name__ == "__main__":
    main()
