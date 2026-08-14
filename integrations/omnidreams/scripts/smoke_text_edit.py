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

"""Smoke test: mid-stream prompt swap on the real distilled model.

Rolls the same seed / HDMap / first frame several ways and reports how
strongly the video diverges after the swap chunk:

    A  control       original clip prompt throughout
    B  swap          hot-swap to ``EDIT_PROMPT`` at chunk ``SWAP_AT``
    C  swap+guide    same swap with two-prompt edit guidance
    D  swap+recache  same swap plus previous-chunk KV re-commit

B and C consume the identical RNG stream as A (the swap itself draws no
noise), so the per-chunk ``|B - A|`` pixel gap is a pure measure of prompt
responsiveness: ~0 before the swap (sanity check), and the post-swap
magnitude/growth is the signal. D draws one extra context-noise sample at
the recache, so its pre-swap sanity still holds but its post-swap gap is
noise-shifted — judge D visually against B.

Env knobs: ``UUID``, ``EDIT_PROMPT``, ``N_CHUNKS``, ``SWAP_AT``,
``GUIDE_SCALE``, ``GUIDE_CHUNKS``, ``SEED``, ``OUT_DIR``.

Run from the repo root::

    .venv/bin/python integrations/omnidreams/scripts/smoke_text_edit.py
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
N_CHUNKS = int(os.environ.get("N_CHUNKS", "16"))
SWAP_AT = int(os.environ.get("SWAP_AT", "8"))
GUIDE_SCALE = float(os.environ.get("GUIDE_SCALE", "2.5"))
GUIDE_CHUNKS = int(os.environ.get("GUIDE_CHUNKS", "4"))
SEED = int(os.environ.get("SEED", "42"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "integrations/omnidreams/scripts/outputs/text_edit_smoke")
)
EDIT_PROMPT = os.environ.get(
    "EDIT_PROMPT",
    "Driving scene from a front-facing car camera at night in a heavy "
    "snowstorm. Thick snow falling, snow-covered road and buildings, "
    "headlights and streetlights glowing through the snow. Photorealistic "
    "dashcam footage.",
)


def _sample_paths(uuid: str) -> tuple[Path, Path, str]:
    hdmaps = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/*_hdmap.mp4"))
    frames = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/first_frame.png"))
    prompts = sorted(SAMPLES_ROOT.glob(f"*/data/single_view/{uuid}/prompt.txt"))
    assert hdmaps and frames and prompts, (
        f"sample {uuid} not in the local HF cache under {SAMPLES_ROOT}"
    )
    return hdmaps[0], frames[0], prompts[0].read_text().strip()


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
    pipe = pipe.to("cuda")
    # EDIT_LORA=<ckpt>: deploy the pre-merged guidance-distillation LoRA, so
    # the guided variants exercise the production use_lora window instead of
    # the two-branch combine.
    if os.environ.get("EDIT_LORA"):
        from omnidreams._edit_lora import TextEditLoRA

        transformer = pipe.diffusion_model.transformer
        edit_lora = TextEditLoRA(transformer.network, os.environ["EDIT_LORA"])
        transformer.set_text_edit_lora(edit_lora)
        print(f"deployed {edit_lora.describe()}", flush=True)
    return pipe


@torch.no_grad()
def _rollout(
    pipe: OmnidreamsPipeline,
    *,
    hdmap: Tensor,
    first: Tensor,
    base_prompt: str,
    swap: dict | None = None,
) -> Tensor:
    """Return the decoded rollout ``[T, 3, H, W]`` in ``[-1, 1]`` on CPU."""
    device = pipe.device
    pipe.diffusion_model._rng = torch.Generator(device=device).manual_seed(SEED)
    cache = pipe.initialize_cache(text=[[base_prompt]], image=first)
    chunks: list[Tensor] = []
    start = 0
    for ar_idx in range(N_CHUNKS):
        repulse = int(os.environ.get("REPULSE_EVERY", "0"))
        if (
            swap is not None
            and repulse > 0
            and ar_idx > swap["at"]
            and (ar_idx - swap["at"]) % repulse == 0
        ):
            # Re-open the edit window before the previous one's style fades:
            # duty-cycled skin for LoRAs whose long-hold drifts.
            pipe.replace_text(
                cache,
                [[swap["prompt"]]],
                guidance_scale=swap.get("scale", 1.0),
                guidance_chunks=swap.get("chunks", 0),
                recache_last_chunk=False,
            )
        if swap is not None and ar_idx == swap["at"]:
            pipe.replace_text(
                cache,
                [[swap["prompt"]]],
                guidance_scale=swap.get("scale", 1.0),
                guidance_chunks=swap.get("chunks", 0),
                recache_last_chunk=swap.get("recache", False),
            )
        num_frames = pipe.get_num_frames(ar_idx)
        end = start + num_frames
        assert end <= hdmap.shape[2], f"hdmap too short at chunk {ar_idx}"
        chunk = pipe.generate(ar_idx, cache, hdmap=hdmap[:, :, start:end])
        pipe.finalize(ar_idx, cache)
        chunks.append(chunk[0, 0].float().cpu())
        start = end
    del cache
    torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def _chunk_bounds() -> list[tuple[int, int]]:
    bounds, start = [], 0
    for ar_idx in range(N_CHUNKS):
        n = 5 if ar_idx == 0 else 8
        bounds.append((start, start + n))
        start += n
    return bounds


def _per_chunk_gap(a: Tensor, b: Tensor) -> list[float]:
    """Mean |a - b| per chunk in uint8 units (0..255)."""
    return [float((a[s:e] - b[s:e]).abs().mean() * 127.5) for s, e in _chunk_bounds()]


def main() -> None:
    hdmap_path, frame_path, clip_prompt = _sample_paths(UUID)
    total_frames = 5 + (N_CHUNKS - 1) * 8
    print(f"clip {UUID}\n  prompt: {clip_prompt}\n  edit:   {EDIT_PROMPT}")
    print(f"  chunks={N_CHUNKS} swap_at={SWAP_AT} frames={total_frames}")

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
    )[None, None]  # [B=1, V=1, 1, C, H, W]

    pipe = _build_pipeline()

    variants: dict[str, dict | None] = {
        "control": None,
        "swap": {"at": SWAP_AT, "prompt": EDIT_PROMPT},
        "swap_guided": {
            "at": SWAP_AT,
            "prompt": EDIT_PROMPT,
            "scale": GUIDE_SCALE,
            "chunks": GUIDE_CHUNKS,
        },
        "swap_recache": {"at": SWAP_AT, "prompt": EDIT_PROMPT, "recache": True},
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos: dict[str, Tensor] = {}
    for name, swap in variants.items():
        print(f"rolling out {name} ...", flush=True)
        videos[name] = _rollout(
            pipe, hdmap=hdmap, first=first, base_prompt=clip_prompt, swap=swap
        )
        write_video_tensor(videos[name], OUT_DIR / f"{name}.mp4", fps=30, layout="tchw")

    control = videos["control"]
    report: dict[str, list[float]] = {}
    for name in ("swap", "swap_guided", "swap_recache"):
        gaps = _per_chunk_gap(videos[name], control)
        report[name] = gaps
        pre = max(gaps[:SWAP_AT])
        post = gaps[SWAP_AT:]
        print(
            f"{name:>13}: pre-swap max gap {pre:6.3f}  "
            f"post-swap per-chunk {' '.join(f'{g:6.2f}' for g in post)}"
        )

    # Side-by-side [control | swap | swap_guided] for eyeballing.
    sbs = torch.cat(
        [control, videos["swap"], videos["swap_guided"]], dim=3
    )  # widths concat
    write_video_tensor(sbs, OUT_DIR / "sbs.mp4", fps=30, layout="tchw")

    meta = {
        "uuid": UUID,
        "clip_prompt": clip_prompt,
        "edit_prompt": EDIT_PROMPT,
        "n_chunks": N_CHUNKS,
        "swap_at": SWAP_AT,
        "guide_scale": GUIDE_SCALE,
        "guide_chunks": GUIDE_CHUNKS,
        "seed": SEED,
        "per_chunk_gap_uint8": report,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(meta, indent=2))
    print(f"videos + report under {OUT_DIR}/")


if __name__ == "__main__":
    main()
