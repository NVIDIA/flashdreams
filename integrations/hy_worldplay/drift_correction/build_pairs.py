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

"""Paired drift data from strafe-loop rollouts (clean lap-1 teacher, zero real videos).

A strafe loop (``w-N, d-N, s-N, a-N`` -- translations only, no yaw) returns
the camera to an identical world pose every ``4 * N`` latents, so every lap
revisits the same views with the same per-frame actions and viewmats. Frame
``j`` of a late (drifted) lap therefore has an exact condition-matched clean
counterpart at frame ``(j mod lap) + lap`` of lap 1 -- the handoff's
"ground-truth-seeded early-window reference", made position- and
camera-consistent by the loop. One long rollout per clip yields both sides
of the pair; no real videos are used anywhere.

Per-clip output (``outputs/pairs/clip_XXXX.pt``): per-chunk patchified clean
latents (history is their concatenation), per-chunk ``memory_frame_indices``,
the rollout-scoped action / camera buffers, and the recipe metadata.
Resumable: existing clip files are skipped. Run from the repo root::

    NUM_CLIPS=10 .venv/bin/python integrations/hy_worldplay/drift_correction/build_pairs.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from _rollout import build_runner, capture_rollout

## Pair-set configuration

NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "10"))
"""Clips to capture in this invocation (resume-aware)."""

NUM_LAPS = int(os.environ.get("NUM_LAPS", "6"))
"""Strafe-loop laps per rollout; laps >= 2 supply drifted windows against the
lap-1 teacher."""

LEGS = [int(x) for x in os.environ.get("LEGS", "4").split(",")]
"""Motion steps per loop leg, rotated per clip. A single ``4`` gives the
16-latent lap matching the memory budget; mixing (e.g. ``3,4,5`` -> laps of
12/16/20 latents) breaks the fixed revisit period so "content returns at lag
16" stops being a learnable shortcut (the v2 repeat-prior failure)."""

LAP_PATTERNS = (
    "w-{n}, d-{n}, s-{n}, a-{n}",
    "w-{n}, a-{n}, s-{n}, d-{n}",
    "d-{n}, w-{n}, a-{n}, s-{n}",
    "a-{n}, w-{n}, d-{n}, s-{n}",
)
"""Closed strafe loops (translation order permutations); rotated per clip for
trajectory variety."""

IMAGES_DIR = os.environ.get("IMAGES_DIR", "")
"""Directory of first-frame images; empty -> the upstream sample image for
every clip (seed/pattern variety only)."""

PROMPTS_FILE = os.environ.get("PROMPTS_FILE", "")
"""Prompt source (one per line, e.g. Self-Forcing's
``MovieGenVideoBench_extended.txt``) for seed-image-matched text conditioning:
``frame_NNNN.png`` seeds (from ``gen_first_frames.py``) roll with line
``NNNN`` of this file instead of the integration's mismatched default
prompt. Other image names keep the default."""

CORRECTOR_LORA = os.environ.get("CORRECTOR_LORA", "")
"""Optional LoRA checkpoint; when set, rollouts run with the corrector
merged in at scale 1 (the DAgger round: pairs reflect the states the
deployed corrector actually visits)."""

OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR", "integrations/hy_worldplay/drift_correction/outputs/pairs"
    )
)
"""Per-clip pair files plus ``manifest.json``."""

_BASE_SEED = 31000
"""Clip ``i`` rolls with diffusion seed ``_BASE_SEED + i``."""


def loop_pose(pattern: str, leg: int, num_laps: int) -> str:
    """Build a multi-lap pose string with ``4 * leg * num_laps - 1`` motion steps.

    The parser prepends an identity pose for the input frame, so the last
    leg is shortened by one step to keep total latents = ``num_chunk * 4``.
    """
    lap = pattern.format(n=leg)
    laps = [lap] * num_laps
    head, n = lap.rsplit("-", 1)
    assert int(n) == leg
    laps[-1] = f"{head}-{leg - 1}" if leg > 1 else ", ".join(lap.split(", ")[:-1])
    return ", ".join(laps)


def matched_prompt(image_path: Path | None) -> str | None:
    """Return the prompt that generated ``frame_NNNN.png``, else ``None``."""
    if image_path is None or not image_path.stem.startswith("frame_"):
        return None
    try:
        idx = int(image_path.stem.split("_")[1])
        lines = [
            ln.strip()
            for ln in Path(PROMPTS_FILE).read_text().splitlines()
            if ln.strip()
        ]
        return lines[idx]
    except (ValueError, IndexError, OSError):
        return None


def main() -> None:
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if IMAGES_DIR:
        images_dir = Path(IMAGES_DIR)
        images: list[Path | None] = [
            *sorted(images_dir.glob("*.png")),
            *sorted(images_dir.glob("*.jpg")),
        ]
        assert images, f"IMAGES_DIR {images_dir} contains no .png/.jpg files"
    else:
        images = [None]

    runner = None
    manifest: dict[str, dict] = {}
    manifest_path = OUT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    for i in range(NUM_CLIPS):
        clip_path = OUT_DIR / f"clip_{i:04d}.pt"
        if clip_path.exists():
            print(f"clip {i:04d}: exists, skipping", flush=True)
            continue
        pattern = LAP_PATTERNS[i % len(LAP_PATTERNS)]
        leg = LEGS[i % len(LEGS)]
        lap_latents = 4 * leg
        num_chunk = lap_latents * NUM_LAPS // 4
        pose = loop_pose(pattern, leg, NUM_LAPS)
        image_path = images[i % len(images)]
        if runner is None:
            runner = build_runner(
                num_chunk=num_chunk,
                pose=pose,
                output_dir=OUT_DIR,
                image_path=image_path,
            )
            if CORRECTOR_LORA:
                from _lora import (
                    apply_lora,
                    load_lora,
                    set_lora_scale,
                    unwrap_compiled,
                )

                network = unwrap_compiled(
                    runner.pipeline.diffusion_model.transformer.network
                )
                apply_lora(network)
                load_lora(network, CORRECTOR_LORA)
                set_lora_scale(network, 1.0)
                print(f"corrector active: {CORRECTOR_LORA}", flush=True)
        else:
            runner.config.pose = pose
            runner.config.num_chunk = num_chunk
            if image_path is not None:
                runner.config.image_path = image_path
        prompt = matched_prompt(image_path)
        if prompt:
            runner.config.prompt = prompt
        seed = _BASE_SEED + i
        print(f"clip {i:04d}: pose[{pattern}] seed={seed} ...", flush=True)
        snaps = capture_rollout(runner, noise_seed=seed)

        ctrl0 = snaps[1].ctrl  # chunk >= 1 carries the rollout-scoped buffers
        assert (
            ctrl0.rollout_action is not None
            and ctrl0.rollout_viewmats is not None
            and ctrl0.rollout_Ks is not None
        ), "chunk >= 1 ctrl must carry the rollout-scoped buffers"
        torch.save(
            {
                "latents": torch.cat(
                    [s.clean_latent.to(torch.float16).cpu() for s in snaps], dim=-2
                ),
                "memory_frame_indices": [s.ctrl.memory_frame_indices for s in snaps],
                "rollout_action": ctrl0.rollout_action.cpu(),
                "rollout_viewmats": ctrl0.rollout_viewmats.to(torch.float16).cpu(),
                "rollout_Ks": ctrl0.rollout_Ks.to(torch.float16).cpu(),
                "lap_latents": lap_latents,
                "num_chunk": num_chunk,
                "pose": pose,
                "seed": seed,
                "image": str(image_path) if image_path else "example",
            },
            clip_path,
        )
        manifest[f"{i:04d}"] = {"pose": pose, "seed": seed, "num_chunk": num_chunk}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"clip {i:04d}: saved {clip_path}", flush=True)


if __name__ == "__main__":
    main()
