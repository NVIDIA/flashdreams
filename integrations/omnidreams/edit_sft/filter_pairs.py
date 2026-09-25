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

"""VLM filter for the edit-SFT pair corpus (Tier-2b, quality gate).

Scores every (source, instruction, edited) triplet with Cosmos-Reason1
(a Qwen2.5-VL — already cached as the pipeline's text encoder) on three
axes, 0-5 each:

* ``edit_applied`` — is the instructed edit visible in the edited video?
* ``persistent`` — is it present across early / mid / late frames?
* ``scene_preserved`` — is everything else (layout, road, buildings)
  unchanged from the source?

A pair passes at >= 3 on all three. Writes ``filter_report.json`` next to
the pairs plus a per-instruction summary table, so weak instruction types
are visible before training.

Run from the repo root::

    .venv/bin/python integrations/omnidreams/edit_sft/filter_pairs.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mediapy as media
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "nvidia/Cosmos-Reason1-7B"
REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
"""Same pin as the pipeline's text encoder — weights already cached."""

PAIRS_DIR = Path(
    os.environ.get("PAIRS_DIR", "integrations/omnidreams/edit_sft/outputs/pairs")
)
SOURCES_DIR = Path(
    os.environ.get("SOURCES_DIR", "integrations/omnidreams/edit_sft/outputs/sources")
)
JOBS_FILES = os.environ.get("JOBS", "jobs.jsonl").split(",")
FRAME_IDS = (30, 110, 190)
"""Early / mid / late sample frames (221-frame clips)."""

MODE = os.environ.get("MODE", "object")
"""``object`` (default): original three-axis gate. ``style``: for global
restyles — replaces ``scene_preserved`` with style-agnostic ROAD-LAYOUT
checks at an early and a late frame, so streaming layout drift (the
observed failure mode: the road is progressively replaced) is caught and
heavy styles can still qualify for early-window-only training."""

PASS_BAR = 3
MAX_SIDE = 640
"""Longest image side fed to the judge (speed/VRAM)."""


def _frames(path: Path, ids=FRAME_IDS) -> list[Image.Image]:
    video = media.read_video(path)
    out = []
    for i in ids:
        frame = video[min(i, len(video) - 1)]
        img = Image.fromarray(np.asarray(frame))
        scale = MAX_SIDE / max(img.size)
        out.append(img.resize((int(img.width * scale), int(img.height * scale))))
    return out


def _parse_scores(text: str, keys: tuple[str, ...]) -> dict[str, int]:
    """Extract integer scores from the judge's reply (JSON or loose text)."""
    scores: dict[str, int] = {}
    try:
        hit = re.search(r"\{.*\}", text, re.S)
        blob = json.loads(hit.group(0) if hit else "")
        for key in keys:
            if key in blob:
                scores[key] = max(0, min(5, int(blob[key])))
    except (AttributeError, ValueError, TypeError, json.JSONDecodeError):
        pass
    for key in keys:
        if key not in scores:
            hit = re.search(rf"{key}\D{{0,12}}([0-5])", text)
            scores[key] = int(hit.group(1)) if hit else 0
    return scores


class Judge:
    """Thin greedy-decoding wrapper around Cosmos-Reason1 as a VLM judge."""

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME, revision=REVISION, local_files_only=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, revision=REVISION, local_files_only=True, dtype=torch.bfloat16
        ).to("cuda")  # ty: ignore[invalid-argument-type]  # stub loses the bound self
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def ask(self, images: list[Image.Image], question: str) -> str:
        content = [{"type": "image"} for _ in images] + [
            {"type": "text", "text": question}
        ]
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(
            "cuda"
        )
        out = self.model.generate(
            **inputs, max_new_tokens=96, do_sample=False, temperature=None, top_p=None
        )
        reply = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        return reply


_LAYOUT_QUESTION = (
    "Image 1 is a real driving photo. Image 2 is a stylized version of the "
    "same scene. Look only at WHERE the road, lane lines, and vehicles are "
    "placed in the picture. First, in one sentence, say whether the road in "
    "image 2 runs through the same part of the picture in the same direction "
    "as in image 1. Then on the last line write SCORE: n where n is 0-5 "
    "(5 = road and objects in the same places, 0 = road moved or missing)."
)
"""Describe-then-score framing: a bare cross-style JSON score collapses to a
uniform 0 (the style gap dominates); letting the judge state the comparison
first produces calibrated scores. A reply with no SCORE line counts as 0 —
in testing that degenerate mode only appeared on true layout failures."""


def _judge_layout(judge: "Judge", source: Path, edited: Path, frame: int) -> int:
    reply = judge.ask(
        _frames(source, ids=(frame,)) + _frames(edited, ids=(frame,)),
        _LAYOUT_QUESTION,
    )
    hit = re.search(r"SCORE:\s*([0-5])", reply)
    return int(hit.group(1)) if hit else 0


def main() -> None:
    jobs = [
        json.loads(line)
        for name in JOBS_FILES
        for line in (PAIRS_DIR / name).read_text().splitlines()
        if line.strip()
    ]
    judge = Judge()
    report: list[dict] = []

    for i, job in enumerate(jobs):
        edited_path = Path(job["output"])
        source_path = Path(job["input"])
        if not edited_path.exists():
            report.append({"output": edited_path.name, "error": "missing"})
            continue
        instruction = job["instruction"]
        slug = edited_path.stem.split("__", 1)[1]

        edited = _frames(edited_path)
        reply1 = judge.ask(
            edited,
            "These are three frames (early, middle, late) from an edited "
            f'driving video. The requested edit was: "{instruction}". '
            'Score strictly and answer ONLY JSON like {"edit_applied": 0-5, '
            '"persistent": 0-5}: edit_applied = how clearly the edit is '
            "visible; persistent = whether it is present in ALL three frames.",
        )
        s1 = _parse_scores(reply1, ("edit_applied", "persistent"))

        entry = {
            "output": edited_path.name,
            "uuid": source_path.stem,
            "slug": slug,
            "instruction": instruction,
            **s1,
        }
        if MODE == "style":
            entry["layout_early"] = _judge_layout(judge, source_path, edited_path, 60)
            entry["layout_late"] = _judge_layout(judge, source_path, edited_path, 190)
            entry["passed"] = (
                min(entry["edit_applied"], entry["persistent"]) >= PASS_BAR
                and entry["layout_late"] >= PASS_BAR
            )
            entry["early_window_ok"] = (
                entry["edit_applied"] >= PASS_BAR and entry["layout_early"] >= PASS_BAR
            )
        else:
            src_mid = _frames(source_path, ids=(110,))
            edit_mid = _frames(edited_path, ids=(110,))
            reply2 = judge.ask(
                src_mid + edit_mid,
                "Image 1 is a frame from an original driving video; image 2 is "
                "the same frame after a video edit was applied. Ignoring the "
                f'requested edit ("{instruction}"), score how well everything '
                "else is preserved (same street layout, buildings, lighting, "
                'other vehicles). Answer ONLY JSON like {"scene_preserved": 0-5}.',
            )
            entry.update(_parse_scores(reply2, ("scene_preserved",)))
            entry["passed"] = all(
                entry[k] >= PASS_BAR
                for k in ("edit_applied", "persistent", "scene_preserved")
            )
        report.append(entry)
        if (i + 1) % 15 == 0:
            print(f"{i + 1}/{len(jobs)} judged", flush=True)

    (PAIRS_DIR / "filter_report.json").write_text(json.dumps(report, indent=2))

    # Per-instruction summary.
    by_slug: dict[str, list[dict]] = {}
    for entry in report:
        if "error" not in entry:
            by_slug.setdefault(entry["slug"], []).append(entry)
    extra = (
        ("layout_early", "layout_late", "early_window_ok")
        if MODE == "style"
        else ("scene_preserved",)
    )
    header = " ".join(f"{k[:9]:>9s}" for k in ("edit_applied", "persistent") + extra)
    print(f"\n{'instruction':22s} {'pass':>5s} {header}")
    for slug in sorted(by_slug):
        rows = by_slug[slug]
        n_pass = sum(r["passed"] for r in rows)
        means = " ".join(
            f"{float(np.mean([r[k] for r in rows])):9.1f}"
            for k in ("edit_applied", "persistent") + extra
        )
        print(f"{slug:22s} {n_pass:2d}/{len(rows):<2d} {means}")
    total_pass = sum(e.get("passed", False) for e in report)
    print(
        f"\nFILTER-DONE | {total_pass}/{len(report)} pairs pass -> filter_report.json"
    )


if __name__ == "__main__":
    main()
