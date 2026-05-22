# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run VLM-based artifact checks on OmniDreams sweep outputs.

This evaluator complements scalar quality metrics such as NIQE/MUSIQ/CLIPIQA.
It samples frames from each cropped generated video, builds an indexed contact
sheet, and asks a pluggable VLM backend to score targeted driving artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flashdreams.eval._video_decode import get_video_frame_batch, get_video_frame_count

DEFAULT_ROOT = Path("/home/gtong/github/flashdreams/outputs/omnidreams-quality-sweep")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
SCHEMA_VERSION = 1

ARTIFACT_CATEGORIES = {
    "hallucinated_vehicle": (
        "Vehicles that appear unsupported by the scene, duplicate unnaturally, "
        "merge with the road, float, or have impossible geometry."
    ),
    "sign_glyph": (
        "Road signs, traffic signs, license plates, or text glyphs that are "
        "scrambled, unreadable, inconsistent, or semantically impossible."
    ),
    "traffic_light": (
        "Traffic lights with impossible colors, duplicated lamps, malformed "
        "housings, or inconsistent signal states."
    ),
    "lane_geometry": (
        "Lane markings, curbs, crosswalks, or road boundaries that bend, split, "
        "vanish, or conflict with the drivable scene."
    ),
    "road_user_anomaly": (
        "Pedestrians, cyclists, cones, barriers, or other road users/objects "
        "with implausible shape, scale, location, or interaction."
    ),
    "temporal_inconsistency": (
        "Artifacts visible across the sampled sequence, such as object pop-in, "
        "identity changes, or inconsistent geometry between frames."
    ),
}


@dataclass(frozen=True)
class ClipInput:
    metrics_path: Path
    output_dir: Path
    relative_output_dir: str
    cropped_video: Path
    source_video: Path | None


@dataclass
class BackendResult:
    parsed: dict[str, Any]
    raw_response: str


class ArtifactBackend(ABC):
    """Backend interface for artifact-scoring VLMs."""

    name: str

    @abstractmethod
    def analyze(self, *, contact_sheet: Path, prompt: str) -> BackendResult:
        """Return parsed artifact scores for one contact sheet."""


class QwenLocalBackend(ArtifactBackend):
    """Local Hugging Face Transformers backend for Qwen2.5-VL."""

    name = "qwen-local"

    def __init__(
        self,
        *,
        model_id: str,
        device_map: str,
        dtype: str,
        max_new_tokens: int,
        temperature: float,
        trust_remote_code: bool,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code
        self._processor = None
        self._model = None

    def _torch_dtype(self):
        import torch

        if self.dtype == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]

    def _load(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = self._torch_dtype()
        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
        }
        if dtype is not None:
            kwargs["dtype"] = dtype

        try:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **kwargs
            )
        except TypeError:
            if "dtype" in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id, **kwargs
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )

    def analyze(self, *, contact_sheet: Path, prompt: str) -> BackendResult:
        self._load()
        assert self._model is not None
        assert self._processor is not None

        import torch

        image = Image.open(contact_sheet).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        device = next(self._model.parameters()).device
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generate_kwargs)

        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            output_ids[len(input_ids[i]) :]
            for i, output_ids in enumerate(generated_ids)
        ]
        raw = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return BackendResult(parsed=parse_artifact_json(raw), raw_response=raw)


class OpenAIBackendPlaceholder(ArtifactBackend):
    """Placeholder backend boundary for a future OpenAI implementation."""

    name = "openai"

    def analyze(self, *, contact_sheet: Path, prompt: str) -> BackendResult:
        raise NotImplementedError(
            "The OpenAI VLM backend is not implemented yet. Use --backend qwen-local."
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)
        f.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_clips(root: Path, metrics_json_name: str) -> list[ClipInput]:
    clips: list[ClipInput] = []
    for metrics_path in sorted(root.rglob(metrics_json_name)):
        metrics = read_json(metrics_path)
        if metrics.get("status") != "ok":
            continue
        cropped = Path(metrics["cropped_video"])
        source = Path(metrics["source_video"]) if metrics.get("source_video") else None
        if not cropped.exists():
            cropped = metrics_path.with_name(cropped.name)
        if source is not None and not source.exists():
            source = metrics_path.with_name(source.name)
        clips.append(
            ClipInput(
                metrics_path=metrics_path,
                output_dir=metrics_path.parent,
                relative_output_dir=metrics.get(
                    "relative_output_dir", str(metrics_path.parent.relative_to(root))
                ),
                cropped_video=cropped,
                source_video=source,
            )
        )
    return clips


def sample_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("cannot sample frames from an empty video")
    if sample_count <= 1:
        return [total_frames // 2]
    if total_frames <= sample_count:
        return list(range(total_frames))
    return [
        int(round(i * (total_frames - 1) / (sample_count - 1)))
        for i in range(sample_count)
    ]


def resize_frame(frame: np.ndarray, thumb_width: int) -> Image.Image:
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid frame size")
    thumb_height = max(1, round(height * thumb_width / width))
    return image.resize((thumb_width, thumb_height), Image.Resampling.BICUBIC)


def build_contact_sheet(
    video_path: Path,
    *,
    output_path: Path,
    sample_count: int,
    columns: int,
    thumb_width: int,
    overwrite: bool,
) -> tuple[list[int], tuple[int, int]]:
    if output_path.exists() and not overwrite:
        total = get_video_frame_count(video_path)
        return sample_indices(total, sample_count), Image.open(output_path).size

    total = get_video_frame_count(video_path)
    indices = sample_indices(total, sample_count)
    frames = get_video_frame_batch(video_path, indices)
    thumbs = [resize_frame(frame, thumb_width) for frame in frames]

    label_height = 28
    columns = max(1, min(columns, len(thumbs)))
    rows = math.ceil(len(thumbs) / columns)
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs) + label_height
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#111111")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (frame_index, image) in enumerate(zip(indices, thumbs)):
        col = idx % columns
        row = idx // columns
        x = col * cell_width
        y = row * cell_height
        draw.rectangle([x, y, x + cell_width, y + label_height], fill="#f5f0e6")
        draw.text(
            (x + 8, y + 8),
            f"sample {idx + 1} / frame {frame_index}",
            fill="#181714",
            font=font,
        )
        sheet.paste(image, (x, y + label_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return indices, sheet.size


def make_prompt(frame_indices: list[int]) -> str:
    categories = "\n".join(
        f"- {name}: {description}" for name, description in ARTIFACT_CATEGORIES.items()
    )
    return f"""
You are evaluating a generated autonomous-driving video from an indexed contact sheet.
Each tile is labeled with its sample number and source frame index.

Task: detect targeted generation artifacts, especially hallucinated vehicles and sign/text glyph problems.

Artifact categories:
{categories}

Scoring rubric:
- severity 0: not visible
- severity 1: minor or uncertain artifact
- severity 2: clear artifact that should be reviewed
- severity 3: severe artifact likely to invalidate the clip
- confidence: number from 0.0 to 1.0

Sampled source frame indices: {frame_indices}

Return only valid JSON with this exact top-level shape:
{{
  "schema_version": {SCHEMA_VERSION},
  "artifact_scores": {{
    "hallucinated_vehicle": {{"severity": 0, "confidence": 0.0, "evidence": ""}},
    "sign_glyph": {{"severity": 0, "confidence": 0.0, "evidence": ""}},
    "traffic_light": {{"severity": 0, "confidence": 0.0, "evidence": ""}},
    "lane_geometry": {{"severity": 0, "confidence": 0.0, "evidence": ""}},
    "road_user_anomaly": {{"severity": 0, "confidence": 0.0, "evidence": ""}},
    "temporal_inconsistency": {{"severity": 0, "confidence": 0.0, "evidence": ""}}
  }},
  "overall_artifact_severity": 0,
  "frame_indices_with_issues": [],
  "quality_notes": ""
}}
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        text = match.group(1)

    start = text.find("{")
    if start < 0:
        raise ValueError("VLM response did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError("VLM response contained an unterminated JSON object")


def clamp_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(high, parsed))


def clamp_float(value: Any, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(high, parsed))


def normalize_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scores = payload.get("artifact_scores") or {}
    normalized_scores: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_CATEGORIES:
        item = scores.get(name) or {}
        severity = clamp_int(item.get("severity", 0), 0, 3)
        normalized_scores[name] = {
            "severity": severity,
            "confidence": clamp_float(item.get("confidence", 0.0), 0.0, 1.0),
            "evidence": str(item.get("evidence", "")),
        }

    overall = payload.get("overall_artifact_severity")
    if overall is None:
        overall = max(item["severity"] for item in normalized_scores.values())
    overall = clamp_int(overall, 0, 3)
    issue_frames = payload.get("frame_indices_with_issues") or []
    issue_frames = [clamp_int(frame, 0, 1_000_000) for frame in issue_frames]

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_scores": normalized_scores,
        "overall_artifact_severity": overall,
        "needs_review": overall >= 2,
        "highest_severity_categories": [
            name
            for name, item in normalized_scores.items()
            if item["severity"] == overall and overall > 0
        ],
        "frame_indices_with_issues": sorted(set(issue_frames)),
        "quality_notes": str(payload.get("quality_notes", "")),
    }


def parse_artifact_json(text: str) -> dict[str, Any]:
    return normalize_artifact_payload(extract_json_object(text))


def make_backend(args: argparse.Namespace) -> ArtifactBackend:
    if args.backend == "qwen-local":
        return QwenLocalBackend(
            model_id=args.model_id,
            device_map=args.device_map,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            trust_remote_code=args.trust_remote_code,
        )
    if args.backend == "openai":
        return OpenAIBackendPlaceholder()
    raise ValueError(f"unknown backend: {args.backend}")


def make_failure_payload(
    clip: ClipInput,
    *,
    error: BaseException,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "evaluated_at_utc": utc_now(),
        "relative_output_dir": clip.relative_output_dir,
        "output_dir": str(clip.output_dir),
        "cropped_video": str(clip.cropped_video),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "config": config,
    }


def evaluate_clip(
    clip: ClipInput,
    *,
    backend: ArtifactBackend,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_path = clip.output_dir / args.output_name
    contact_sheet_path = clip.output_dir / args.contact_sheet_name

    if output_path.exists() and not args.overwrite:
        payload = read_json(output_path)
        payload = {**payload, "status": "skipped"}
        return payload

    t0 = time.time()
    frame_indices, sheet_size = build_contact_sheet(
        clip.cropped_video,
        output_path=contact_sheet_path,
        sample_count=args.sample_frames,
        columns=args.sheet_columns,
        thumb_width=args.thumb_width,
        overwrite=args.overwrite_contact_sheets,
    )
    prompt = make_prompt(frame_indices)

    if args.prepare_only:
        artifact_payload = normalize_artifact_payload({})
        raw_response = ""
    else:
        result = backend.analyze(contact_sheet=contact_sheet_path, prompt=prompt)
        artifact_payload = result.parsed
        raw_response = result.raw_response

    payload = {
        "status": "ok",
        "evaluated_at_utc": utc_now(),
        "backend": backend.name,
        "model_id": args.model_id if backend.name == "qwen-local" else None,
        "relative_output_dir": clip.relative_output_dir,
        "output_dir": str(clip.output_dir),
        "metrics_json": str(clip.metrics_path),
        "source_video": str(clip.source_video) if clip.source_video else None,
        "cropped_video": str(clip.cropped_video),
        "contact_sheet": str(contact_sheet_path),
        "sampled_frame_indices": frame_indices,
        "contact_sheet_size": list(sheet_size),
        "artifacts": artifact_payload,
        "raw_response": raw_response,
        "timings": {"total_seconds": time.time() - t0},
        "config": config,
    }
    write_json(output_path, payload)
    return payload


def summarize_record(payload: dict[str, Any], output_name: str) -> dict[str, Any]:
    artifacts = payload.get("artifacts") or {}
    scores = artifacts.get("artifact_scores") or {}
    return {
        "status": payload.get("status"),
        "relative_output_dir": payload.get("relative_output_dir"),
        "vlm_artifacts_json": str(Path(payload.get("output_dir", "")) / output_name),
        "contact_sheet": payload.get("contact_sheet"),
        "overall_artifact_severity": artifacts.get("overall_artifact_severity"),
        "needs_review": artifacts.get("needs_review"),
        "highest_severity_categories": artifacts.get("highest_severity_categories", []),
        "artifact_scores": {
            name: {
                "severity": item.get("severity"),
                "confidence": item.get("confidence"),
            }
            for name, item in scores.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate targeted OmniDreams video artifacts with a pluggable VLM backend."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metrics-json-name", default="metrics.json")
    parser.add_argument("--output-name", default="vlm_artifacts.json")
    parser.add_argument("--contact-sheet-name", default="vlm_contact_sheet.jpg")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Defaults to <root>/vlm_artifacts_summary.json.",
    )
    parser.add_argument("--backend", choices=["qwen-local", "openai"], default="qwen-local")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--sample-frames", type=int, default=12)
    parser.add_argument("--sheet-columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-contact-sheets", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build contact sheets and placeholder JSON; do not load a VLM.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = parse_args()
    root = args.root.expanduser().resolve()
    summary_path = args.summary_path or root / "vlm_artifacts_summary.json"
    clips = discover_clips(root, args.metrics_json_name)
    if args.limit is not None:
        clips = clips[: args.limit]
    if not clips:
        raise ValueError(f"No {args.metrics_json_name!r} files found under {root}")

    config = vars(args).copy()
    config["root"] = str(root)
    config["summary_path"] = str(summary_path)

    print(f"Found {len(clips)} clip(s) under {root}")
    print(f"Backend: {args.backend}")
    if args.backend == "qwen-local" and not args.prepare_only:
        print(f"Model: {args.model_id}")

    backend = make_backend(args)

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = tqdm(clips, desc="VLM artifact eval", unit="clip") if tqdm else clips
    started = time.time()
    records: list[dict[str, Any]] = []
    status_counts = {"ok": 0, "failed": 0, "skipped": 0}

    for clip in iterator:
        try:
            payload = evaluate_clip(
                clip,
                backend=backend,
                args=args,
                config=config,
            )
            status = payload.get("status", "ok")
            if status == "skipped":
                status_counts["skipped"] += 1
            else:
                status_counts["ok"] += 1
            records.append(summarize_record(payload, args.output_name))
            artifacts = payload.get("artifacts") or {}
            severity = artifacts.get("overall_artifact_severity", "-")
            categories = ",".join(artifacts.get("highest_severity_categories", []))
            print(f"{clip.relative_output_dir}: severity={severity} {categories}")
        except Exception as exc:
            failure = make_failure_payload(clip, error=exc, config=config)
            write_json(clip.output_dir / args.output_name, failure)
            status_counts["failed"] += 1
            records.append(summarize_record(failure, args.output_name))
            if not args.keep_going:
                raise
            print(
                f"{clip.relative_output_dir}: failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    summary = {
        "status": "ok" if status_counts["failed"] == 0 else "failed",
        "evaluated_at_utc": utc_now(),
        "root": str(root),
        "elapsed_seconds": time.time() - started,
        "status_counts": status_counts,
        "artifact_categories": ARTIFACT_CATEGORIES,
        "records": sorted(
            records,
            key=lambda item: (
                -(item.get("overall_artifact_severity") or 0),
                item.get("relative_output_dir") or "",
            ),
        ),
        "config": config,
    }
    write_json(summary_path, summary)
    print(f"Summary saved to {summary_path}")
    print(f"Status counts: {status_counts}")
    return 0 if status_counts["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
