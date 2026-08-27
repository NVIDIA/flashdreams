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

"""Run the LTX 2.5 validation matrix and build a self-contained HTML index."""

import argparse
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch

from flashdreams.runtime_v2.metrics_output_sink import MetricsOutputSink
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_runner import run_session

from .app import LTX25Application
from .backend import (
    DEFAULT_AUDIO_CHANNELS,
    DEFAULT_AUDIO_SAMPLE_RATE,
    MODEL_ID,
    MODEL_REVISION,
    OffloadMode,
)

_FRAMES_PER_SECOND = 24
_AAC_FRAME_S = 1024 / DEFAULT_AUDIO_SAMPLE_RATE


@dataclass(frozen=True, kw_only=True, slots=True)
class MatrixCase:
    """One prompt, duration, resolution, and seed in the validation matrix."""

    label: str
    """Filesystem-safe stable case identifier."""

    prompt: str
    """Joint audio-video conditioning prompt."""

    num_frames: int
    """Requested frames on the LTX temporal grid."""

    width: int
    """Requested output width."""

    height: int
    """Requested output height."""

    seed: int
    """Deterministic generation seed."""


DEFAULT_CASES = (
    MatrixCase(
        label="sync_percussion_short",
        prompt=(
            "Locked wide shot of a jazz drummer in a warm studio. The drummer "
            "strikes the snare three distinct times, with each visible stick impact "
            "matched by a crisp drum transient; subtle room ambience, no music."
        ),
        num_frames=25,
        width=768,
        height=512,
        seed=101,
    ),
    MatrixCase(
        label="dialogue_portrait",
        prompt=(
            "Close portrait of a woman at a quiet cafe saying, 'The last train "
            "leaves at midnight.' Natural lip motion, clear speech, soft cups and "
            "room tone behind her, stable camera."
        ),
        num_frames=25,
        width=960,
        height=544,
        seed=202,
    ),
    MatrixCase(
        label="ocean_wide",
        prompt=(
            "Cinematic wide coastline at sunrise as one wave breaks across dark "
            "rocks; matching surf crash, gull calls, and steady wind, detailed "
            "water motion, no narration."
        ),
        num_frames=25,
        width=1280,
        height=736,
        seed=303,
    ),
    MatrixCase(
        label="train_medium",
        prompt=(
            "Side tracking shot of a red steam train crossing a snowy valley. "
            "Wheels turn with a rhythmic rail clatter, steam hisses, then one horn "
            "sounds as the train passes the camera."
        ),
        num_frames=121,
        width=768,
        height=512,
        seed=404,
    ),
    MatrixCase(
        label="market_medium",
        prompt=(
            "Handheld walk through a busy open-air night market, vendors speaking, "
            "footsteps, pans sizzling, fabric awnings moving in the breeze, coherent "
            "forward motion and layered natural ambience."
        ),
        num_frames=121,
        width=960,
        height=544,
        seed=505,
    ),
    MatrixCase(
        label="multishot_max",
        prompt=(
            "A connected three-shot sequence: rain tapping on a greenhouse roof, "
            "a gardener opens the glass door with a squeak, then walks inside and "
            "sets metal shears on a wooden table. Continuous rain ambience and "
            "precise sounds across every cut."
        ),
        num_frames=241,
        width=768,
        height=512,
        seed=606,
    ),
)
"""Prompt, temporal, and spatial diversity matrix validated by this adapter."""


def run_matrix(
    *,
    output_dir: Path,
    cases: Sequence[MatrixCase],
    device: str,
    offload: OffloadMode,
    local_files_only: bool,
    overwrite: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    """Generate, inspect, and index selected cases while loading LTX only once.

    Args:
        output_dir: Directory receiving MP4, stats, manifest, and gallery files.
        cases: Cases to generate in order.
        device: Torch inference device.
        offload: Diffusers CPU-offload policy.
        local_files_only: Whether model loading may access the network.
        overwrite: Whether existing case artifacts may be replaced.
        continue_on_error: Whether a failed case leaves evidence and advances.

    Returns:
        Complete serializable validation manifest.

    Raises:
        FileExistsError: A selected output already exists without overwrite.
        RuntimeError: A required media program is missing or a case fails when
            continuation was not requested.
    """
    _require_programs()
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_targets(output_dir, cases, overwrite=overwrite)
    manifest = _new_manifest(device=device, offload=offload, cases=cases)
    app = LTX25Application()
    try:
        for case in cases:
            print(
                f"[{case.label}] {case.num_frames} frames at "
                f"{case.width}x{case.height}",
                flush=True,
            )
            record = _case_record(case)
            manifest["cases"].append(record)
            video_path = output_dir / f"{case.label}.mp4"
            stats_path = output_dir / f"{case.label}-stats.json"
            started_at = time.perf_counter()
            failure: BaseException | None = None
            try:
                app.init(
                    _application_args(
                        case,
                        device=device,
                        offload=offload,
                        local_files_only=local_files_only,
                    )
                )
                desc = replace(
                    app.session_desc(),
                    video_width=case.width,
                    video_height=case.height,
                    frames_per_second_for_step=_FRAMES_PER_SECOND,
                )
                session = app.create_session(desc)
                run_session(
                    session,
                    Mp4ClientWindow(video_path),
                    metrics_output_sink=MetricsOutputSink(stats_path),
                )
                record.update(
                    inspect_artifact(
                        video_path=video_path,
                        stats_path=stats_path,
                        case=case,
                    )
                )
                failed_checks = [
                    name for name, passed in record["checks"].items() if not passed
                ]
                if failed_checks:
                    raise RuntimeError(
                        f"{case.label} failed media checks: " + ", ".join(failed_checks)
                    )
                record["status"] = "passed"
            except BaseException as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                failure = error
            finally:
                record["wall_s"] = time.perf_counter() - started_at
                _write_indexes(output_dir, manifest)
            print(
                f"[{case.label}] {record['status']} in {record['wall_s']:.2f} s",
                flush=True,
            )
            if failure is not None and not continue_on_error:
                raise failure
    finally:
        app.close()
        _write_indexes(output_dir, manifest)
    return manifest


def inspect_artifact(
    *,
    video_path: Path,
    stats_path: Path,
    case: MatrixCase,
) -> dict[str, Any]:
    """Measure codecs, timing, signal, motion, runtime stats, size, and hash."""
    probe = _probe(video_path)
    streams = probe.get("streams", [])
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
    video_duration_s = _optional_float(video.get("duration"))
    audio_duration_s = _optional_float(audio.get("duration"))
    av_drift_s = (
        None
        if video_duration_s is None or audio_duration_s is None
        else abs(video_duration_s - audio_duration_s)
    )
    audio_signal = _audio_signal(video_path)
    video_signal = _video_signal(video_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    metrics = {
        str(sample["name"]): sample["value"] for sample in stats.get("samples", [])
    }
    checks = {
        "h264_video": video.get("codec_name") == "h264",
        "aac_audio": audio.get("codec_name") == "aac",
        "requested_dimensions": (
            int(video.get("width", -1)) == case.width
            and int(video.get("height", -1)) == case.height
        ),
        "exact_video_frames": (
            int(video.get("nb_read_frames", -1)) == case.num_frames
            and video_signal["decoded_frames"] == case.num_frames
        ),
        "stereo_48khz": (
            int(audio.get("channels", -1)) == DEFAULT_AUDIO_CHANNELS
            and int(audio.get("sample_rate", -1)) == DEFAULT_AUDIO_SAMPLE_RATE
        ),
        "finite_audio": bool(audio_signal["finite"]),
        "audible_audio": float(audio_signal["rms"]) > 1e-4,
        "changing_video": float(video_signal["frame_delta_mean"]) > 0.1,
        "av_drift_within_aac_frame": (
            av_drift_s is not None and av_drift_s <= _AAC_FRAME_S + 1e-3
        ),
    }
    return {
        "video_path": video_path.name,
        "stats_path": stats_path.name,
        "sha256": _sha256(video_path),
        "bytes": video_path.stat().st_size,
        "metrics": metrics,
        "media": {
            "video_codec": video.get("codec_name"),
            "audio_codec": audio.get("codec_name"),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "encoded_frames": int(video["nb_read_frames"]),
            "video_duration_s": video_duration_s,
            "audio_duration_s": audio_duration_s,
            "av_drift_s": av_drift_s,
            "audio_sample_rate": int(audio["sample_rate"]),
            "audio_channels": int(audio["channels"]),
        },
        "audio_signal": audio_signal,
        "video_signal": video_signal,
        "checks": checks,
    }


def render_gallery(manifest: dict[str, Any]) -> str:
    """Return a portable HTML gallery for one validation manifest."""
    cards = "\n".join(_gallery_card(record) for record in manifest.get("cases", []))
    generated_at = html.escape(str(manifest.get("generated_at", "")))
    model_id = html.escape(str(manifest.get("model_id", MODEL_ID)))
    revision = html.escape(str(manifest.get("model_revision", MODEL_REVISION)))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LTX 2.5 audio-video validation gallery</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
body {{ margin: 0; background: #0b0f14; color: #e8edf2; }}
header {{ padding: 2rem clamp(1rem, 5vw, 4rem); background: #111923; }}
h1 {{ margin: 0 0 .4rem; font-size: clamp(1.6rem, 4vw, 2.8rem); }}
.meta {{ color: #aab8c5; overflow-wrap: anywhere; }}
main {{ display: grid; gap: 1.25rem; padding: 1.25rem; grid-template-columns:
repeat(auto-fit, minmax(min(100%, 32rem), 1fr)); }}
article {{ background: #151d27; border: 1px solid #2b3947; border-radius: .8rem;
overflow: hidden; box-shadow: 0 .4rem 1.4rem #0006; }}
video {{ width: 100%; aspect-ratio: 16 / 9; background: #000; display: block; }}
.body {{ padding: 1rem 1.1rem 1.2rem; }}
.title {{ display: flex; justify-content: space-between; gap: 1rem; align-items: center; }}
h2 {{ margin: 0; font-size: 1.1rem; }}
.status {{ padding: .25rem .55rem; border-radius: 99rem; background: #334154;
font-size: .78rem; text-transform: uppercase; }}
.passed {{ background: #14532d; }} .failed {{ background: #7f1d1d; }}
.prompt {{ color: #c6d0da; line-height: 1.45; }}
dl {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .55rem; }}
dt {{ color: #8293a4; font-size: .75rem; text-transform: uppercase; }}
dd {{ margin: .1rem 0 0; font-variant-numeric: tabular-nums; }}
a {{ color: #7dd3fc; }}
</style>
</head>
<body>
<header><h1>LTX 2.5 synchronized audio-video validation</h1>
<div class="meta">{model_id} @ {revision}<br>Generated {generated_at} ·
<a href="manifest.json">machine-readable manifest</a></div></header>
<main>{cards}</main>
</body>
</html>
"""


def _gallery_card(record: dict[str, Any]) -> str:
    """Render one escaped gallery card."""
    label = html.escape(str(record.get("label", "unknown")))
    status_value = str(record.get("status", "pending"))
    status = html.escape(status_value)
    prompt = html.escape(str(record.get("prompt", "")))
    video_path = record.get("video_path")
    video = (
        '<div class="body">No playable artifact.</div>'
        if not video_path
        else (
            f'<video controls preload="metadata" src="{quote(str(video_path))}">'
            "Your browser cannot play this MP4.</video>"
        )
    )
    metrics = record.get("metrics", {})
    generation_s = _format_number(metrics.get("generation_s"), " s")
    generation_fps = _format_number(metrics.get("generation_fps"), " fps")
    peak_memory = _format_number(metrics.get("peak_cuda_memory_gib"), " GiB")
    drift = _format_number(record.get("media", {}).get("av_drift_s"), " s")
    frames = html.escape(str(record.get("num_frames", "—")))
    dimensions = html.escape(f"{record.get('width', '—')}×{record.get('height', '—')}")
    error = record.get("error")
    error_html = "" if error is None else f"<p>{html.escape(str(error))}</p>"
    return f"""<article>
{video}
<div class="body">
<div class="title"><h2>{label}</h2><span class="status {status}">{status}</span></div>
<p class="prompt">{prompt}</p>{error_html}
<dl>
<div><dt>Output</dt><dd>{frames} frames · {dimensions}</dd></div>
<div><dt>Generation</dt><dd>{generation_s}</dd></div>
<div><dt>Throughput</dt><dd>{generation_fps}</dd></div>
<div><dt>Peak CUDA</dt><dd>{peak_memory}</dd></div>
<div><dt>A/V drift</dt><dd>{drift}</dd></div>
<div><dt>Seed</dt><dd>{html.escape(str(record.get("seed", "—")))}</dd></div>
</dl>
</div></article>"""


def _application_args(
    case: MatrixCase,
    *,
    device: str,
    offload: OffloadMode,
    local_files_only: bool,
) -> list[str]:
    """Return one case's model arguments."""
    arguments = [
        "--prompt",
        case.prompt,
        "--num-frames",
        str(case.num_frames),
        "--seed",
        str(case.seed),
        "--device",
        device,
        "--offload",
        offload,
    ]
    if local_files_only:
        arguments.append("--local-files-only")
    return arguments


def _new_manifest(
    *,
    device: str,
    offload: OffloadMode,
    cases: Sequence[MatrixCase],
) -> dict[str, Any]:
    """Return manifest metadata known before model loading."""
    return {
        "schema_version": 1,
        "artifact_type": "flashdreams.ltx25.validation_gallery",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "flashdreams_commit": _git_commit(),
        "device": device,
        "offload": offload,
        "selected_labels": [case.label for case in cases],
        "cases": [],
    }


def _case_record(case: MatrixCase) -> dict[str, Any]:
    """Return serializable immutable inputs and pending status for one case."""
    return {
        **asdict(case),
        "frames_per_second": _FRAMES_PER_SECOND,
        "requested_duration_s": case.num_frames / _FRAMES_PER_SECOND,
        "status": "running",
    }


def _write_indexes(output_dir: Path, manifest: dict[str, Any]) -> None:
    """Atomically publish the JSON manifest and regenerate its HTML view."""
    _write_text_atomic(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(output_dir / "gallery.html", render_gallery(manifest))


def _write_text_atomic(path: Path, contents: str) -> None:
    """Replace a UTF-8 text artifact after its complete temporary is written."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _check_targets(
    output_dir: Path,
    cases: Sequence[MatrixCase],
    *,
    overwrite: bool,
) -> None:
    """Reject accidental replacement of selected media and stats artifacts."""
    if overwrite:
        return
    existing = [
        path
        for case in cases
        for path in (
            output_dir / f"{case.label}.mp4",
            output_dir / f"{case.label}-stats.json",
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to replace existing artifacts without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _require_programs() -> None:
    """Require both external programs used for encoding and inspection."""
    missing = [
        program for program in ("ffmpeg", "ffprobe") if shutil.which(program) is None
    ]
    if missing:
        raise RuntimeError("Missing required program(s): " + ", ".join(missing))


def _probe(path: Path) -> dict[str, Any]:
    """Return FFprobe stream and container metadata."""
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def _audio_signal(path: Path) -> dict[str, float | int | bool]:
    """Decode stereo float PCM and return finite, peak, and RMS measurements."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(DEFAULT_AUDIO_CHANNELS),
            "-ar",
            str(DEFAULT_AUDIO_SAMPLE_RATE),
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    samples = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    finite = bool(torch.isfinite(samples).all())
    rms = float(torch.sqrt(torch.mean(samples.square()))) if samples.numel() else 0.0
    peak = float(samples.abs().max()) if samples.numel() else 0.0
    return {
        "decoded_samples_per_channel": samples.numel() // DEFAULT_AUDIO_CHANNELS,
        "finite": finite,
        "rms": rms,
        "peak": peak,
    }


def _video_signal(path: Path) -> dict[str, float | int]:
    """Decode small grayscale frames and measure nonblank content and motion."""
    width, height = 160, 90
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:-1:-1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    frame_pixels = width * height
    if not pixels.numel() or pixels.numel() % frame_pixels:
        raise RuntimeError("Decoded video does not contain complete analysis frames.")
    frames = pixels.reshape(-1, frame_pixels).to(torch.float32)
    frame_delta_mean = (
        float((frames[1:] - frames[:-1]).abs().mean()) if len(frames) > 1 else 0.0
    )
    return {
        "decoded_frames": len(frames),
        "luma_mean": float(frames.mean()),
        "luma_std": float(frames.std()),
        "frame_delta_mean": frame_delta_mean,
    }


def _sha256(path: Path) -> str:
    """Return the complete SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    """Return the current repository commit when invoked from a worktree."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _optional_float(value: Any) -> float | None:
    """Return a finite float for FFprobe text, or none when unavailable."""
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _format_number(value: object, suffix: str) -> str:
    """Format a finite gallery measurement or return an em dash."""
    converted = _optional_float(value)
    return "—" if converted is None else f"{converted:.3f}{suffix}"


def _parser() -> argparse.ArgumentParser:
    """Return the command-line parser for repeatable matrix runs."""
    labels = tuple(case.label for case in DEFAULT_CASES)
    parser = argparse.ArgumentParser(
        prog="flashdreams-ltx25-benchmark",
        description="Run LTX 2.5 cases with one model load and build an HTML gallery.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        choices=labels,
        dest="cases",
        help="Case to run; repeat it for several. Defaults to the complete matrix.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--offload",
        choices=("model", "sequential", "none"),
        default="model",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run selected cases and report the gallery and manifest paths."""
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    wanted = set(args.cases or ())
    cases = (
        DEFAULT_CASES
        if not wanted
        else tuple(case for case in DEFAULT_CASES if case.label in wanted)
    )
    manifest = run_matrix(
        output_dir=args.output_dir,
        cases=cases,
        device=args.device,
        offload=args.offload,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
    )
    passed = sum(case["status"] == "passed" for case in manifest["cases"])
    print(f"Passed {passed}/{len(cases)} cases.", flush=True)
    print(f"Gallery: {(args.output_dir / 'gallery.html').resolve()}", flush=True)
    print(f"Manifest: {(args.output_dir / 'manifest.json').resolve()}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CASES",
    "MatrixCase",
    "inspect_artifact",
    "main",
    "render_gallery",
    "run_matrix",
]
