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

"""Machine-local VP9 benchmark and automatic VP8 fallback policy."""

import json
import os
import platform
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from flashdreams_webm._native import WebmWriter, versions

Codec = Literal["vp8", "vp9"]

_SCHEMA_VERSION = 1
"""Cache schema for invalidating incompatible benchmark records."""

_BENCHMARK_WIDTH = 768
"""Width required by the FlashDreams WebM codec-selection benchmark."""

_BENCHMARK_HEIGHT = 768
"""Height required by the FlashDreams WebM codec-selection benchmark."""

_BENCHMARK_FRAMES_PER_SECOND = 24
"""Frame rate required by the FlashDreams WebM codec-selection benchmark."""

_BENCHMARK_WARMUP_FRAMES = 8
"""Frames excluded while libvpx fills its initial encoder state."""

_BENCHMARK_MEASURED_FRAMES = 24
"""Frames retained for the codec latency decision."""

_CODEC_OVERRIDE = "FLASHDREAMS_WEBM_CODEC"
"""Environment variable forcing ``vp8`` or ``vp9`` without benchmarking."""


def benchmark_video_codec(
    codec: Codec,
    *,
    width: int = _BENCHMARK_WIDTH,
    height: int = _BENCHMARK_HEIGHT,
    frames_per_second: int = _BENCHMARK_FRAMES_PER_SECOND,
    warmup_frames: int = _BENCHMARK_WARMUP_FRAMES,
    measured_frames: int = _BENCHMARK_MEASURED_FRAMES,
) -> dict[str, Any]:
    """Measure RGB conversion and native VPx latency one frame at a time.

    Args:
        codec: Native video codec to measure.
        width: Benchmark frame width in pixels.
        height: Benchmark frame height in pixels.
        frames_per_second: Playback rate and per-frame latency budget.
        warmup_frames: Initial frames excluded from reported latency.
        measured_frames: Frames included in reported latency.

    Returns:
        Machine-readable settings, raw frame latencies, median, and p90.

    Raises:
        ValueError: A setting is non-positive or ``codec`` is unsupported.
        RuntimeError: Native encoding or finalization fails.
    """
    if codec not in ("vp8", "vp9"):
        raise ValueError("codec must be 'vp8' or 'vp9'")
    for name, value in (
        ("width", width),
        ("height", height),
        ("frames_per_second", frames_per_second),
        ("warmup_frames", warmup_frames),
        ("measured_frames", measured_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if width % 2 or height % 2:
        raise ValueError("benchmark frame dimensions must be even")

    frames = _benchmark_frames(width, height)
    latencies_ms: list[float] = []
    with tempfile.TemporaryDirectory(prefix="flashdreams-webm-benchmark-") as temp:
        writer = WebmWriter(
            Path(temp) / f"{codec}.webm",
            width,
            height,
            frames_per_second,
            codec,
        )
        try:
            total_frames = warmup_frames + measured_frames
            for frame_index in range(total_frames):
                started_ns = time.perf_counter_ns()
                writer.write_video(frames[frame_index % len(frames)])
                elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
                if frame_index >= warmup_frames:
                    latencies_ms.append(elapsed_ms)
            writer.close()
        except BaseException:
            writer.abort()
            raise

    frame_budget_ms = 1000.0 / frames_per_second
    return {
        "codec": codec,
        "width": width,
        "height": height,
        "frames_per_second": frames_per_second,
        "warmup_frames": warmup_frames,
        "measured_frames": measured_frames,
        "latencies_ms": latencies_ms,
        "median_ms": statistics.median(latencies_ms),
        "p90_ms": _percentile(latencies_ms, 0.90),
        "frame_budget_ms": frame_budget_ms,
    }


def codec_selection(
    *, refresh: bool = False, cache_path: str | Path | None = None
) -> dict[str, Any]:
    """Return the cached or newly benchmarked native video-codec decision.

    VP9 remains selected while its p90 encode latency fits one 24-fps frame
    interval. A slower result selects VP8. The decision is cached against the
    native library versions and local CPU description.

    Args:
        refresh: Ignore a compatible cached record and benchmark again.
        cache_path: JSON record location; ``None`` uses the user cache.

    Returns:
        Codec decision and its benchmark evidence.

    Raises:
        ValueError: ``FLASHDREAMS_WEBM_CODEC`` names an unsupported codec.
        RuntimeError: The native VP9 benchmark fails.
    """
    override = os.environ.get(_CODEC_OVERRIDE)
    if override is not None:
        if override not in ("vp8", "vp9"):
            raise ValueError(
                f"{_CODEC_OVERRIDE} must be 'vp8' or 'vp9', got {override!r}"
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "codec": override,
            "source": "environment",
            "environment_variable": _CODEC_OVERRIDE,
            "fingerprint": _fingerprint(),
        }

    resolved_cache_path = (
        _default_cache_path() if cache_path is None else Path(cache_path)
    )
    fingerprint = _fingerprint()
    if not refresh:
        cached = _read_cache(resolved_cache_path)
        if (
            cached is not None
            and cached.get("schema_version") == _SCHEMA_VERSION
            and cached.get("fingerprint") == fingerprint
            and cached.get("codec") in ("vp8", "vp9")
        ):
            cached["source"] = "cache"
            return cached

    benchmark = benchmark_video_codec("vp9")
    excessive = benchmark["p90_ms"] > benchmark["frame_budget_ms"]
    selected: Codec = "vp8" if excessive else "vp9"
    record: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "codec": selected,
        "source": "benchmark",
        "status": "recommended" if selected == "vp9" else "useful fallback",
        "reason": (
            "VP9 p90 exceeded one 24-fps frame interval."
            if excessive
            else "VP9 p90 fit within one 24-fps frame interval."
        ),
        "excessive_latency": excessive,
        "fingerprint": fingerprint,
        "benchmark": benchmark,
        "command": "flashdreams-webm-benchmark --refresh",
        "measured_at_unix_seconds": time.time(),
    }
    _write_cache(resolved_cache_path, record)
    return record


def select_video_codec(
    *, refresh: bool = False, cache_path: str | Path | None = None
) -> Codec:
    """Select VP9 unless the required benchmark makes VP8 the fallback.

    Args:
        refresh: Ignore a compatible cached benchmark.
        cache_path: JSON record location; ``None`` uses the user cache.

    Returns:
        ``"vp9"`` or ``"vp8"``.
    """
    return cast(Codec, codec_selection(refresh=refresh, cache_path=cache_path)["codec"])


def _benchmark_frames(width: int, height: int) -> tuple[bytes, ...]:
    """Return deterministic spatial patterns with inter-frame motion."""
    frame_bytes = width * height * 3
    base = bytes(range(256))
    repeats = (frame_bytes + len(base) - 1) // len(base)
    frames: list[bytes] = []
    for offset in (0, 37, 91, 149):
        shifted = base[offset:] + base[:offset]
        frames.append((shifted * repeats)[:frame_bytes])
    return tuple(frames)


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a nearest-rank percentile from a non-empty sample."""
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * percentile + 0.999999999))
    return ordered[min(rank, len(ordered)) - 1]


def _fingerprint() -> dict[str, Any]:
    """Return native and CPU facts that invalidate stale benchmark choices."""
    return {
        "native_versions": versions(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "benchmark": {
            "width": _BENCHMARK_WIDTH,
            "height": _BENCHMARK_HEIGHT,
            "frames_per_second": _BENCHMARK_FRAMES_PER_SECOND,
            "warmup_frames": _BENCHMARK_WARMUP_FRAMES,
            "measured_frames": _BENCHMARK_MEASURED_FRAMES,
        },
    }


def _default_cache_path() -> Path:
    """Return the per-user machine benchmark record path."""
    cache_root = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_root) if cache_root else Path.home() / ".cache"
    return base / "flashdreams" / "webm-codec-v1.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    """Read a benchmark record, treating absence or corruption as a miss."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_cache(path: Path, record: dict[str, Any]) -> None:
    """Best-effort atomically persist a machine-local benchmark record."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["benchmark_video_codec", "codec_selection", "select_video_codec"]
