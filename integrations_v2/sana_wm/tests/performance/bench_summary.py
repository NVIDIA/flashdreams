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

"""Summarize SANA-WM upstream and FlashDreams benchmark stats."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load_stats(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("run_*/stats.json"))
    if not paths and (root / "stats.json").exists():
        paths = [root / "stats.json"]
    items = []
    for path in paths:
        item = json.loads(path.read_text(encoding="utf-8")) | {"_path": str(path)}
        command_path = path.with_name("command.txt")
        if command_path.exists():
            item["_command"] = command_path.read_text(encoding="utf-8").strip()
        items.append(item)
    return items


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.9 * (len(ordered) - 1))))
    return ordered[index]


def _upstream_stage_ms(item: dict[str, Any], key: str) -> float | None:
    timings = item.get("timings_s")
    if not isinstance(timings, dict):
        return None
    for candidate in key.split("|"):
        value = timings.get(candidate)
        if isinstance(value, (int, float)):
            return float(value) * 1000.0
    return None


def _flashdreams_stage_ms(item: dict[str, Any], key: str) -> float | None:
    stats = item.get("stats_ms")
    if not isinstance(stats, dict):
        return None
    value = stats.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _upstream_mem_gib(item: dict[str, Any]) -> float | None:
    value = item.get("mem_peak_gib", item.get("peak_mem_gb"))
    return float(value) if isinstance(value, (int, float)) else None


def _flashdreams_mem_gib(item: dict[str, Any]) -> float | None:
    stats = item.get("stats_ms")
    if not isinstance(stats, dict):
        return None
    value = stats.get("mem_peak_gib")
    return float(value) if isinstance(value, (int, float)) else None


def _expand_generation_records(
    items: list[dict[str, Any]],
    warmup_generations: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for process_index, item in enumerate(items):
        generation_records = item.get("generation_records")
        if not isinstance(generation_records, list) or not generation_records:
            path = item.get("_path", "<unknown>")
            raise ValueError(
                "bidirectional benchmark stats must contain generation_records; "
                f"discard stale benchmark output and rerun bench.sh ({path})"
            )

        process_metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "generation_records",
                "wall_s",
                "timings_s",
                "stats_ms",
                "video_path",
                "video_shape",
            }
        }
        for fallback_index, generation in enumerate(generation_records):
            if not isinstance(generation, dict):
                continue
            raw_generation_index = generation.get("generation_index", fallback_index)
            generation_index = (
                int(raw_generation_index)
                if isinstance(raw_generation_index, int | float | str)
                else fallback_index
            )
            merged = process_metadata | generation
            parent_path = item.get("_path")
            if isinstance(parent_path, str):
                merged["_path"] = f"{parent_path}#generation_{generation_index}"
                merged["_process_path"] = parent_path
            merged["_command"] = item.get("_command")
            merged["process_index"] = process_index
            merged["generation_index"] = generation_index
            merged["_warmup"] = bool(
                generation.get("warmup", generation_index < warmup_generations)
            )
            records.append(merged)
    return records


def _collect_bidirectional(
    items: list[dict[str, Any]],
    warmup_generations: int,
    stage_reader,
    mem_reader,
    stages: dict[str, str],
) -> dict[str, Any]:
    records = _expand_generation_records(items, warmup_generations)
    kept = [item for item in records if not item.get("_warmup", False)]
    warmup_count = len(records) - len(kept)
    rows: dict[str, Any] = {
        "runs_total": len(records),
        "warmup_runs": warmup_count,
        "runs_measured": len(kept),
        "generations_total": len(records),
        "warmup_generations": warmup_count,
        "generations_measured": len(kept),
        "paths": [item.get("_path") for item in kept],
        "run_records": [
            _run_record(
                item,
                index=index,
                warmup=bool(item.get("_warmup", index < warmup_generations)),
            )
            for index, item in enumerate(records)
        ],
    }
    wall = [
        float(item["wall_s"])
        for item in kept
        if isinstance(item.get("wall_s"), (int, float))
    ]
    rows["wall_median_s"] = _median(wall)
    rows["wall_p90_s"] = _p90(wall)
    memory = [value for item in kept if (value := mem_reader(item)) is not None]
    rows["mem_peak_median_gib"] = _median(memory)
    rows["mem_peak_p90_gib"] = _p90(memory)
    for label, key in stages.items():
        values = [
            value for item in kept if (value := stage_reader(item, key)) is not None
        ]
        rows[f"{label}_median_ms"] = _median(values)
        rows[f"{label}_p90_ms"] = _p90(values)
    return rows


def _run_record(
    item: dict[str, Any],
    *,
    index: int,
    warmup: bool,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in item.items()
        if key not in {"_path", "_command", "_warmup"}
    }
    record["path"] = item.get("_path")
    record["command"] = item.get("_command")
    record["run_index"] = int(record.get("run_index", index))
    record["warmup"] = warmup
    return record


def _first_command(items: list[dict[str, Any]]) -> str | None:
    for item in items:
        value = item.get("_command")
        if isinstance(value, str) and value:
            return value
    return None


def _first_numeric(item: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _streaming_steady_state_chunk_ms(item: dict[str, Any]) -> float | None:
    steady_s = _first_numeric(item, "steady_state_seconds")
    if steady_s is None:
        return None
    chunks_value = item.get("n_decode_chunks")
    if not isinstance(chunks_value, int):
        return None
    # `steady_state_seconds` starts after the first decoded chunk, so exclude
    # that chunk from the denominator to avoid mixing cache-fill/compile latency
    # into the steady-state headline.
    measured_chunks = (
        chunks_value - 1
        if item.get("first_chunk_seconds") is not None
        else chunks_value
    )
    if measured_chunks <= 0:
        return None
    return steady_s * 1000.0 / measured_chunks


def _collect_streaming(
    items: list[dict[str, Any]],
    warmup_runs: int,
) -> dict[str, Any]:
    kept = items[warmup_runs:]
    rows: dict[str, Any] = {
        "runs_total": len(items),
        "warmup_runs": warmup_runs,
        "runs_measured": len(kept),
        "paths": [item.get("_path") for item in kept],
        "run_records": [
            _run_record(item, index=index, warmup=index < warmup_runs)
            for index, item in enumerate(items)
        ],
    }
    wall = [
        value
        for item in kept
        if (
            value := _first_numeric(
                item, "stream_wall_seconds", "wall_seconds", "wall_s"
            )
        )
        is not None
    ]
    rows["wall_median_s"] = _median(wall)
    rows["wall_p90_s"] = _p90(wall)
    end_to_end = [
        value
        for item in kept
        if (value := _first_numeric(item, "end_to_end_seconds")) is not None
    ]
    rows["end_to_end_median_s"] = _median(end_to_end)
    rows["end_to_end_p90_s"] = _p90(end_to_end)
    first_chunk = [
        value
        for item in kept
        if (value := _first_numeric(item, "first_chunk_seconds")) is not None
    ]
    rows["first_chunk_median_s"] = _median(first_chunk)
    rows["first_chunk_p90_s"] = _p90(first_chunk)
    chunk_ms = [
        value
        for item in kept
        if (value := _streaming_steady_state_chunk_ms(item)) is not None
    ]
    rows["steady_state_chunk_median_ms"] = _median(chunk_ms)
    rows["steady_state_chunk_p90_ms"] = _p90(chunk_ms)
    steady_fps = [
        value
        for item in kept
        if (value := _first_numeric(item, "steady_state_frames_per_second")) is not None
    ]
    rows["steady_state_fps_median"] = _median(steady_fps)
    rows["steady_state_fps_p90"] = _p90(steady_fps)
    realtime = [
        value
        for item in kept
        if (value := _first_numeric(item, "steady_state_realtime_factor")) is not None
    ]
    rows["steady_state_realtime_median"] = _median(realtime)
    rows["steady_state_realtime_p90"] = _p90(realtime)
    memory = [value for item in kept if (value := _upstream_mem_gib(item)) is not None]
    rows["mem_peak_median_gib"] = _median(memory)
    rows["mem_peak_p90_gib"] = _p90(memory)
    for label, key in {
        "stage1_cuda": "stage1_cuda_seconds",
        "refiner_cuda": "refiner_cuda_seconds",
        "decode_cuda": "decode_cuda_seconds",
    }.items():
        values = [
            value * 1000.0
            for item in kept
            if (value := _first_numeric(item, key)) is not None
        ]
        rows[f"{label}_median_ms"] = _median(values)
        rows[f"{label}_p90_ms"] = _p90(values)
    for label, key in {
        "n_pixel_frames": "n_pixel_frames",
        "n_decode_chunks": "n_decode_chunks",
        "n_refiner_blocks": "n_refiner_blocks",
    }.items():
        values = [
            value for item in kept if (value := _first_numeric(item, key)) is not None
        ]
        rows[f"{label}_median"] = _median(values)
        rows[f"{label}_p90"] = _p90(values)
    rows["artifact_paths"] = [
        item.get("output_path")
        for item in kept
        if isinstance(item.get("output_path"), str) and item.get("output_path")
    ]
    return rows


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _ms_per_frame(wall_s: float | None, num_frames: int) -> float | None:
    if wall_s is None or num_frames <= 0:
        return None
    return wall_s * 1000.0 / num_frames


def _ms_per_clip(wall_s: float | None) -> float | None:
    if wall_s is None:
        return None
    return wall_s * 1000.0


def _metric_value(summary: dict[str, Any], side: str, key: str) -> float | None:
    side_summary = summary.get(side)
    if not isinstance(side_summary, dict):
        return None
    value = side_summary.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _generation_ms_per_clip(
    summary: dict[str, Any],
    side: str,
    *,
    percentile: str = "median",
) -> float | None:
    return _ms_per_clip(_metric_value(summary, side, f"wall_{percentile}_s"))


def _streaming_generation_ms_per_chunk(
    summary: dict[str, Any],
    side: str,
    *,
    percentile: str = "median",
) -> float | None:
    side_summary = summary.get(side)
    if not isinstance(side_summary, dict):
        return None
    value = side_summary.get(f"steady_state_chunk_{percentile}_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _generation_ms_per_frame(
    summary: dict[str, Any],
    side: str,
    *,
    percentile: str = "median",
) -> float | None:
    num_frames = int(summary["inputs"]["num_frames"])
    return _ms_per_frame(
        _metric_value(summary, side, f"wall_{percentile}_s"),
        num_frames,
    )


def _sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def _render_bidirectional_markdown(summary: dict[str, Any]) -> str:
    upstream = summary.get("upstream")
    flashdreams = summary.get("flashdreams")
    has_upstream = isinstance(upstream, dict)
    has_flashdreams = isinstance(flashdreams, dict)
    refiner_scope = (
        "With `NO_REFINER=1`, the timed work covers conditioning, Stage-1 DiT, and SANA VAE decode."
        if summary["inputs"]["no_refiner"]
        else "With `NO_REFINER=0`, the timed work covers conditioning, Stage-1 DiT, LTX-2 refiner, and SANA VAE decode."
    )
    rows = [
        "# SANA-WM parity harness benchmark",
        "",
        "## Inputs",
        "",
        f"- image: `{summary['inputs']['image_path']}`",
        f"- prompt: `{summary['inputs']['prompt_path']}`",
        f"- camera: `{summary['inputs']['camera_path']}`",
        f"- intrinsics: `{summary['inputs']['intrinsics_path']}`",
        f"- num_frames: `{summary['inputs']['num_frames']}`",
        f"- seed: `{summary['inputs']['seed']}`",
        f"- variant: `{summary['inputs'].get('variant', 'bidirectional')}`",
        f"- bench_side: `{summary.get('bench_side', 'both')}`",
        f"- no_refiner: `{summary['inputs']['no_refiner']}`",
        f"- stage1_precision: `{summary['inputs']['stage1_precision']}`",
        f"- refiner_precision: `{summary['inputs']['refiner_precision']}`",
        f"- quant_backend: `{summary['inputs']['quant_backend']}`",
        f"- compile_stage1: `{summary['inputs']['compile_stage1']}`",
        f"- force_cudnn_sdpa: `{summary['inputs']['force_cudnn_sdpa']}`",
        f"- warmup generations discarded: `{summary['inputs']['warmup_runs']}`",
        "",
        "## Benchmark metric",
        "",
        "The chart metric is steady-state in-process generation latency per generated clip.",
        "Warmup generations run on the live model and are excluded from the headline metric.",
        "Model construction, checkpoint loading, video writing, and frame dumps are outside this timing boundary.",
        "SANA-WM renders each requested bidirectional clip in one generation pass, not as independently timed frames.",
        refiner_scope,
        "",
    ]
    if has_upstream and has_flashdreams:
        rows.extend(
            [
                "| metric | upstream | FlashDreams |",
                "| --- | ---: | ---: |",
                f"| measured generations | {upstream['generations_measured']} | {flashdreams['generations_measured']} |",
                f"| generation median / clip | {_fmt(_generation_ms_per_clip(summary, 'upstream'), ' ms')} | {_fmt(_generation_ms_per_clip(summary, 'flashdreams'), ' ms')} |",
                f"| generation p90 / clip | {_fmt(_generation_ms_per_clip(summary, 'upstream', percentile='p90'), ' ms')} | {_fmt(_generation_ms_per_clip(summary, 'flashdreams', percentile='p90'), ' ms')} |",
                f"| generation median / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'upstream'), ' ms')} | {_fmt(_generation_ms_per_frame(summary, 'flashdreams'), ' ms')} |",
                f"| generation p90 / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'upstream', percentile='p90'), ' ms')} | {_fmt(_generation_ms_per_frame(summary, 'flashdreams', percentile='p90'), ' ms')} |",
                f"| wall median | {_fmt(upstream['wall_median_s'], ' s')} | {_fmt(flashdreams['wall_median_s'], ' s')} |",
                f"| wall p90 | {_fmt(upstream['wall_p90_s'], ' s')} | {_fmt(flashdreams['wall_p90_s'], ' s')} |",
                "",
                "## Timing breakdown",
                "",
                "| stage | upstream median | FlashDreams median |",
                "| --- | ---: | ---: |",
                f"| Stage-1 incl. conditioning median | {_fmt(upstream['stage1_total_median_ms'], ' ms')} | {_fmt(_sum_optional(flashdreams['encode_median_ms'], flashdreams['dit_median_ms']), ' ms')} |",
                f"| Stage-1 DiT median | {_fmt(upstream['dit_median_ms'], ' ms')} | {_fmt(flashdreams['dit_median_ms'], ' ms')} |",
                f"| conditioning/encode median | n/a | {_fmt(flashdreams['encode_median_ms'], ' ms')} |",
            ]
        )
        if summary["inputs"]["no_refiner"]:
            # No refiner: FlashDreams `decode_ms` and upstream `vae_decode_s` are
            # both the pure SANA VAE decode, so they compare directly.
            rows.append(
                f"| VAE decode median | {_fmt(upstream['vae_decode_median_ms'], ' ms')} | {_fmt(flashdreams['vae_decode_median_ms'], ' ms')} |"
            )
        else:
            # Refiner enabled: both sides bundle the refiner denoise together
            # with its VAE decode into a single measurement (upstream
            # `refiner_s`, FlashDreams `decode_ms`). They are only
            # apples-to-apples as one combined row; the standalone upstream
            # `vae_decode_s` is the Stage-1 decode and is NOT comparable to
            # FlashDreams `decode_ms`.
            rows.append(
                f"| refiner + VAE decode median | {_fmt(upstream.get('refiner_median_ms'), ' ms')} | {_fmt(flashdreams['vae_decode_median_ms'], ' ms')} |"
            )
        rows.append(
            f"| peak memory median | {_fmt(upstream['mem_peak_median_gib'], ' GiB')} | {_fmt(flashdreams['mem_peak_median_gib'], ' GiB')} |"
        )
    elif has_upstream:
        rows.extend(
            [
                "| metric | upstream |",
                "| --- | ---: |",
                f"| measured generations | {upstream['generations_measured']} |",
                f"| generation median / clip | {_fmt(_generation_ms_per_clip(summary, 'upstream'), ' ms')} |",
                f"| generation p90 / clip | {_fmt(_generation_ms_per_clip(summary, 'upstream', percentile='p90'), ' ms')} |",
                f"| generation median / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'upstream'), ' ms')} |",
                f"| generation p90 / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'upstream', percentile='p90'), ' ms')} |",
                f"| wall median | {_fmt(upstream['wall_median_s'], ' s')} |",
                f"| wall p90 | {_fmt(upstream['wall_p90_s'], ' s')} |",
                "",
                "## Timing breakdown",
                "",
                "| stage | upstream median |",
                "| --- | ---: |",
                f"| Stage-1 incl. conditioning median | {_fmt(upstream['stage1_total_median_ms'], ' ms')} |",
                f"| Stage-1 DiT median | {_fmt(upstream['dit_median_ms'], ' ms')} |",
                f"| VAE decode median | {_fmt(upstream['vae_decode_median_ms'], ' ms')} |",
                f"| peak memory median | {_fmt(upstream['mem_peak_median_gib'], ' GiB')} |",
            ]
        )
    elif has_flashdreams:
        decode_label = (
            "VAE decode median"
            if summary["inputs"]["no_refiner"]
            else "refiner + VAE decode median"
        )
        rows.extend(
            [
                "| metric | FlashDreams |",
                "| --- | ---: |",
                f"| measured generations | {flashdreams['generations_measured']} |",
                f"| generation median / clip | {_fmt(_generation_ms_per_clip(summary, 'flashdreams'), ' ms')} |",
                f"| generation p90 / clip | {_fmt(_generation_ms_per_clip(summary, 'flashdreams', percentile='p90'), ' ms')} |",
                f"| generation median / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'flashdreams'), ' ms')} |",
                f"| generation p90 / frame, diagnostic | {_fmt(_generation_ms_per_frame(summary, 'flashdreams', percentile='p90'), ' ms')} |",
                f"| wall median | {_fmt(flashdreams['wall_median_s'], ' s')} |",
                f"| wall p90 | {_fmt(flashdreams['wall_p90_s'], ' s')} |",
                "",
                "## Timing breakdown",
                "",
                "| stage | FlashDreams median |",
                "| --- | ---: |",
                f"| Stage-1 incl. conditioning median | {_fmt(_sum_optional(flashdreams['encode_median_ms'], flashdreams['dit_median_ms']), ' ms')} |",
                f"| Stage-1 DiT median | {_fmt(flashdreams['dit_median_ms'], ' ms')} |",
                f"| conditioning/encode median | {_fmt(flashdreams['encode_median_ms'], ' ms')} |",
                f"| {decode_label} | {_fmt(flashdreams['vae_decode_median_ms'], ' ms')} |",
                f"| peak memory median | {_fmt(flashdreams['mem_peak_median_gib'], ' GiB')} |",
            ]
        )
    else:
        raise ValueError("cannot render bidirectional benchmark without any stats")
    rows.append("")
    return "\n".join(rows)


def _render_streaming_markdown(summary: dict[str, Any]) -> str:
    upstream = summary["upstream"]
    flashdreams = summary.get("flashdreams")
    has_flashdreams = isinstance(flashdreams, dict)
    title = (
        "# SANA-WM streaming benchmark"
        if has_flashdreams
        else "# SANA-WM streaming upstream benchmark"
    )
    rows = [
        title,
        "",
        "## Inputs",
        "",
        f"- image: `{summary['inputs']['image_path']}`",
        f"- prompt: `{summary['inputs']['prompt_path']}`",
        f"- camera_source: `{summary['inputs'].get('camera_source')}`",
        f"- camera: `{summary['inputs'].get('camera_path')}`",
        f"- action: `{summary['inputs'].get('action')}`",
        f"- intrinsics: `{summary['inputs']['intrinsics_path']}`",
        f"- requested_num_frames: `{summary['inputs']['num_frames']}`",
        f"- seed: `{summary['inputs']['seed']}`",
        f"- stage1_precision: `{summary['inputs']['stage1_precision']}`",
        f"- refiner_precision: `{summary['inputs']['refiner_precision']}`",
        f"- warmup runs discarded: `{summary['inputs']['warmup_runs']}`",
        f"- output_mode: `{summary['inputs'].get('output_mode')}`",
        "",
        "## Benchmark metric",
        "",
        "The headline metric is steady-state generation latency per produced chunk.",
        "Warmup runs and the first decoded chunk are excluded so cold compile and cache-fill do not enter the headline.",
        "Each sample is a separate Python process; discarded warmup samples prime process-external state, not a live model instance.",
        "Full-clip wall time is retained as supporting data.",
        "",
    ]
    if has_flashdreams:
        rows.extend(
            [
                "| metric | upstream | FlashDreams |",
                "| --- | ---: | ---: |",
                f"| measured runs | {upstream['runs_measured']} | {flashdreams['runs_measured']} |",
                f"| steady-state generation median / chunk | {_fmt(_streaming_generation_ms_per_chunk(summary, 'upstream'), ' ms')} | {_fmt(_streaming_generation_ms_per_chunk(summary, 'flashdreams'), ' ms')} |",
                f"| steady-state generation p90 / chunk | {_fmt(_streaming_generation_ms_per_chunk(summary, 'upstream', percentile='p90'), ' ms')} | {_fmt(_streaming_generation_ms_per_chunk(summary, 'flashdreams', percentile='p90'), ' ms')} |",
                f"| full-clip wall median | {_fmt(upstream['wall_median_s'], ' s')} | {_fmt(flashdreams['wall_median_s'], ' s')} |",
                f"| first chunk median | {_fmt(upstream['first_chunk_median_s'], ' s')} | {_fmt(flashdreams['first_chunk_median_s'], ' s')} |",
                f"| peak memory median | {_fmt(upstream['mem_peak_median_gib'], ' GiB')} | {_fmt(flashdreams['mem_peak_median_gib'], ' GiB')} |",
            ]
        )
    else:
        rows.extend(
            [
                "| metric | upstream |",
                "| --- | ---: |",
                f"| measured runs | {upstream['runs_measured']} |",
                f"| steady-state generation median / chunk | {_fmt(_streaming_generation_ms_per_chunk(summary, 'upstream'), ' ms')} |",
                f"| steady-state generation p90 / chunk | {_fmt(_streaming_generation_ms_per_chunk(summary, 'upstream', percentile='p90'), ' ms')} |",
                f"| full-clip wall median | {_fmt(upstream['wall_median_s'], ' s')} |",
                f"| full-clip wall p90 | {_fmt(upstream['wall_p90_s'], ' s')} |",
                f"| first chunk median | {_fmt(upstream['first_chunk_median_s'], ' s')} |",
                f"| generated frames median | {_fmt(upstream['n_pixel_frames_median'])} |",
                f"| decode chunks median | {_fmt(upstream['n_decode_chunks_median'])} |",
                f"| peak memory median | {_fmt(upstream['mem_peak_median_gib'], ' GiB')} |",
            ]
        )
    rows.extend(
        [
            "",
            "## Timing breakdown",
            "",
        ]
    )
    if has_flashdreams:
        rows.extend(
            [
                "| stage | upstream median | FlashDreams median |",
                "| --- | ---: | ---: |",
                f"| Stage-1 CUDA total | {_fmt(upstream['stage1_cuda_median_ms'], ' ms')} | {_fmt(flashdreams['stage1_cuda_median_ms'], ' ms')} |",
                f"| refiner CUDA total | {_fmt(upstream['refiner_cuda_median_ms'], ' ms')} | {_fmt(flashdreams['refiner_cuda_median_ms'], ' ms')} |",
                f"| decode CUDA total | {_fmt(upstream['decode_cuda_median_ms'], ' ms')} | {_fmt(flashdreams['decode_cuda_median_ms'], ' ms')} |",
            ]
        )
    else:
        rows.extend(
            [
                "| stage | upstream median |",
                "| --- | ---: |",
                f"| Stage-1 CUDA total | {_fmt(upstream['stage1_cuda_median_ms'], ' ms')} |",
                f"| refiner CUDA total | {_fmt(upstream['refiner_cuda_median_ms'], ' ms')} |",
                f"| decode CUDA total | {_fmt(upstream['decode_cuda_median_ms'], ' ms')} |",
            ]
        )
    rows.append("")
    return "\n".join(rows)


def _render_markdown(summary: dict[str, Any]) -> str:
    if (
        summary.get("variant") == "streaming"
        or summary["inputs"].get("variant") == "streaming"
    ):
        return _render_streaming_markdown(summary)
    return _render_bidirectional_markdown(summary)


def _render_chart_markdown(summary: dict[str, Any], device_label: str) -> str:
    streaming = (
        summary.get("variant") == "streaming"
        or summary["inputs"].get("variant") == "streaming"
    )
    official = (
        _streaming_generation_ms_per_chunk(summary, "upstream")
        if streaming
        else _generation_ms_per_clip(summary, "upstream")
    )
    flashdreams = (
        _streaming_generation_ms_per_chunk(summary, "flashdreams")
        if streaming
        else _generation_ms_per_clip(summary, "flashdreams")
    )
    if official is None and flashdreams is None:
        raise ValueError("cannot render chart data without benchmark stats")
    title = (
        "# SANA-WM Streaming Benchmark Data (ms/chunk)"
        if streaming
        else "# SANA-WM Benchmark Data (ms)"
    )
    if flashdreams is None:
        return "\n".join(
            [
                title,
                "",
                "| device | official |",
                "| --- | ---: |",
                f"| {device_label} | {official:.2f} |",
                "",
            ]
        )
    if official is None:
        return "\n".join(
            [
                title,
                "",
                "| device | flashdreams |",
                "| --- | ---: |",
                f"| {device_label} | {flashdreams:.2f} |",
                "",
            ]
        )
    return "\n".join(
        [
            title,
            "",
            "| device | official | flashdreams |",
            "| --- | ---: | ---: |",
            f"| {device_label} | {official:.2f} | {flashdreams:.2f} |",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=["bidirectional", "streaming"], default="bidirectional"
    )
    parser.add_argument(
        "--bench-side",
        choices=["upstream", "flashdreams", "both"],
        default="both",
    )
    parser.add_argument("--upstream-dir", type=Path, default=None)
    parser.add_argument("--flashdreams-dir", type=Path, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument(
        "--camera-source", choices=["camera", "action"], default="camera"
    )
    parser.add_argument("--action", default=None)
    parser.add_argument("--intrinsics-path", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--no-refiner", action="store_true")
    parser.add_argument(
        "--stage1-precision", choices=["bf16", "fp8", "fp4"], default="bf16"
    )
    parser.add_argument(
        "--refiner-precision", choices=["bf16", "fp8", "fp4"], default="bf16"
    )
    parser.add_argument(
        "--quant-backend",
        choices=["auto", "torch", "torch-fp8", "torch-fp4"],
        default="auto",
    )
    parser.add_argument("--compile-stage1", action="store_true")
    parser.add_argument("--force-cudnn-sdpa", action="store_true")
    parser.add_argument("--device-label", default="GPU")
    parser.add_argument("--chart-label", default=None)
    parser.add_argument("--upstream-commit", default=None)
    parser.add_argument("--flashdreams-commit", default=None)
    parser.add_argument("--output-mode", default=None)
    parser.add_argument("--denoising-step-list", default=None)
    parser.add_argument("--num-frame-per-block", type=int, default=None)
    parser.add_argument("--refiner-block-size", type=int, default=None)
    parser.add_argument("--refiner-kv-max-frames", type=int, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-chart-md", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.bench_side != "flashdreams" and args.upstream_dir is None:
        parser.error("--upstream-dir is required unless --bench-side=flashdreams")
    if args.bench_side != "upstream" and args.flashdreams_dir is None:
        parser.error("--flashdreams-dir is required unless --bench-side=upstream")
    if (
        args.variant == "bidirectional"
        and (args.stage1_precision != "bf16" or args.refiner_precision != "bf16")
        and args.bench_side != "flashdreams"
    ):
        parser.error(
            "upstream SANA-WM_bidirectional benchmarks are BF16-only; "
            "FP8 and FP4 precision flags are only supported by upstream "
            "SANA-WM_streaming. Use --bench-side flashdreams for FlashDreams-only "
            "bidirectional diagnostics, use --stage1-precision bf16 "
            "--refiner-precision bf16, or set --variant streaming."
        )
    if args.variant == "streaming" and args.bench_side == "flashdreams":
        parser.error("--bench-side=flashdreams is only implemented for bidirectional")

    upstream_items = (
        _load_stats(args.upstream_dir)
        if args.upstream_dir is not None and args.bench_side != "flashdreams"
        else []
    )
    flashdreams_items = (
        _load_stats(args.flashdreams_dir)
        if args.flashdreams_dir is not None and args.bench_side != "upstream"
        else []
    )
    if args.bench_side != "flashdreams" and not upstream_items:
        raise ValueError(f"no upstream stats found in {args.upstream_dir}")
    if args.bench_side != "upstream" and not flashdreams_items:
        raise ValueError(f"no FlashDreams stats found in {args.flashdreams_dir}")
    summary = {
        "variant": args.variant,
        "bench_side": args.bench_side,
        "upstream_commit": args.upstream_commit,
        "flashdreams_commit": args.flashdreams_commit,
        "commands": {
            "upstream": _first_command(upstream_items),
            "flashdreams": _first_command(flashdreams_items),
        },
        "inputs": {
            "image_path": str(args.image_path),
            "prompt_path": str(args.prompt_path),
            "camera_path": str(args.camera_path),
            "camera_source": args.camera_source,
            "action": args.action,
            "intrinsics_path": str(args.intrinsics_path),
            "num_frames": args.num_frames,
            "seed": args.seed,
            "variant": args.variant,
            "bench_side": args.bench_side,
            "no_refiner": args.no_refiner,
            "stage1_precision": args.stage1_precision,
            "refiner_precision": args.refiner_precision,
            "quant_backend": args.quant_backend,
            "compile_stage1": args.compile_stage1,
            "force_cudnn_sdpa": args.force_cudnn_sdpa,
            "warmup_runs": args.warmup_runs,
            "warmup_generations": (
                args.warmup_runs if args.variant == "bidirectional" else None
            ),
            "device_label": args.device_label,
            "chart_label": args.chart_label or args.device_label,
            "output_mode": args.output_mode,
            "denoising_step_list": args.denoising_step_list,
            "num_frame_per_block": args.num_frame_per_block,
            "refiner_block_size": args.refiner_block_size,
            "refiner_kv_max_frames": args.refiner_kv_max_frames,
        },
    }
    if args.variant == "streaming":
        summary["upstream"] = _collect_streaming(upstream_items, args.warmup_runs)
        if args.bench_side == "both":
            summary["flashdreams"] = _collect_streaming(
                flashdreams_items, args.warmup_runs
            )
        summary["benchmark"] = {
            "metric": "steady_state_generation_ms_per_chunk",
            "unit": "ms",
            "timing_boundary": (
                "streaming pipeline steady state after warmup runs and after the first "
                "decoded chunk; excludes model construction and checkpoint loading"
            ),
            "device_label": args.device_label,
            "chart_label": args.chart_label or args.device_label,
            "official": _streaming_generation_ms_per_chunk(summary, "upstream"),
            "official_p90": _streaming_generation_ms_per_chunk(
                summary, "upstream", percentile="p90"
            ),
            "flashdreams": _streaming_generation_ms_per_chunk(summary, "flashdreams")
            if args.bench_side == "both"
            else None,
            "full_clip_wall_median_s": summary["upstream"]["wall_median_s"],
            "variant": args.variant,
            "precision": args.stage1_precision,
        }
    else:
        if args.bench_side != "flashdreams":
            summary["upstream"] = _collect_bidirectional(
                upstream_items,
                args.warmup_runs,
                _upstream_stage_ms,
                _upstream_mem_gib,
                {
                    "dit": "stage1_dit_s|stage1_sample_s",
                    "stage1_total": "stage1_sample_s",
                    "refiner": "refiner_s",
                    "vae_decode": "vae_decode_s",
                },
            )
        if args.bench_side != "upstream":
            summary["flashdreams"] = _collect_bidirectional(
                flashdreams_items,
                args.warmup_runs,
                _flashdreams_stage_ms,
                _flashdreams_mem_gib,
                {
                    "encode": "encode_ms",
                    "dit": "diffuse_ms",
                    "vae_decode": "decode_ms",
                },
            )
        summary["benchmark"] = {
            "metric": "steady_state_generation_ms_per_clip",
            "unit": "ms",
            "timing_boundary": (
                "pipeline.generate after model setup with live warmup generations "
                "excluded; excludes model construction, checkpoint loading, video "
                "writing, and frame dumps"
            ),
            "device_label": args.device_label,
            "chart_label": args.chart_label or args.device_label,
            "official": _generation_ms_per_clip(summary, "upstream")
            if args.bench_side != "flashdreams"
            else None,
            "flashdreams": _generation_ms_per_clip(summary, "flashdreams")
            if args.bench_side != "upstream"
            else None,
            "warmup_generations": (
                summary.get("upstream", summary.get("flashdreams"))[
                    "warmup_generations"
                ]
            ),
            "measured_generations": (
                summary.get("upstream", summary.get("flashdreams"))[
                    "generations_measured"
                ]
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    report = _render_markdown(summary)
    args.output_md.write_text(report, encoding="utf-8")
    if args.output_chart_md is not None:
        args.output_chart_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_chart_md.write_text(
            _render_chart_markdown(summary, args.chart_label or args.device_label),
            encoding="utf-8",
        )
    print(report)


if __name__ == "__main__":
    main()
