# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for QVG reproduction benchmark helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_example_module(name: str):
    path = EXAMPLES_DIR / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_first_decoded_chunk_range_is_first_three_frames() -> None:
    bench = _load_example_module("qvg_benchmark.py")

    ranges, unavailable = bench._build_metric_ranges(
        frame_count=10,
        generated_frame_start=0,
        post_compression_start=3,
        first_compressed_frame_start=3,
    )

    first = ranges["first_decoded_chunk"]
    assert (first.start, first.end, first.frame_count) == (0, 3, 3)
    compressed = ranges["first_compressed_decoded_chunk"]
    assert (compressed.start, compressed.end, compressed.frame_count) == (3, 6, 3)
    assert not unavailable


def test_first_decoded_chunk_can_start_after_conditioning_frames() -> None:
    bench = _load_example_module("qvg_benchmark.py")

    ranges, _ = bench._build_metric_ranges(
        frame_count=12,
        generated_frame_start=5,
        post_compression_start=9,
        first_compressed_frame_start=9,
    )

    first = ranges["first_decoded_chunk"]
    assert (first.start, first.end, first.frame_count) == (5, 8, 3)


def test_first_decoded_chunk_requires_three_paired_frames() -> None:
    bench = _load_example_module("qvg_benchmark.py")

    with pytest.raises(ValueError, match="first_decoded_chunk requires"):
        bench._build_metric_ranges(
            frame_count=2,
            generated_frame_start=0,
            post_compression_start=3,
            first_compressed_frame_start=3,
        )


def test_lpips_range_filtering_uses_only_selected_range(monkeypatch) -> None:
    bench = _load_example_module("qvg_benchmark.py")
    baseline = np.zeros((6, 4, 4, 3), dtype=np.float32)
    candidate = baseline.copy()
    candidate[:3] = 10.0
    ranges = {
        "first_decoded_chunk": bench.FrameRange(
            "first_decoded_chunk",
            0,
            3,
            "primary",
        ),
        "all": bench.FrameRange("all", 0, 6, "diagnostic"),
        "last12": bench.FrameRange("last12", 0, 6, "diagnostic"),
    }
    metrics = {
        "ranges": {
            name: bench._metrics_for_slice(baseline, candidate, frame_range)
            for name, frame_range in ranges.items()
        }
    }
    seen_lengths = []

    def fake_lpips_video(base, cand, **_kwargs):
        seen_lengths.append(len(base))
        return 0.123, None

    monkeypatch.setattr(bench, "_lpips_video", fake_lpips_video)

    bench._add_lpips_range_metrics(
        metrics=metrics,
        baseline=baseline,
        candidate=candidate,
        ranges=ranges,
        selected_ranges=["first_decoded_chunk"],
        net="alex",
        batch_size=8,
        resize_short_side=256,
    )
    bench._add_primary_aliases(metrics)

    assert seen_lengths == [3]
    assert metrics["ranges"]["first_decoded_chunk"]["lpips"] == 0.123
    assert "lpips" not in metrics["ranges"]["all"]


def test_vbench_wrapper_reports_missing_dependency(monkeypatch) -> None:
    bench = _load_example_module("qvg_benchmark.py")
    original_find_spec = bench.importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "vbench":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(bench.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="VBench is not installed"):
        bench._import_vbench_class()


def test_vbench_normalizer_extracts_required_dimensions() -> None:
    bench = _load_example_module("qvg_benchmark.py")

    normalized = bench.normalize_vbench_payload(
        {
            "background_consistency": [
                0.91,
                [
                    {
                        "video_path": "/tmp/000000.mp4",
                        "video_results": 0.90,
                    }
                ],
            ],
            "imaging_quality": [
                82.0,
                [
                    {
                        "video_path": "/tmp/000000.mp4",
                        "video_results": 81.0,
                    }
                ],
            ],
            "subject_consistency": [0.88],
            "aesthetic_quality": {"mean": 0.76},
        },
        videos=[Path("/videos/a.mp4")],
        prompts=["prompt a"],
    )

    assert normalized["scores"]["background_consistency"] == 0.91
    assert normalized["scores"]["image_quality"] == 0.82
    assert normalized["scores"]["subject_consistency"] == 0.88
    assert normalized["scores"]["aesthetic_quality"] == 0.76
    assert normalized["per_video"][0]["video"] == "/videos/a.mp4"
    assert normalized["per_video"][0]["prompt"] == "prompt a"
    assert normalized["per_video"][0]["scores"]["background_consistency"] == 0.90
    assert normalized["per_video"][0]["scores"]["image_quality"] == 0.81


def test_aggregator_passes_when_all_eight_metrics_align(tmp_path: Path) -> None:
    bench = _load_example_module("qvg_benchmark.py")

    def write_json(name: str, payload: object) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return path

    official_compare = write_json(
        "official_compare.json",
        {
            "ranges": {
                "all": {
                    "psnr_db": 31.0,
                    "ssim_rgb_gaussian": 0.91,
                    "lpips": 0.05,
                    "frame_indices": [0, 189],
                },
                "first_decoded_chunk": {
                    "psnr_db": 31.0,
                    "ssim_rgb_gaussian": 0.91,
                    "lpips": 0.05,
                }
            }
        },
    )
    flashdreams_compare = write_json(
        "flashdreams_compare.json",
        {
            "ranges": {
                "all": {
                    "psnr_db": 30.8,
                    "ssim_rgb_gaussian": 0.905,
                    "lpips": 0.055,
                    "frame_indices": [0, 189],
                },
                "first_decoded_chunk": {
                    "psnr_db": 30.8,
                    "ssim_rgb_gaussian": 0.905,
                    "lpips": 0.055,
                }
            }
        },
    )
    official_vbench = write_json(
        "official_vbench.json",
        {
            "scores": {
                "background_consistency": 0.90,
                "image_quality": 0.82,
                "subject_consistency": 0.88,
                "aesthetic_quality": 0.76,
            },
            "per_video": [
                {
                    "scores": {
                        "background_consistency": 0.90,
                        "image_quality": 0.82,
                        "subject_consistency": 0.88,
                        "aesthetic_quality": 0.76,
                    }
                }
            ],
        },
    )
    official_bf16_vbench = write_json(
        "official_bf16_vbench.json",
        {
            "scores": {
                "background_consistency": 0.902,
                "image_quality": 0.821,
                "subject_consistency": 0.881,
                "aesthetic_quality": 0.762,
            },
            "per_video": [
                {
                    "scores": {
                        "background_consistency": 0.902,
                        "image_quality": 0.821,
                        "subject_consistency": 0.881,
                        "aesthetic_quality": 0.762,
                    }
                }
            ],
        },
    )
    flashdreams_vbench = write_json(
        "flashdreams_vbench.json",
        {
            "scores": {
                "background_consistency": 0.895,
                "image_quality": 0.815,
                "subject_consistency": 0.875,
                "aesthetic_quality": 0.755,
            },
            "per_video": [
                {
                    "scores": {
                        "background_consistency": 0.895,
                        "image_quality": 0.815,
                        "subject_consistency": 0.875,
                        "aesthetic_quality": 0.755,
                    }
                }
            ],
        },
    )
    flashdreams_bf16_vbench = write_json(
        "flashdreams_bf16_vbench.json",
        {
            "scores": {
                "background_consistency": 0.896,
                "image_quality": 0.816,
                "subject_consistency": 0.876,
                "aesthetic_quality": 0.756,
            },
            "per_video": [
                {
                    "scores": {
                        "background_consistency": 0.896,
                        "image_quality": 0.816,
                        "subject_consistency": 0.876,
                        "aesthetic_quality": 0.756,
                    }
                }
            ],
        },
    )
    flashdreams_stats = write_json(
        "flashdreams_stats.json",
        [{"kv_cache_compression_ratio": 6.55}],
    )
    official_bf16_run_stats = write_json(
        "official_bf16_run_stats.json",
        {"generated_frames": 100, "generation_seconds": 5.0, "wall_seconds": 10.0},
    )
    official_qvg_run_stats = write_json(
        "official_qvg_run_stats.json",
        {"generated_frames": 100, "generation_seconds": 6.25, "wall_seconds": 12.5},
    )
    flashdreams_bf16_run_stats = write_json(
        "flashdreams_bf16_run_stats.json",
        {"generated_frames": 120, "generation_seconds": 6.0, "wall_seconds": 10.0},
    )
    flashdreams_qvg_run_stats = write_json(
        "flashdreams_qvg_run_stats.json",
        {"generated_frames": 120, "generation_seconds": 8.0, "wall_seconds": 12.0},
    )

    report = bench.aggregate_report(
        official_compares=[bench._load_json(official_compare)],
        flashdreams_compares=[bench._load_json(flashdreams_compare)],
        official_vbench=bench._load_json(official_vbench),
        flashdreams_vbench=bench._load_json(flashdreams_vbench),
        official_bf16_vbench=bench._load_json(official_bf16_vbench),
        flashdreams_bf16_vbench=bench._load_json(flashdreams_bf16_vbench),
        official_compression_ratio=6.60,
        flashdreams_stats=[flashdreams_stats],
        official_bf16_run_stats=[official_bf16_run_stats],
        official_qvg_run_stats=[official_qvg_run_stats],
        flashdreams_bf16_run_stats=[flashdreams_bf16_run_stats],
        flashdreams_qvg_run_stats=[flashdreams_qvg_run_stats],
        quant_label="int2",
        fidelity_range="all",
    )

    assert report["all_required_metrics_pass"]
    assert report["fidelity_range"] == "all"
    assert report["fidelity_frame_indices"] == [0, 189]
    assert set(report["metrics"]) == {
        "compression_ratio",
        "psnr_db",
        "ssim_rgb_gaussian",
        "lpips",
        "background_consistency",
        "image_quality",
        "subject_consistency",
        "aesthetic_quality",
    }
    background = report["metrics"]["background_consistency"]
    assert background["comparison_mode"] == "bf16_delta"
    assert background["official_display"] == "0.9020 (-0.0020)"
    assert background["flashdreams_display"] == "0.8960 (-0.0010)"
    assert "flashdreams_minus_official" not in background
    assert report["metric_rows"][-1]["row"] == "average"
    assert report["average_metrics"] == report["metrics"]
    assert len(report["per_prompt_metrics"]) == 1
    fps = report["performance_metrics"]["generation_fps"]
    assert fps["official_display"] == "20.0000 (-4.0000)"
    assert fps["flashdreams_display"] == "20.0000 (-5.0000)"
    assert fps["source"] == "generated_frames / generation_seconds"
    generation_seconds = report["performance_metrics"]["generation_seconds"]
    assert generation_seconds["official_display"] == "5.0000 (+1.2500)"
    assert generation_seconds["flashdreams_display"] == "6.0000 (+2.0000)"
    assert generation_seconds["source"] == "model_generation_only_excludes_checkpoint_load_and_video_write"
    end_to_end_fps = report["performance_metrics"]["end_to_end_fps"]
    assert end_to_end_fps["official_display"] == "10.0000 (-2.0000)"
    assert end_to_end_fps["flashdreams_display"] == "12.0000 (-2.0000)"
    assert end_to_end_fps["source"] == "generated_frames / end_to_end_wall_seconds"
    wall_seconds = report["performance_metrics"]["end_to_end_wall_seconds"]
    assert wall_seconds["official_display"] == "10.0000 (+2.5000)"
    assert wall_seconds["flashdreams_display"] == "10.0000 (+2.0000)"
    assert wall_seconds["source"] == "end_to_end_command_wall_clock"
    assert report["metric_rows"][0]["performance_metrics"]["generation_fps"][
        "official_display"
    ] == "20.0000 (-4.0000)"
    assert report["metric_rows"][0]["performance_metrics"]["end_to_end_wall_seconds"][
        "official_display"
    ] == "10.0000 (+2.5000)"


def test_visual_grid_writes_two_by_two_video(monkeypatch, tmp_path: Path) -> None:
    bench = _load_example_module("qvg_benchmark.py")
    source = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    captured = {}

    def fake_read_video(path: Path) -> np.ndarray:
        offsets = {"official_bf16.mp4": 1, "official_int2.mp4": 2, "fd_bf16.mp4": 3, "fd_int2.mp4": 4}
        return source + offsets[path.name]

    def fake_write_video(path: Path, frames: np.ndarray, *, fps: float) -> None:
        captured["path"] = path
        captured["frames"] = frames
        captured["fps"] = fps

    monkeypatch.setattr(bench, "_read_video", fake_read_video)
    monkeypatch.setattr(bench, "_write_video", fake_write_video)
    monkeypatch.setattr(bench, "cv2", None)

    result = bench.write_visual_grid(
        official_bf16_video=Path("official_bf16.mp4"),
        official_qvg_video=Path("official_int2.mp4"),
        flashdreams_bf16_video=Path("fd_bf16.mp4"),
        flashdreams_qvg_video=Path("fd_int2.mp4"),
        output_path=tmp_path / "grid.mp4",
        max_frames=2,
        fps=12.0,
    )

    assert result["layout"] == [["Official BF16", "Official INT2"], ["FlashDreams BF16", "FlashDreams INT2"]]
    assert result["frames"] == 2
    assert captured["path"] == tmp_path / "grid.mp4"
    assert captured["fps"] == 12.0
    assert captured["frames"].shape == (2, 8, 10, 3)
    assert np.all(captured["frames"][:, :4, :5] == 1)
    assert np.all(captured["frames"][:, :4, 5:] == 2)
    assert np.all(captured["frames"][:, 4:, :5] == 3)
    assert np.all(captured["frames"][:, 4:, 5:] == 4)
