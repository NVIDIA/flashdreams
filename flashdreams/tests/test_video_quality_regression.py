# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.video_quality.manifest import load_manifest
from tools.video_quality.metrics import compute_video_metrics, synthetic_video
from tools.video_quality.run_regression import main

pytestmark = pytest.mark.ci_cpu


def test_starter_manifest_loads() -> None:
    manifest = load_manifest(Path("configs/video_quality_cases.yml"))

    assert manifest.schema_version == 1
    assert "calibration" in manifest.suites
    assert (
        manifest.select_cases(suite="per_commit")[0].id
        == "synthetic_core_metric_sentinels"
    )


def test_core_metrics_separate_synthetic_failures() -> None:
    good = synthetic_video("textured_motion", frames=16, height=64, width=64, seed=17)
    grey = synthetic_video("grey_blank", frames=16, height=64, width=64)
    blurry = synthetic_video("blurry_gradient", frames=16, height=64, width=64)
    stripes = synthetic_video("horizontal_stripes", frames=16, height=64, width=64)

    good_metrics = compute_video_metrics(good.frames, fps=good.fps)
    grey_metrics = compute_video_metrics(grey.frames, fps=grey.fps)
    blurry_metrics = compute_video_metrics(blurry.frames, fps=blurry.fps)
    stripe_metrics = compute_video_metrics(stripes.frames, fps=stripes.fps)

    assert good_metrics["luma_std"] > 0.08
    assert grey_metrics["luma_std"] < 0.01
    assert grey_metrics["grey_pixel_ratio"] == 1.0
    assert good_metrics["laplacian_variance"] > blurry_metrics["laplacian_variance"]
    assert (
        stripe_metrics["fft_axis_energy_ratio"] > good_metrics["fft_axis_energy_ratio"]
    )
    assert stripe_metrics["row_autocorr_peak"] > good_metrics["row_autocorr_peak"]


def test_runner_writes_evaluate_only_manifest(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--manifest",
            "configs/video_quality_cases.yml",
            "--suite",
            "calibration",
            "--output-dir",
            str(tmp_path),
            "--evaluate-only",
        ]
    )

    assert exit_code == 0
    run_manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert run_manifest["decision"] == "pass"
    assert run_manifest["case_count"] == 1
    case = run_manifest["cases"][0]
    assert case["decision"]["status"] == "pass"
    known_bad = {
        clip["id"]: clip for clip in case["clips"] if clip["role"] == "known_bad"
    }
    assert known_bad["grey_blank"]["decision"]["calibration_expected_failure_observed"]
