# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from flashdreams.quality.video_quality.manifest import load_manifest
from flashdreams.quality.video_quality.metrics import (
    compute_video_metrics,
    synthetic_video,
)
from flashdreams.quality.video_quality.run_regression import main

pytestmark = pytest.mark.ci_cpu


def test_starter_manifest_loads() -> None:
    manifest = load_manifest(Path("configs/video_quality_cases.yml"))

    assert manifest.schema_version == 1
    assert "calibration" in manifest.suites
    case = manifest.select_cases(suite="calibration")[0]
    assert case.hf_dataset is None
    assert case.metrics == ("decode_metadata", "grey_blank", "sharpness", "stripe")
    assert (
        manifest.select_cases(suite="per_commit")[0].id
        == "synthetic_core_metric_sentinels"
    )


def test_manifest_rejects_bool_schema_version(tmp_path: Path) -> None:
    data = yaml.safe_load(Path("configs/video_quality_cases.yml").read_text())
    data["schema_version"] = True
    manifest_path = tmp_path / "cases.yml"
    manifest_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_manifest(manifest_path)


def test_core_metrics_separate_synthetic_failures() -> None:
    good = synthetic_video("textured_motion", frames=16, height=64, width=64, seed=17)
    grey = synthetic_video("grey_blank", frames=16, height=64, width=64)
    blurry = synthetic_video("blurry_gradient", frames=16, height=64, width=64)
    stripes = synthetic_video("horizontal_stripes", frames=16, height=64, width=64)

    good_metrics = compute_video_metrics(good.frames, fps=good.fps)
    grey_metrics = compute_video_metrics(grey.frames, fps=grey.fps)
    blurry_metrics = compute_video_metrics(blurry.frames, fps=blurry.fps)
    stripe_metrics = compute_video_metrics(stripes.frames, fps=stripes.fps)

    assert _float_metric(good_metrics, "luma_std") > 0.08
    assert _float_metric(grey_metrics, "luma_std") < 0.01
    assert grey_metrics["grey_pixel_ratio"] == 1.0
    assert _float_metric(good_metrics, "laplacian_variance") > _float_metric(
        blurry_metrics, "laplacian_variance"
    )
    assert _float_metric(stripe_metrics, "fft_axis_energy_ratio") > _float_metric(
        good_metrics, "fft_axis_energy_ratio"
    )
    assert _float_metric(stripe_metrics, "row_autocorr_peak") > _float_metric(
        good_metrics, "row_autocorr_peak"
    )


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
    for clip in known_bad.values():
        assert clip["decision"]["status"] == "pass"
        assert clip["decision"]["unexpected_failure_ids"] == []


def _float_metric(metrics: dict[str, float | int | bool | None], key: str) -> float:
    value = metrics[key]
    assert isinstance(value, (float, int)) and not isinstance(value, bool)
    return float(value)
