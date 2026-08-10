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

"""CPU-safe tests for the SANA-WM performance benchmark summary."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.ci_cpu


def _load_bench_summary() -> ModuleType:
    path = Path("integrations/sana/tests/performance/bench_summary.py")
    spec = importlib.util.spec_from_file_location("sana_wm_bench_summary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_bench_sweep_summary() -> ModuleType:
    path = Path("integrations/sana/tests/performance/bench_sweep_summary.py")
    spec = importlib.util.spec_from_file_location("sana_wm_bench_sweep_summary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bidirectional_summary_uses_steady_state_generation_ms_per_clip_for_chart() -> (
    None
):
    module = _load_bench_summary()
    summary = {
        "inputs": {
            "image_path": "image.png",
            "prompt_path": "prompt.txt",
            "camera_path": "pose.npy",
            "intrinsics_path": "intrinsics.npy",
            "num_frames": 20,
            "seed": 42,
            "no_refiner": True,
            "stage1_precision": "bf16",
            "refiner_precision": "bf16",
            "quant_backend": "auto",
            "compile_stage1": False,
            "force_cudnn_sdpa": True,
            "warmup_runs": 1,
        },
        "upstream": {
            "runs_measured": 3,
            "generations_measured": 3,
            "wall_median_s": 2.0,
            "wall_p90_s": 2.2,
            "stage1_total_median_ms": 1600.0,
            "dit_median_ms": 1500.0,
            "vae_decode_median_ms": 300.0,
            "mem_peak_median_gib": 10.0,
        },
        "flashdreams": {
            "runs_measured": 3,
            "generations_measured": 3,
            "wall_median_s": 1.5,
            "wall_p90_s": 1.8,
            "encode_median_ms": 100.0,
            "dit_median_ms": 1100.0,
            "vae_decode_median_ms": 250.0,
            "mem_peak_median_gib": 12.0,
        },
    }

    report = module._render_markdown(summary)
    chart = module._render_chart_markdown(summary, "Test GPU")

    assert "## Benchmark metric" in report
    assert "- stage1_precision: `bf16`" in report
    assert "steady-state in-process generation latency per generated clip" in report
    assert "measured generations | 3 | 3" in report
    assert "generation median / clip | 2000.00 ms | 1500.00 ms" in report
    assert "generation median / frame, diagnostic | 100.00 ms | 75.00 ms" in report
    assert "## Timing breakdown" in report
    assert "| Stage-1 DiT median | 1500.00 ms | 1100.00 ms |" in report
    assert "Stage-1 DiT metric" not in report
    assert "Generation after model load" not in report
    assert chart == (
        "# SANA-WM Benchmark Data (ms)\n"
        "\n"
        "| device | official | flashdreams |\n"
        "| --- | ---: | ---: |\n"
        "| Test GPU | 2000.00 | 1500.00 |\n"
    )


def test_benchmark_summary_writes_chart_data(tmp_path: Path) -> None:
    module = _load_bench_summary()
    upstream = tmp_path / "upstream" / "run_0"
    flashdreams = tmp_path / "flashdreams" / "run_0"
    upstream.mkdir(parents=True)
    flashdreams.mkdir(parents=True)
    (upstream / "stats.json").write_text(
        json.dumps(
            {
                "generation_records": [
                    {
                        "generation_index": 0,
                        "warmup": True,
                        "wall_s": 99.0,
                        "mem_peak_gib": 99.0,
                        "timings_s": {
                            "stage1_sample_s": 90.0,
                            "stage1_dit_s": 80.0,
                            "vae_decode_s": 9.0,
                        },
                    },
                    {
                        "generation_index": 1,
                        "warmup": False,
                        "wall_s": 3.0,
                        "mem_peak_gib": 9.0,
                        "timings_s": {
                            "stage1_sample_s": 2.5,
                            "stage1_dit_s": 2.0,
                            "vae_decode_s": 0.5,
                        },
                    },
                    {
                        "generation_index": 2,
                        "warmup": False,
                        "wall_s": 4.0,
                        "mem_peak_gib": 10.0,
                        "timings_s": {
                            "stage1_sample_s": 3.5,
                            "stage1_dit_s": 3.0,
                            "vae_decode_s": 0.5,
                        },
                    },
                    {
                        "generation_index": 3,
                        "warmup": False,
                        "wall_s": 5.0,
                        "mem_peak_gib": 11.0,
                        "timings_s": {
                            "stage1_sample_s": 4.5,
                            "stage1_dit_s": 4.0,
                            "vae_decode_s": 0.5,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (flashdreams / "stats.json").write_text(
        json.dumps(
            {
                "generation_records": [
                    {
                        "generation_index": 0,
                        "warmup": True,
                        "wall_s": 77.0,
                        "stats_ms": {
                            "encode_ms": 700.0,
                            "diffuse_ms": 70000.0,
                            "decode_ms": 6000.0,
                            "mem_peak_gib": 70.0,
                        },
                    },
                    {
                        "generation_index": 1,
                        "warmup": False,
                        "wall_s": 1.5,
                        "stats_ms": {
                            "encode_ms": 80.0,
                            "diffuse_ms": 1200.0,
                            "decode_ms": 300.0,
                            "mem_peak_gib": 11.0,
                        },
                    },
                    {
                        "generation_index": 2,
                        "warmup": False,
                        "wall_s": 2.0,
                        "stats_ms": {
                            "encode_ms": 100.0,
                            "diffuse_ms": 1500.0,
                            "decode_ms": 400.0,
                            "mem_peak_gib": 12.0,
                        },
                    },
                    {
                        "generation_index": 3,
                        "warmup": False,
                        "wall_s": 2.5,
                        "stats_ms": {
                            "encode_ms": 120.0,
                            "diffuse_ms": 1800.0,
                            "decode_ms": 500.0,
                            "mem_peak_gib": 13.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--upstream-dir",
            str(tmp_path / "upstream"),
            "--flashdreams-dir",
            str(tmp_path / "flashdreams"),
            "--warmup-runs",
            "1",
            "--image-path",
            "image.png",
            "--prompt-path",
            "prompt.txt",
            "--camera-path",
            "pose.npy",
            "--intrinsics-path",
            "intrinsics.npy",
            "--num-frames",
            "40",
            "--seed",
            "42",
            "--device-label",
            "Test GPU",
            "--stage1-precision",
            "bf16",
            "--refiner-precision",
            "bf16",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert "| Test GPU | 4000.00 | 2000.00 |" in out_chart.read_text(encoding="utf-8")
    report = out_md.read_text(encoding="utf-8")
    assert "warmup generations discarded: `1`" in report
    assert "measured generations | 3 | 3" in report
    assert "generation median / clip | 4000.00 ms | 2000.00 ms" in report
    assert "generation median / frame, diagnostic | 100.00 ms | 50.00 ms" in report
    assert "generation median / frame |" not in report
    assert "per generated clip" in report
    assert "not as independently timed frames" in report
    assert "per generated frame" not in report
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["benchmark"] == {
        "metric": "steady_state_generation_ms_per_clip",
        "unit": "ms",
        "timing_boundary": (
            "pipeline.generate after model setup with live warmup generations "
            "excluded; excludes model construction, checkpoint loading, video "
            "writing, and frame dumps"
        ),
        "device_label": "Test GPU",
        "chart_label": "Test GPU",
        "official": 4000.0,
        "flashdreams": 2000.0,
        "warmup_generations": 1,
        "measured_generations": 3,
    }
    assert summary["upstream"]["generations_measured"] == 3
    assert summary["upstream"]["run_records"][0]["warmup"] is True
    assert summary["upstream"]["run_records"][1]["path"].endswith(
        "stats.json#generation_1"
    )


def test_bidirectional_summary_rejects_stale_single_record_stats(
    tmp_path: Path,
) -> None:
    module = _load_bench_summary()
    upstream = tmp_path / "upstream" / "run_0"
    upstream.mkdir(parents=True)
    (upstream / "stats.json").write_text(
        json.dumps({"wall_s": 4.0, "timings_s": {"stage1_sample_s": 3.5}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation_records"):
        module.main(
            [
                "--variant",
                "bidirectional",
                "--bench-side",
                "upstream",
                "--upstream-dir",
                str(tmp_path / "upstream"),
                "--image-path",
                "image.png",
                "--prompt-path",
                "prompt.txt",
                "--camera-path",
                "pose.npy",
                "--intrinsics-path",
                "intrinsics.npy",
                "--num-frames",
                "40",
                "--seed",
                "42",
                "--stage1-precision",
                "bf16",
                "--refiner-precision",
                "bf16",
                "--output-json",
                str(tmp_path / "bench.json"),
                "--output-md",
                str(tmp_path / "bench.md"),
            ]
        )


def test_bidirectional_summary_accepts_flashdreams_only_low_precision(
    tmp_path: Path,
) -> None:
    module = _load_bench_summary()
    flashdreams = tmp_path / "flashdreams" / "run_0"
    flashdreams.mkdir(parents=True)
    (flashdreams / "stats.json").write_text(
        json.dumps(
            {
                "generation_records": [
                    {
                        "generation_index": 0,
                        "warmup": True,
                        "wall_s": 99.0,
                        "stats_ms": {
                            "encode_ms": 900.0,
                            "diffuse_ms": 90000.0,
                            "decode_ms": 9000.0,
                            "mem_peak_gib": 70.0,
                        },
                    },
                    {
                        "generation_index": 1,
                        "warmup": False,
                        "wall_s": 2.0,
                        "stats_ms": {
                            "encode_ms": 100.0,
                            "diffuse_ms": 1500.0,
                            "decode_ms": 400.0,
                            "mem_peak_gib": 12.0,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--variant",
            "bidirectional",
            "--bench-side",
            "flashdreams",
            "--flashdreams-dir",
            str(tmp_path / "flashdreams"),
            "--warmup-runs",
            "1",
            "--image-path",
            "image.png",
            "--prompt-path",
            "prompt.txt",
            "--camera-path",
            "pose.npy",
            "--intrinsics-path",
            "intrinsics.npy",
            "--num-frames",
            "40",
            "--seed",
            "42",
            "--stage1-precision",
            "fp8",
            "--refiner-precision",
            "fp8",
            "--device-label",
            "Test GPU",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Benchmark Data (ms)\n"
        "\n"
        "| device | flashdreams |\n"
        "| --- | ---: |\n"
        "| Test GPU | 2000.00 |\n"
    )
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["benchmark"]["official"] is None
    assert summary["benchmark"]["flashdreams"] == 2000.0
    assert summary["benchmark"]["measured_generations"] == 1
    report = out_md.read_text(encoding="utf-8")
    assert "| metric | FlashDreams |" in report
    assert "generation median / clip | 2000.00 ms" in report


def test_bidirectional_summary_rejects_low_precision_comparison(tmp_path: Path) -> None:
    module = _load_bench_summary()
    with pytest.raises(SystemExit):
        module.main(
            [
                "--variant",
                "bidirectional",
                "--bench-side",
                "both",
                "--upstream-dir",
                str(tmp_path / "upstream"),
                "--flashdreams-dir",
                str(tmp_path / "flashdreams"),
                "--image-path",
                "image.png",
                "--prompt-path",
                "prompt.txt",
                "--camera-path",
                "pose.npy",
                "--intrinsics-path",
                "intrinsics.npy",
                "--num-frames",
                "40",
                "--seed",
                "42",
                "--stage1-precision",
                "fp8",
                "--refiner-precision",
                "fp8",
                "--output-json",
                str(tmp_path / "bench.json"),
                "--output-md",
                str(tmp_path / "bench.md"),
            ]
        )


def test_bidirectional_bench_script_rejects_low_precision_dry_run() -> None:
    env = os.environ.copy()
    env.update(
        {
            "BENCH_DRY_RUN": "1",
            "SANA_WM_VARIANT": "bidirectional",
            "STAGE1_PRECISION": "fp4",
            "REFINER_PRECISION": "fp4",
        }
    )
    result = subprocess.run(
        ["bash", "integrations/sana/tests/performance/bench.sh"],
        check=False,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "upstream SANA-WM_bidirectional benchmarks are BF16-only" in result.stderr


@pytest.mark.parametrize(
    ("variant", "side", "precision", "expected_quant"),
    [
        ("bidirectional", "flashdreams", "fp4", "no"),
        ("streaming", "flashdreams", "fp8", "no"),
        ("streaming", "both", "bf16", "no"),
        ("streaming", "upstream", "fp4", "yes"),
        ("streaming", "both", "fp8", "yes"),
    ],
)
def test_bench_script_quant_extra_is_upstream_streaming_low_precision_only(
    variant: str,
    side: str,
    precision: str,
    expected_quant: str,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "BENCH_DRY_RUN": "1",
            "SANA_WM_VARIANT": variant,
            "BENCH_SIDE": side,
            "STAGE1_PRECISION": precision,
            "REFINER_PRECISION": precision,
            "BENCH_PRECISIONS": "",
        }
    )
    result = subprocess.run(
        ["bash", "integrations/sana/tests/performance/bench.sh"],
        check=False,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert f"upstream quant extra: {expected_quant}" in result.stdout


def test_benchmark_summary_keeps_frame_normalized_diagnostics() -> None:
    module = _load_bench_summary()
    assert (
        module._generation_ms_per_frame(
            {
                "inputs": {"num_frames": 40},
                "upstream": {"wall_median_s": 4.0},
            },
            "upstream",
        )
        == 100.0
    )
    assert (
        module._generation_ms_per_clip(
            {
                "inputs": {"num_frames": 40},
                "upstream": {"wall_median_s": 4.0},
            },
            "upstream",
        )
        == 4000.0
    )


def test_streaming_upstream_summary_uses_steady_state_ms_per_chunk(
    tmp_path: Path,
) -> None:
    module = _load_bench_summary()
    warmup = tmp_path / "upstream" / "run_0"
    measured = tmp_path / "upstream" / "run_1"
    warmup.mkdir(parents=True)
    measured.mkdir(parents=True)
    (warmup / "stats.json").write_text(
        json.dumps(
            {
                "variant": "streaming",
                "stream_wall_seconds": 12.0,
                "steady_state_seconds": 8.0,
                "first_chunk_seconds": 4.0,
                "n_decode_chunks": 5,
                "n_pixel_frames": 97,
                "peak_mem_gb": 40.0,
                "output_path": "warmup.mp4",
            }
        ),
        encoding="utf-8",
    )
    (measured / "stats.json").write_text(
        json.dumps(
            {
                "variant": "streaming",
                "stream_wall_seconds": 10.0,
                "end_to_end_seconds": 10.2,
                "steady_state_seconds": 6.0,
                "first_chunk_seconds": 4.0,
                "first_chunk_frames": 25,
                "n_decode_chunks": 4,
                "n_refiner_blocks": 4,
                "n_pixel_frames": 97,
                "steady_state_frames_per_second": 12.0,
                "steady_state_realtime_factor": 0.75,
                "stage1_cuda_seconds": 2.0,
                "refiner_cuda_seconds": 3.0,
                "decode_cuda_seconds": 1.0,
                "peak_mem_gb": 48.0,
                "output_path": "measured.mp4",
            }
        ),
        encoding="utf-8",
    )
    (measured / "command.txt").write_text(
        "uv run python upstream.py\n", encoding="utf-8"
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"

    module.main(
        [
            "--variant",
            "streaming",
            "--bench-side",
            "upstream",
            "--upstream-dir",
            str(tmp_path / "upstream"),
            "--warmup-runs",
            "1",
            "--image-path",
            "image.png",
            "--prompt-path",
            "prompt.txt",
            "--camera-path",
            "pose.npy",
            "--camera-source",
            "action",
            "--action",
            "w-80",
            "--intrinsics-path",
            "intrinsics.npy",
            "--num-frames",
            "97",
            "--seed",
            "42",
            "--stage1-precision",
            "bf16",
            "--refiner-precision",
            "bf16",
            "--device-label",
            "Test GPU",
            "--upstream-commit",
            "6298508",
            "--output-mode",
            "mp4",
            "--denoising-step-list",
            "1000,960,889,727,0",
            "--num-frame-per-block",
            "3",
            "--refiner-block-size",
            "3",
            "--refiner-kv-max-frames",
            "11",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["variant"] == "streaming"
    assert summary["bench_side"] == "upstream"
    assert summary["commands"]["upstream"] == "uv run python upstream.py"
    assert summary["upstream"]["steady_state_chunk_median_ms"] == 2000.0
    assert summary["upstream"]["wall_median_s"] == 10.0
    assert summary["upstream"]["mem_peak_median_gib"] == 48.0
    assert summary["upstream"]["stage1_cuda_median_ms"] == 2000.0
    assert summary["benchmark"]["metric"] == "steady_state_generation_ms_per_chunk"
    assert summary["benchmark"]["official"] == 2000.0
    assert summary["benchmark"]["flashdreams"] is None
    report = out_md.read_text(encoding="utf-8")
    assert "# SANA-WM streaming upstream benchmark" in report
    assert "steady-state generation median / chunk | 2000.00 ms" in report
    assert "full-clip wall median | 10.00 s" in report


def test_streaming_comparison_summary_writes_chart_data(tmp_path: Path) -> None:
    module = _load_bench_summary()
    upstream = tmp_path / "upstream" / "run_0"
    flashdreams = tmp_path / "flashdreams" / "run_0"
    upstream.mkdir(parents=True)
    flashdreams.mkdir(parents=True)
    (upstream / "stats.json").write_text(
        json.dumps(
            {
                "variant": "streaming",
                "stream_wall_seconds": 12.0,
                "steady_state_seconds": 8.0,
                "first_chunk_seconds": 4.0,
                "first_chunk_frames": 24,
                "n_decode_chunks": 5,
                "n_refiner_blocks": 5,
                "n_pixel_frames": 120,
                "stage1_cuda_seconds": 2.0,
                "refiner_cuda_seconds": 4.0,
                "decode_cuda_seconds": 1.0,
                "peak_mem_gb": 48.0,
                "output_path": "upstream.mp4",
            }
        ),
        encoding="utf-8",
    )
    (flashdreams / "stats.json").write_text(
        json.dumps(
            {
                "variant": "streaming",
                "stream_wall_seconds": 8.0,
                "steady_state_seconds": 4.0,
                "first_chunk_seconds": 4.0,
                "first_chunk_frames": 24,
                "n_decode_chunks": 5,
                "n_refiner_blocks": 5,
                "n_pixel_frames": 120,
                "stage1_cuda_seconds": 1.5,
                "refiner_cuda_seconds": None,
                "decode_cuda_seconds": 2.0,
                "mem_peak_gib": 44.0,
                "output_path": "flashdreams.mp4",
            }
        ),
        encoding="utf-8",
    )
    (upstream / "command.txt").write_text(
        "uv run python upstream.py\n", encoding="utf-8"
    )
    (flashdreams / "command.txt").write_text(
        "uv run python run_flashdreams_streaming.py\n",
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--variant",
            "streaming",
            "--bench-side",
            "both",
            "--upstream-dir",
            str(tmp_path / "upstream"),
            "--flashdreams-dir",
            str(tmp_path / "flashdreams"),
            "--warmup-runs",
            "0",
            "--image-path",
            "image.png",
            "--prompt-path",
            "prompt.txt",
            "--camera-path",
            "pose.npy",
            "--camera-source",
            "action",
            "--action",
            "w-80",
            "--intrinsics-path",
            "intrinsics.npy",
            "--num-frames",
            "121",
            "--seed",
            "42",
            "--stage1-precision",
            "bf16",
            "--refiner-precision",
            "bf16",
            "--device-label",
            "Test GPU",
            "--output-mode",
            "mp4",
            "--denoising-step-list",
            "1000,960,889,727,0",
            "--num-frame-per-block",
            "3",
            "--refiner-block-size",
            "3",
            "--refiner-kv-max-frames",
            "11",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Streaming Benchmark Data (ms/chunk)\n"
        "\n"
        "| device | official | flashdreams |\n"
        "| --- | ---: | ---: |\n"
        "| Test GPU | 2000.00 | 1000.00 |\n"
    )
    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["bench_side"] == "both"
    assert summary["commands"] == {
        "upstream": "uv run python upstream.py",
        "flashdreams": "uv run python run_flashdreams_streaming.py",
    }
    assert summary["benchmark"]["official"] == 2000.0
    assert summary["benchmark"]["flashdreams"] == 1000.0
    report = out_md.read_text(encoding="utf-8")
    assert "# SANA-WM streaming benchmark" in report
    assert "steady-state generation median / chunk | 2000.00 ms | 1000.00 ms" in report


def test_benchmark_sweep_summary_accepts_bidirectional_bf16_chart_data(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    bf16_json = tmp_path / "bf16.json"
    bf16_json.write_text(
        json.dumps(
            {
                "variant": "bidirectional",
                "inputs": {
                    "stage1_precision": "bf16",
                    "refiner_precision": "bf16",
                },
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_clip",
                    "official": 1000.0,
                    "flashdreams": 900.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--item",
            f"BF16:{bf16_json}",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Precision Sweep Summary (ms)\n"
        "\n"
        "| precision | official | flashdreams |\n"
        "| --- | ---: | ---: |\n"
        "| BF16 | 1000.00 | 900.00 |\n"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"]["metric"] == "steady_state_generation_ms_per_clip"
    assert payload["benchmark"]["unit"] == "ms"
    assert payload["benchmark"]["has_official"] is True
    assert payload["benchmark"]["has_flashdreams"] is True
    assert [row["label"] for row in payload["rows"]] == ["BF16"]


def test_benchmark_sweep_summary_rejects_stale_bidirectional_metric(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    stale_json = tmp_path / "stale.json"
    stale_json.write_text(
        json.dumps(
            {
                "variant": "bidirectional",
                "inputs": {
                    "stage1_precision": "bf16",
                    "refiner_precision": "bf16",
                },
                "benchmark": {
                    "metric": "generation_ms_per_clip",
                    "official": 1000.0,
                    "flashdreams": 900.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="discard stale"):
        module.main(
            [
                "--item",
                f"BF16:{stale_json}",
                "--output-json",
                str(tmp_path / "bench.json"),
                "--output-md",
                str(tmp_path / "bench.md"),
                "--output-chart-md",
                str(tmp_path / "perf.md"),
            ]
        )


def test_benchmark_sweep_summary_accepts_bidirectional_flashdreams_only_low_precision(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    bf16_json = tmp_path / "bf16.json"
    fp8_json = tmp_path / "fp8.json"
    bf16_json.write_text(
        json.dumps(
            {
                "variant": "bidirectional",
                "inputs": {
                    "stage1_precision": "bf16",
                    "refiner_precision": "bf16",
                },
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_clip",
                    "official": 1000.0,
                    "flashdreams": 900.0,
                },
            }
        ),
        encoding="utf-8",
    )
    fp8_json.write_text(
        json.dumps(
            {
                "variant": "bidirectional",
                "inputs": {
                    "stage1_precision": "fp8",
                    "refiner_precision": "fp8",
                },
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_clip",
                    "official": None,
                    "flashdreams": 700.0,
                },
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--item",
            f"BF16:{bf16_json}",
            "--item",
            f"FP8:{fp8_json}",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Precision Sweep Summary (ms)\n"
        "\n"
        "| precision | official | flashdreams |\n"
        "| --- | ---: | ---: |\n"
        "| BF16 | 1000.00 | 900.00 |\n"
        "| FP8 | n/a | 700.00 |\n"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"]["has_official"] is True
    assert payload["benchmark"]["has_flashdreams"] is True
    assert payload["rows"][1]["official"] is None


def test_benchmark_sweep_summary_rejects_bidirectional_low_precision(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    fp8_json = tmp_path / "fp8.json"
    fp8_json.write_text(
        json.dumps(
            {
                "variant": "bidirectional",
                "inputs": {
                    "stage1_precision": "fp8",
                    "refiner_precision": "fp8",
                },
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_clip",
                    "official": 800.0,
                    "flashdreams": 700.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BF16-only"):
        module.main(
            [
                "--item",
                f"FP8:{fp8_json}",
                "--output-json",
                str(tmp_path / "bench.json"),
                "--output-md",
                str(tmp_path / "bench.md"),
                "--output-chart-md",
                str(tmp_path / "perf.md"),
            ]
        )


def test_benchmark_sweep_summary_accepts_streaming_upstream_only(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    bf16_json = tmp_path / "bf16.json"
    fp8_json = tmp_path / "fp8.json"
    bf16_json.write_text(
        json.dumps(
            {
                "variant": "streaming",
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_chunk",
                    "unit": "ms",
                    "official": 1200.0,
                    "flashdreams": None,
                },
            }
        ),
        encoding="utf-8",
    )
    fp8_json.write_text(
        json.dumps(
            {
                "variant": "streaming",
                "benchmark": {
                    "metric": "steady_state_generation_ms_per_chunk",
                    "unit": "ms",
                    "official": 900.0,
                    "flashdreams": None,
                },
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            "--item",
            f"BF16:{bf16_json}",
            "--item",
            f"FP8:{fp8_json}",
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Precision Sweep Upstream Summary (ms/chunk)\n"
        "\n"
        "| precision | official |\n"
        "| --- | ---: |\n"
        "| BF16 | 1200.00 |\n"
        "| FP8 | 900.00 |\n"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["benchmark"]["metric"] == "steady_state_generation_ms_per_chunk"
    assert payload["benchmark"]["has_official"] is True
    assert payload["benchmark"]["has_flashdreams"] is False
    assert [row["flashdreams"] for row in payload["rows"]] == [None, None]


def test_benchmark_sweep_summary_reports_streaming_comparison_values(
    tmp_path: Path,
) -> None:
    module = _load_bench_sweep_summary()
    items = {
        "BF16": (100.0, 80.0),
        "FP8": (90.0, 95.0),
        "FP4": (70.0, 35.0),
    }
    paths: list[tuple[str, Path]] = []
    for label, (official, flashdreams) in items.items():
        path = tmp_path / f"{label.lower()}.json"
        path.write_text(
            json.dumps(
                {
                    "variant": "streaming",
                    "benchmark": {
                        "metric": "steady_state_generation_ms_per_chunk",
                        "unit": "ms",
                        "official": official,
                        "flashdreams": flashdreams,
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append((label, path))
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    out_chart = tmp_path / "perf.md"

    module.main(
        [
            *[
                value
                for label, path in paths
                for value in ("--item", f"{label}:{path}")
            ],
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--output-chart-md",
            str(out_chart),
        ]
    )

    assert out_chart.read_text(encoding="utf-8") == (
        "# SANA-WM Precision Sweep Summary (ms/chunk)\n"
        "\n"
        "| precision | official | flashdreams |\n"
        "| --- | ---: | ---: |\n"
        "| BF16 | 100.00 | 80.00 |\n"
        "| FP8 | 90.00 | 95.00 |\n"
        "| FP4 | 70.00 | 35.00 |\n"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert [row["official"] for row in payload["rows"]] == [100.0, 90.0, 70.0]
    assert [row["flashdreams"] for row in payload["rows"]] == [80.0, 95.0, 35.0]
    report = out_md.read_text(encoding="utf-8")
    assert "| precision | official | FlashDreams | source |" in report
    assert "| BF16 | 100.00 ms | 80.00 ms |" in report
    assert "| FP8 | 90.00 ms | 95.00 ms |" in report
