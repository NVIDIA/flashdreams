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

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest
from omnidreams.runner import (
    OmnidreamsRunner,
    _load_batch_records,
    _parse_batch_item,
)

pytestmark = pytest.mark.ci_cpu


def test_jsonl_batch_record_parses_prompt_file_and_paths(tmp_path) -> None:
    """Batch JSONL rows accept scalar aliases and prompt files."""
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("A downtown driving scene.", encoding="utf-8")
    manifest = tmp_path / "batch.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip-a",
                "dataset": "samples",
                "prompt_id": "vlm",
                "prompt_path": str(prompt_path),
                "hdmap_path": "maps/clip-a.mp4",
                "first_frame_path": "frames/clip-a.png",
                "camera_name": "camera_front_wide_120fov",
                "seed": "12",
                "metadata": {"sensor_stack": "8.1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = _load_batch_records(manifest)
    item = _parse_batch_item(records[0], index=0)

    assert item.clip_id == "clip-a"
    assert item.dataset == "samples"
    assert item.prompt_id == "vlm"
    assert item.prompt == "A downtown driving scene."
    assert item.hdmap_video_paths[0].as_posix() == "maps/clip-a.mp4"
    assert item.first_frame_paths[0].as_posix() == "frames/clip-a.png"
    assert item.camera_names == ("camera_front_wide_120fov",)
    assert item.seed == 12
    assert item.metadata["sensor_stack"] == "8.1"


def test_csv_batch_record_supports_json_arrays_and_prompt_commas(tmp_path) -> None:
    """CSV rows can carry JSON arrays without splitting prompt commas."""
    manifest = tmp_path / "batch.csv"
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "clip_id",
                "prompts",
                "hdmap_video_paths",
                "first_frame_paths",
                "camera_names",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "clip-b",
                "prompts": json.dumps(["front prompt, with a comma"]),
                "hdmap_video_paths": json.dumps(["front.mp4"]),
                "first_frame_paths": json.dumps(["front.png"]),
                "camera_names": json.dumps(["front"]),
            }
        )

    item = _parse_batch_item(_load_batch_records(manifest)[0], index=0)

    assert item.prompts == ("front prompt, with a comma",)
    assert item.hdmap_video_paths[0].as_posix() == "front.mp4"
    assert item.first_frame_paths[0].as_posix() == "front.png"
    assert item.camera_names == ("front",)


def test_batch_default_output_path_matches_sweep_layout(tmp_path) -> None:
    """Dataset/prompt-aware records default to the sweep output structure."""
    runner = OmnidreamsRunner.__new__(OmnidreamsRunner)
    runner.config = SimpleNamespace(
        output_dir=tmp_path,
        runner_name="omnidreams-model",
        pipeline=SimpleNamespace(diffusion_model=SimpleNamespace(seed=99)),
    )
    item = _parse_batch_item(
        {
            "dataset": "samples",
            "clip_id": "clip-c",
            "prompt_id": "simple",
            "seed": 7,
        },
        index=0,
    )

    assert runner._batch_output_video_path(item) == (
        tmp_path / "samples/clip-c/omnidreams-model/simple/7/video.mp4"
    )
    assert runner._batch_stats_path(item) == (
        tmp_path / "samples/clip-c/omnidreams-model/simple/7/stats.json"
    )
    assert runner._batch_meta_path(item) == (
        tmp_path / "samples/clip-c/omnidreams-model/simple/7/meta.json"
    )


def test_batch_metadata_writer_creates_meta_and_manifest(tmp_path) -> None:
    """Rank 0 writes per-record metadata plus the aggregate CSV."""
    runner = OmnidreamsRunner.__new__(OmnidreamsRunner)
    runner.is_rank_zero = True
    runner.config = SimpleNamespace(
        runner_name="omnidreams-model",
        output_dir=tmp_path,
        batch_results_path=tmp_path / "results.csv",
        camera_names=(),
        hdmap_video_paths=(),
        first_frame_paths=(),
        total_blocks=4,
        pipeline=SimpleNamespace(diffusion_model=SimpleNamespace(seed=7)),
    )
    runner.pipeline = SimpleNamespace(
        diffusion_model=SimpleNamespace(config=SimpleNamespace(seed=7))
    )
    item = _parse_batch_item(
        {
            "clip_id": "clip-d",
            "seed": 3,
            "output_dir": str(tmp_path / "clip-d"),
            "metadata": {"sensor_stack": "8.0"},
        },
        index=0,
    )

    runner._write_batch_metadata_and_result(
        item=item,
        status="completed",
        started_at="2026-05-22T00:00:00+00:00",
        finished_at="2026-05-22T00:01:00+00:00",
        exit_code=0,
        error=None,
    )

    meta = json.loads((tmp_path / "clip-d/meta.json").read_text(encoding="utf-8"))
    assert meta["clip_id"] == "clip-d"
    assert meta["sensor_stack"] == "8.0"
    assert meta["output_video"].endswith("clip-d/video.mp4")

    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert rows == [
        {
            "status": "completed",
            "clip_id": "clip-d",
            "dataset": "",
            "model": "omnidreams-model",
            "prompt_id": "",
            "seed": "3",
            "output_video": str(tmp_path / "clip-d/video.mp4"),
            "stats_json": str(tmp_path / "clip-d/stats.json"),
            "meta_json": str(tmp_path / "clip-d/meta.json"),
            "started_at": "2026-05-22T00:00:00+00:00",
            "finished_at": "2026-05-22T00:01:00+00:00",
            "exit_code": "0",
            "error": "",
        }
    ]


def test_batch_seed_reset_offsets_explicit_seed_by_rank() -> None:
    """Explicit per-row seeds follow the same rank offset as single runs."""
    runner = OmnidreamsRunner.__new__(OmnidreamsRunner)
    runner.global_rank = 2
    runner.config = SimpleNamespace(
        offset_seed_by_global_rank=True,
        pipeline=SimpleNamespace(diffusion_model=SimpleNamespace(seed=5)),
    )
    diffusion_model = SimpleNamespace(config=SimpleNamespace(seed=5), _rng=object())
    runner.pipeline = SimpleNamespace(diffusion_model=diffusion_model)
    item = _parse_batch_item({"clip_id": "clip-e", "seed": 10}, index=0)

    runner._reset_rollout_seed(item)

    assert diffusion_model.config.seed == 12
    assert diffusion_model._rng is None
