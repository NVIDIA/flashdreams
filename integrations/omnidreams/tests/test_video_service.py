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

import asyncio
from pathlib import Path

import pytest
from omnidreams.service.server import (
    DEFAULT_RUNNER_NAME,
    GenerationJob,
    ServiceConfig,
    build_runner_command,
    expand_to_num_views,
    resolve_runner_config,
    runner_layout,
    total_blocks_for_frame_count,
)

pytestmark = pytest.mark.ci_cpu


def test_total_blocks_for_frame_count_uses_full_omnidreams_chunks() -> None:
    assert total_blocks_for_frame_count(13, len_t=4) == 1
    assert total_blocks_for_frame_count(28, len_t=4) == 1
    assert total_blocks_for_frame_count(29, len_t=4) == 2
    assert total_blocks_for_frame_count(45, len_t=4) == 3


def test_total_blocks_rejects_videos_shorter_than_the_first_chunk() -> None:
    with pytest.raises(ValueError, match="requires at least 13 frames"):
        total_blocks_for_frame_count(12, len_t=4)


def test_expand_to_num_views_repeats_single_asset() -> None:
    path = Path("first_frame.png")

    assert expand_to_num_views((path,), 4, name="first_frame") == (
        path,
        path,
        path,
        path,
    )


def test_expand_to_num_views_rejects_partial_multiview_upload() -> None:
    with pytest.raises(ValueError, match="Upload exactly one file"):
        expand_to_num_views(
            (Path("left.mp4"), Path("right.mp4")),
            4,
            name="hdmap_video",
        )


def test_default_runner_layout_is_public_single_view() -> None:
    layout = runner_layout(DEFAULT_RUNNER_NAME)
    cfg = resolve_runner_config(DEFAULT_RUNNER_NAME)

    assert layout.num_views == 1
    assert layout.len_t == 2
    assert cfg.pipeline.diffusion_model.transformer.checkpoint_path.startswith(
        "https://huggingface.co/nvidia/omni-dreams-models/"
    )


def test_build_runner_command_passes_service_managed_overrides() -> None:
    loop = asyncio.new_event_loop()
    try:
        job = GenerationJob(
            job_id="job",
            runner_name=DEFAULT_RUNNER_NAME,
            prompt="city street",
            first_frame_paths=(Path("first.png"),),
            hdmap_video_paths=(Path("hdmap.mp4"),),
            total_blocks=7,
            output_dir=Path("out"),
            future=loop.create_future(),
        )
        config = ServiceConfig(
            runner_command=("flashdreams-run",),
            runner_args=("--pipeline.diffusion-model.seed", "123"),
        )

        command = build_runner_command(config=config, job=job)
    finally:
        loop.close()

    assert command == [
        "flashdreams-run",
        DEFAULT_RUNNER_NAME,
        "--prompt",
        "city street",
        "--first-frame-paths",
        "first.png",
        "--hdmap-video-paths",
        "hdmap.mp4",
        "--total-blocks",
        "7",
        "--output-dir",
        "out",
        "--pipeline.diffusion-model.seed",
        "123",
    ]
