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

import io
import zipfile
from pathlib import Path

import pytest
from omnidreams.conditioning.world_scenario.data_loaders import (
    list_loaders,
    load_scene,
)

pytestmark = pytest.mark.ci_gpu


def test_load_scene_direct_from_example_zip(
    tmp_path: Path, example_scene_zip_bytes: bytes
) -> None:
    loaders = list_loaders()
    assert "clipgt" in loaders, (
        "clipgt loader is not registered; direct scene loading cannot work. "
        f"Registered loaders: {loaders}"
    )

    extracted_scene_dir = tmp_path / "clipgt_scene"
    extracted_scene_dir.mkdir()
    with zipfile.ZipFile(io.BytesIO(example_scene_zip_bytes), "r") as zf:
        zf.extractall(extracted_scene_dir)

    scene_data = load_scene(
        extracted_scene_dir,
        camera_names=["camera_front_wide_120fov"],
        max_frames=8,
        input_pose_fps=30,
        resize_resolution_hw=(704, 1280),
    )

    assert scene_data.scene_id
    assert scene_data.num_frames > 0
    assert len(scene_data.ego_poses) == scene_data.num_frames
    assert "camera_front_wide_120fov" in scene_data.camera_models
