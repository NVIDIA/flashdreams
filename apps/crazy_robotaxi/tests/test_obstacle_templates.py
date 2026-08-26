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

"""CPU tests for the bundled live-edit obstacle template catalog."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from crazy_robotaxi.live_edit.obstacle_template_authoring import (
    catalog_arrays_from_records,
)
from crazy_robotaxi.live_edit.obstacle_templates import (
    load_obstacle_template_catalog,
    load_obstacle_template_catalog_from_file,
)

pytestmark = pytest.mark.ci_cpu


def _observation(
    track_id: str,
    timestamp_us: int,
    *,
    category: str = "Car",
    center: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> dict[str, Any]:
    return {
        "key": {"timestamp_micros": timestamp_us},
        "obstacle": {
            "trackline_id": track_id,
            "category": category,
            "center": dict(zip(("x", "y", "z"), center, strict=True)),
            "size": {"x": 4.5, "y": 1.8, "z": 1.5},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }


def test_bundled_catalog_has_expected_pr494_template_sets() -> None:
    catalog = load_obstacle_template_catalog()

    moving = catalog.moving(
        min_drift_m=15.0,
        min_coverage_s=4.0,
        length_range_m=(3.4, 5.6),
    )
    parked = catalog.parked(length_range_m=(3.4, 5.6))

    assert len(catalog.templates) == 668
    assert len(moving) == 63
    assert len(parked) == 236
    assert {template.object_type for template in catalog.templates} == {"Car", "Truck"}
    assert all(template.timestamps_us[0] == 0 for template in catalog.templates)
    assert all(
        np.allclose(template.translations_local_m[0], 0.0)
        for template in catalog.templates
    )


def test_authoring_arrays_are_order_independent_and_load_without_pickle(
    tmp_path,
) -> None:
    rows = [
        _observation("b", 2_000_000, center=(4.0, 0.0, 1.5)),
        _observation("a", 2_000_000, category="Truck", center=(0.0, 3.0, 1.0)),
        _observation("b", 1_000_000, center=(1.0, 0.0, 1.5)),
        _observation("a", 1_000_000, category="Truck", center=(0.0, 1.0, 1.0)),
        _observation("ignored", 1_000_000, category="Pedestrian"),
        _observation("ignored", 2_000_000, category="Pedestrian"),
    ]
    ground = np.asarray(
        [[-1.0, -1.0, 0.25], [0.0, 1.0, 0.25], [1.0, 0.0, 0.25]],
        dtype=np.float32,
    )

    first = catalog_arrays_from_records(rows, ground)
    second = catalog_arrays_from_records(tuple(reversed(rows)), ground)
    assert all(np.array_equal(first[name], second[name]) for name in first)

    output = tmp_path / "catalog.npz"
    with output.open("wb") as handle:
        np.savez_compressed(handle, **first)
    catalog = load_obstacle_template_catalog_from_file(output)

    assert len(catalog.templates) == 2
    assert catalog.templates[0].object_type == "Truck"
    assert catalog.templates[0].translations_local_m[-1] == pytest.approx(
        [0.0, 2.0, 0.0]
    )
    assert catalog.templates[1].translations_local_m[-1] == pytest.approx(
        [3.0, 0.0, 0.0]
    )
    assert catalog.templates[0].source_ground_offset_m == pytest.approx(0.75)
    assert catalog.templates[1].source_ground_offset_m == pytest.approx(1.25)
