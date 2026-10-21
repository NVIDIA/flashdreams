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

"""Checkpoint action normalization contracts on CPU."""

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_bimanual import action_format

pytestmark = pytest.mark.ci_cpu


def test_checkpoint_quantiles_and_reserved_dimensions() -> None:
    metadata = action_format.load_agibot_normalizer()
    raw = np.asarray(
        [metadata["global"]["q01"], metadata["global"]["q99"]],
        dtype=np.float32,
    )
    normalized = action_format.normalize_action(raw)
    assert normalized.shape == (2, 59)
    np.testing.assert_allclose(normalized, [[-1.0] * 59, [1.0] * 59], atol=1e-6)
    np.testing.assert_allclose(
        action_format.denormalize_action(normalized), raw, atol=1e-7
    )
    assert action_format.NORMALIZER_PATH.is_file()
    assert not action_format.NORMALIZER_PATH.is_symlink()


def test_normalization_does_not_clip_valid_motion() -> None:
    offset, scale = action_format.load_agibot_norm_stats()
    raw = offset + 3.0 * scale
    np.testing.assert_allclose(
        action_format.normalize_action(raw), np.arcsinh(3) / np.arcsinh(1), atol=1e-6
    )
    assert not offset.flags.writeable and not scale.flags.writeable


@pytest.mark.parametrize(
    "value", [np.ones((1, 29)), np.ones((1, 64)), np.full((1, 59), np.nan)]
)
def test_reject_invalid_action_rows(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        action_format.pad_action(value)


def test_stationary_actions_have_identity_rotations() -> None:
    raw = action_format.identity_raw_action(2)
    for start in (0, 9, 34):
        np.testing.assert_array_equal(
            raw[:, start : start + 9], [[0, 0, 0, 1, 0, 0, 0, 1, 0]] * 2
        )
    np.testing.assert_array_equal(raw[:, [33, 58]], 0)


def test_model_geometry_uses_complete_content_cells() -> None:
    assert action_format.logical_latent_geometry(360, 640) == (22, 40)
    assert action_format.TOKEN_GRID == (11, 20)
    assert (action_format.OUTPUT_HEIGHT, action_format.OUTPUT_WIDTH) == (352, 640)
    with pytest.raises(ValueError, match="geometry"):
        action_format.logical_latent_geometry(720, 1280)


@pytest.mark.parametrize("shape", [(0, 640), (360, 0), (-1, 640), (360, -1)])
def test_reject_nonpositive_content_geometry(shape) -> None:
    with pytest.raises(ValueError, match="positive"):
        action_format.logical_latent_geometry(*shape, expected=None)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((15, 640), (1, 40)), ((360, 15), (22, 1)), ((1, 1), (1, 1))],
)
def test_training_geometry_retains_at_least_one_cell(shape, expected) -> None:
    assert action_format.logical_latent_geometry(*shape, expected=None) == expected


def test_ga_normalization_matches_training_formula_and_masks_missing_fingers() -> None:
    # Independent Torch implementation of the exported global_asinh contract.
    stats = action_format.load_agibot_normalizer()["global"]
    low, high = (torch.tensor(stats[key]) for key in ("q01", "q99"))
    raw = action_format.identity_raw_action(3)
    raw[:, 9:12] = [[0, 0, 0], [0.1, -0.2, 0.3], [1, 2, -3]]
    raw[:, 33] = np.array([0, 0.5, 1]) * action_format.GRIPPER_OPEN_WIDTH_METERS
    values = torch.from_numpy(raw)
    expected = torch.asinh(
        (values - (low + high) / 2) / ((high - low).clamp(min=1e-8) / 2)
    ) / torch.asinh(torch.tensor(1.0))
    expected[:, 18:33] = 0
    expected[:, 43:58] = 0
    actual = action_format.normalize_and_pad(raw)
    np.testing.assert_allclose(actual[:, :59], expected.numpy(), atol=3e-7, rtol=3e-7)
    np.testing.assert_array_equal(actual[:, 18:33], 0)
    np.testing.assert_array_equal(actual[:, 43:58], 0)
    np.testing.assert_array_equal(actual[:, 59:], 0)
    assert np.count_nonzero(actual[:, :9]) > 0  # Stationary ego rotation is valid.
    assert actual[-1, 33] > 1  # AgiBot opens to .12 m, not the .082 m reference.


def test_normalization_does_not_mutate_input() -> None:
    raw = action_format.identity_raw_action(2)
    saved = raw.copy()
    normalized = action_format.normalize_and_pad(raw)
    normalized[:] = 10
    np.testing.assert_array_equal(raw, saved)
