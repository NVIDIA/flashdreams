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

"""CPU checks for UI back-buffer preparation."""

import pytest
import torch

from flashdreams.runtime_v2.ui_compositing import prepare_ui_back_buffer

pytestmark = pytest.mark.ci_cpu


def test_prepare_ui_back_buffer_normalizes_and_resizes_integer_frames() -> None:
    back_buffer = torch.tensor([0, 255, 128], dtype=torch.uint8).reshape(3, 1, 1)
    overlay = torch.zeros((4, 2, 3), dtype=torch.float32)

    prepared = prepare_ui_back_buffer(back_buffer, overlay)

    assert prepared is not None
    assert prepared.shape == (3, 2, 3)
    assert prepared.dtype is overlay.dtype
    assert prepared.device == overlay.device
    torch.testing.assert_close(
        prepared[:, 0, 0],
        torch.tensor([-1.0, 1.0, 1.0 / 255.0]),
    )
