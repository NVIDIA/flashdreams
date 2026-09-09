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

"""CPU tests for the model-independent multi-view rollout seam."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from flashdreams.core.attention.multiview import (
    ChunkMask,
    ChunkRollout,
    ClipGeometry,
    ControlSource,
    MemoryLayout,
    MultiViewRolloutState,
    build_memory_layout,
    run_rollout,
)
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

pytestmark = pytest.mark.ci_cpu

GEOMETRY = ClipGeometry(
    num_views=2,
    frames_per_view=3,
    patch_h=1,
    patch_w=1,
    frames_per_chunk=1,
    condition_frames=1,
)


class _State(MultiViewRolloutState):
    """Tiny state adapter that records masks and committed frame ranges."""

    def __init__(self) -> None:
        self.history: list[tuple[int, int]] = []
        self.masks: list[ChunkMask] = []
        self.topups: list[int] = []

    @property
    def memory_layout(self) -> MemoryLayout:
        """Describe a full control track, conditioning, and committed history."""
        return build_memory_layout(
            GEOMETRY,
            history_frame_ranges=self.history,
            history_slot_frames=GEOMETRY.frames_per_chunk,
        )

    @property
    def num_text_tokens(self) -> int:
        """Use two cached prompt tokens."""
        return 2

    def top_up_control(self, needed_end: int) -> None:
        """Record the scheduler's requested control horizon."""
        self.topups.append(needed_end)

    def predict_velocity(
        self,
        latent: Tensor,
        timestep: Tensor,
        positions: Tensor,
        *,
        mask: ChunkMask,
    ) -> Tensor:
        """Return a simple deterministic velocity and record its mask."""
        del timestep, positions
        self.masks.append(mask)
        return torch.full_like(latent, 0.25)

    def commit(
        self,
        denoised: Tensor,
        positions: Tensor,
        *,
        mask: ChunkMask,
        frames: tuple[int, int],
    ) -> None:
        """Record clean replay inputs and expose the chunk as history."""
        del denoised, positions
        self.masks.append(mask)
        self.history.append(frames)


class _Model:
    """Minimal checkpoint adapter satisfying the public rollout protocol."""

    def __init__(self) -> None:
        self.state = _State()
        self.prepared: dict[str, object] = {}

    @property
    def device(self) -> torch.device:
        """Run the fixture on CPU."""
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        """Keep fixture values in float32."""
        return torch.float32

    @property
    def token_width(self) -> int:
        """Use one scalar per latent token."""
        return 1

    @property
    def latent_channels(self) -> int:
        """Use one latent channel."""
        return 1

    @property
    def latent_patch_size(self) -> int:
        """Use one spatial value per patch."""
        return 1

    @property
    def default_schedule(self) -> Sequence[float]:
        """Use one distilled denoising pass."""
        return [1.0]

    def prepare_rollout(
        self,
        *,
        geometry: ClipGeometry,
        controls: Tensor | ControlSource,
        text_ids: Tensor,
        condition_tokens: Tensor | None,
        fps: float,
        history_slots: int,
        control_ranges: tuple[tuple[int, int], ...] | None,
        control_slot_frames: int | None,
        use_block_mask: bool,
    ) -> MultiViewRolloutState:
        """Record all generic-to-model boundary values and return test state."""
        self.prepared = {
            "geometry": geometry,
            "controls": controls,
            "text_ids": text_ids,
            "condition_tokens": condition_tokens,
            "fps": fps,
            "history_slots": history_slots,
            "control_ranges": control_ranges,
            "control_slot_frames": control_slot_frames,
            "use_block_mask": use_block_mask,
        }
        return self.state


def _inputs() -> tuple[Tensor, Tensor, Tensor]:
    controls = torch.zeros(6, 1)
    text_ids = torch.tensor([3, 4])
    condition = torch.tensor([[10.0], [20.0]])
    return controls, text_ids, condition


def test_generic_rollout_owns_chunking_sampling_masks_and_output() -> None:
    """Drive an adapter through noisy passes, clean commits, and unpatchifying."""
    model = _Model()
    controls, text_ids, condition = _inputs()

    result = run_rollout(
        model,
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
        seed=7,
    )

    assert [(trace.start, trace.end) for trace in result.trace] == [(1, 2), (2, 3)]
    assert [trace.clean_pass for trace in result.trace] == [True, False]
    assert result.tokens.shape == (6, 1)
    assert result.latent.shape == (2, 1, 3, 1, 1)
    assert result.latent[:, :, 0].flatten().tolist() == [10.0, 20.0]
    assert model.state.topups == [2, 3]
    assert model.state.history == [(1, 2)]
    assert all(isinstance(mask, Tensor) for mask in model.state.masks)


def test_block_mask_selection_crosses_the_protocol_boundary() -> None:
    """Supply the same visibility contract in block-sparse form."""
    model = _Model()
    controls, text_ids, condition = _inputs()
    rollout = ChunkRollout(
        model,
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
        use_block_mask=True,
    )

    rollout.step()

    assert model.state.masks
    assert all(isinstance(mask, BlockMask) for mask in model.state.masks)


def test_seeded_rollouts_are_reproducible_through_an_adapter() -> None:
    """Keep random sampling deterministic independently of model implementation."""
    controls, text_ids, condition = _inputs()

    first = run_rollout(
        _Model(),
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
        seed=19,
    )
    second = run_rollout(
        _Model(),
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
        seed=19,
    )

    assert torch.equal(first.tokens, second.tokens)


def test_rollout_validates_conditioning_and_schedule_before_prefill() -> None:
    """Reject invalid generic inputs before invoking the checkpoint adapter."""
    controls, text_ids, condition = _inputs()

    with pytest.raises(ValueError, match="condition_tokens is required"):
        ChunkRollout(_Model(), geometry=GEOMETRY, controls=controls, text_ids=text_ids)
    with pytest.raises(ValueError, match="strictly decreasing"):
        ChunkRollout(
            _Model(),
            geometry=GEOMETRY,
            controls=controls,
            text_ids=text_ids,
            condition_tokens=condition,
            schedule=[0.5, 0.75],
        )
