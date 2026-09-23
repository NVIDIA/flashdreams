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
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

import flashdreams.core.attention.multiview.rollout as rollout_module
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
        view_text_tokens: tuple[int, ...] | None,
        condition_tokens: Tensor | None,
        fps: float,
        history_slots: int,
        control_ranges: tuple[tuple[int, int], ...] | None,
        control_slot_frames: int | None,
        attention_pattern: str,
        attention_scope: str,
        decomposed_temporal_window_seconds: float | None,
        view_offsets: bool,
        use_block_mask: bool,
    ) -> MultiViewRolloutState:
        """Record all generic-to-model boundary values and return test state."""
        self.prepared = {
            "geometry": geometry,
            "controls": controls,
            "text_ids": text_ids,
            "view_text_tokens": view_text_tokens,
            "condition_tokens": condition_tokens,
            "fps": fps,
            "history_slots": history_slots,
            "control_ranges": control_ranges,
            "control_slot_frames": control_slot_frames,
            "attention_pattern": attention_pattern,
            "attention_scope": attention_scope,
            "decomposed_temporal_window_seconds": (decomposed_temporal_window_seconds),
            "view_offsets": view_offsets,
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


def test_rollout_requests_position_ids_on_the_model_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    controls, text_ids, condition = _inputs()
    requested: list[torch.device | None] = []
    original = rollout_module.chunk_mrope_ids

    def tracked_chunk_mrope_ids(
        geometry: ClipGeometry,
        *,
        chunk_start: int,
        chunk_frames: int,
        fps: float,
        temporal_offset: float,
        view_offsets: bool,
        device: torch.device | None,
    ) -> Tensor:
        requested.append(device)
        return original(
            geometry,
            chunk_start=chunk_start,
            chunk_frames=chunk_frames,
            fps=fps,
            temporal_offset=temporal_offset,
            view_offsets=view_offsets,
            device=device,
        )

    monkeypatch.setattr(rollout_module, "chunk_mrope_ids", tracked_chunk_mrope_ids)

    run_rollout(
        model,
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
    )

    assert requested == [model.device, model.device]


def test_view_caption_and_position_options_cross_the_rollout_seam() -> None:
    model = _Model()
    controls, text_ids, condition = _inputs()

    run_rollout(
        model,
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        view_text_tokens=(1, 1),
        condition_tokens=condition,
        attention_scope="same_view",
        view_offsets=False,
    )

    assert model.prepared["view_text_tokens"] == (1, 1)
    assert model.prepared["attention_scope"] == "same_view"
    assert model.prepared["view_offsets"] is False
    first_mask = model.state.masks[0]
    assert isinstance(first_mask, Tensor)
    assert first_mask[:, :2].tolist() == [[True, False], [False, True]]


def test_block_mask_selection_crosses_the_protocol_boundary() -> None:
    """Supply the same visibility contract with the requested block geometry."""
    model = _Model()
    controls, text_ids, condition = _inputs()
    run_rollout(
        model,
        geometry=GEOMETRY,
        controls=controls,
        text_ids=text_ids,
        condition_tokens=condition,
        use_block_mask=True,
        mask_block_size=(16, 32),
    )

    assert model.state.masks
    for mask in model.state.masks:
        assert isinstance(mask, BlockMask)
        assert mask.BLOCK_SIZE == (16, 32)


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
    with pytest.raises(ValueError, match="entries for 2 views"):
        ChunkRollout(
            _Model(),
            geometry=GEOMETRY,
            controls=controls,
            text_ids=text_ids,
            view_text_tokens=(2,),
            condition_tokens=condition,
        )


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_rollout_rejects_invalid_fps_before_prefill(fps: float) -> None:
    model = _Model()
    controls, text_ids, condition = _inputs()

    with pytest.raises(ValueError, match="finite and positive"):
        ChunkRollout(
            model,
            geometry=GEOMETRY,
            controls=controls,
            text_ids=text_ids,
            condition_tokens=condition,
            fps=fps,
        )

    assert not model.prepared
