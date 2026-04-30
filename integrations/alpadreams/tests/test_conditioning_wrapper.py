"""Unit tests for the Alpadreams conditioning wrapper.

These tests intentionally use lightweight fakes to validate wrapper-specific
contracts quickly and deterministically: start/continue/finalize state flow,
autoregressive index progression, dtype/range conversion, skip-generation
behavior, and cleanup/error handling.

Rationale: this complements heavier gRPC integration coverage (which validates
end-to-end wiring with real server/runtime dependencies) and lower-level core
tests in ``flashdreams`` (which validate internals like caches), by providing
fast module-level regression protection for wrapper logic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from alpadreams.conditioning.conditioning_wrapper import (
    AlpadreamsConditioningState,
    AlpadreamsConditioningWrapper,
    TextPrompt,
)
from torch import nn


class _FakePipeline:
    def __init__(self) -> None:
        self.diffusion_model = SimpleNamespace(rng=torch.Generator(device="cpu"))
        self.finalize_calls: list[tuple[int, object]] = []

    def initialize_cache(
        self, *, text: list[list[str]], image: torch.Tensor, view_names: list[str]
    ) -> object:
        del text, view_names
        return SimpleNamespace(
            image_shape=tuple(image.shape),
            autoregressive_index=None,
        )

    def generate(
        self, *, autoregressive_index: int, hdmap: torch.Tensor, cache: object
    ) -> torch.Tensor:
        # Mirror StreamInferencePipeline behavior: generate() records the latest AR
        # index on the cache so callers can derive the next step.
        cache.autoregressive_index = autoregressive_index  # ty:ignore[unresolved-attribute]
        # Return model-range tensor in [-1, 1] so wrapper conversion can be validated.
        return torch.linspace(
            -1.0,
            1.0,
            steps=hdmap.numel(),
            dtype=torch.float32,
            device=hdmap.device,
        ).reshape(hdmap.shape)

    def finalize(self, *, autoregressive_index: int, cache: object) -> None:
        self.finalize_calls.append((autoregressive_index, cache))


class _FakeRenderer:
    def __init__(self, *, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.cleaned_up = False

    def render_all_frames_and_cameras(
        self,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
    ) -> torch.Tensor:
        del camera_poses_per_view
        n_views = len(camera_names)
        n_frames = len(frame_timestamps_us)
        # [V, T, 3, H, W]
        return torch.full(
            (n_views, n_frames, 3, self.height, self.width),
            fill_value=127,
            dtype=torch.uint8,
        )

    def cleanup(self) -> None:
        self.cleaned_up = True


def _make_wrapper(
    *,
    n_cameras: int = 1,
    frame_chunk_size: int = 4,
    resolution_wh: tuple[int, int] = (8, 6),
) -> AlpadreamsConditioningWrapper:
    wrapper = AlpadreamsConditioningWrapper.__new__(AlpadreamsConditioningWrapper)
    nn.Module.__init__(wrapper)
    wrapper._device = torch.device("cpu")
    wrapper._n_cameras = n_cameras
    wrapper.video_resolution_wh = resolution_wh
    wrapper._rollout_seed = 42
    wrapper.fps = 30
    wrapper.frame_chunk_size = frame_chunk_size
    wrapper.initial_frame_chunk_size = 1 + (frame_chunk_size // 4 - 1) * 4
    wrapper.pipeline = _FakePipeline()
    return wrapper


def _make_camera_poses(
    camera_names: list[str], num_frames: int
) -> dict[str, torch.Tensor]:
    per_camera = {}
    for name in camera_names:
        per_camera[name] = torch.eye(4).repeat(num_frames, 1, 1)
    return per_camera


def test_start_continue_and_finalize_flow() -> None:
    wrapper = _make_wrapper()
    renderer = _FakeRenderer(height=6, width=8)
    camera_names = ["front"]
    text_prompts = [TextPrompt(positive="drive")]

    start_output = wrapper.start_generation(
        text_prompts=text_prompts,
        initial_rgb_frames=torch.zeros((1, 1, 3, 6, 8), dtype=torch.uint8),
        renderer=renderer,  # ty:ignore[invalid-argument-type]
        camera_names=camera_names,
        camera_poses_per_view=_make_camera_poses(camera_names, num_frames=1),
        frame_timestamps_us=[1_000_000],
    )

    assert isinstance(start_output.state, AlpadreamsConditioningState)
    assert start_output.rgb_frames is not None
    assert start_output.rgb_frames.dtype == torch.uint8
    assert start_output.rgb_frames.shape == (1, 1, 1, 3, 6, 8)
    assert start_output.finalization_state == {"autoregressive_index": 0}
    assert start_output.state.pipeline_cache is not None
    assert start_output.state.pipeline_cache.autoregressive_index == 0

    continue_output = wrapper.continue_generation(
        state=start_output.state,
        camera_names=camera_names,
        camera_poses_per_view=_make_camera_poses(camera_names, num_frames=4),
        frame_timestamps_us=[1_033_333, 1_066_666, 1_099_999, 1_133_332],
    )

    assert continue_output.rgb_frames is not None
    assert continue_output.rgb_frames.dtype == torch.uint8
    assert continue_output.rgb_frames.shape == (1, 1, 4, 3, 6, 8)
    assert continue_output.finalization_state == {"autoregressive_index": 1}
    assert continue_output.state.pipeline_cache is not None
    assert continue_output.state.pipeline_cache.autoregressive_index == 1

    assert continue_output.state.pipeline_cache is not None
    wrapper.finalize_block_generation(
        continue_output.state.pipeline_cache,
        continue_output.finalization_state,
    )
    assert wrapper.pipeline.finalize_calls == [
        (1, continue_output.state.pipeline_cache)
    ]


def test_skip_generation_and_cleanup() -> None:
    wrapper = _make_wrapper()
    renderer = _FakeRenderer(height=6, width=8)
    camera_names = ["front"]

    output = wrapper.start_generation(
        text_prompts=[TextPrompt(positive="drive")],
        initial_rgb_frames=torch.zeros((1, 1, 3, 6, 8), dtype=torch.uint8),
        renderer=renderer,  # ty:ignore[invalid-argument-type]
        camera_names=camera_names,
        camera_poses_per_view=_make_camera_poses(camera_names, num_frames=1),
        frame_timestamps_us=[1_000_000],
        skip_video_generation=True,
    )
    assert output.rgb_frames is None
    assert output.state.pipeline_cache is None

    with pytest.raises(ValueError, match="pipeline_cache is None"):
        wrapper.continue_generation(
            state=output.state,
            camera_names=camera_names,
            camera_poses_per_view=_make_camera_poses(camera_names, num_frames=4),
            frame_timestamps_us=[1, 2, 3, 4],
            skip_video_generation=False,
        )

    wrapper.cleanup(output.state)
    assert renderer.cleaned_up is True


def test_start_generation_rejects_mismatched_timestamp_length() -> None:
    wrapper = _make_wrapper()
    renderer = _FakeRenderer(height=6, width=8)
    camera_names = ["front"]

    with pytest.raises(ValueError, match="frame_timestamps_us length"):
        wrapper.start_generation(
            text_prompts=[TextPrompt(positive="drive")],
            initial_rgb_frames=torch.zeros((1, 1, 3, 6, 8), dtype=torch.uint8),
            renderer=renderer,  # ty:ignore[invalid-argument-type]
            camera_names=camera_names,
            camera_poses_per_view=_make_camera_poses(camera_names, num_frames=2),
            frame_timestamps_us=[1, 2],
        )
