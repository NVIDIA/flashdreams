# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt and autoregressive cache owned by the tutorial model loop."""

from dataclasses import dataclass
from typing import Any

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents


@dataclass
class TextToVideoState:
    """Mutable rollout state, accessed on the model thread."""

    pipeline: Any
    """Text-conditioned pipeline shared across sessions."""
    desc: SessionDesc
    """Pixel dimensions and frame layout requested by the runtime."""
    total_blocks: int
    """Maximum blocks the selected model supports per prompt."""
    prompt: str = ""
    """Current prompt; empty text pauses generation."""
    cache: Any = None
    """Per-prompt cache, created lazily on the model thread."""
    index: int = 0
    """Next autoregressive block, independent of the runtime step index."""

    def set_prompt(self, prompt: str) -> None:
        """Start a fresh rollout or pause generation for empty text."""
        self.prompt = prompt.strip()
        self.cache = None
        self.index = 0


class TextToVideoLoop(IModelLoop[TextToVideoState]):
    """Generate bounded rollouts while waiting for further UI prompts."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate and finalize one block, or wait for another prompt."""
        state = self.state
        if not state.prompt or state.index >= state.total_blocks:
            return []
        if state.cache is None:
            ratio = state.pipeline.decoder.spatial_compression_ratio
            state.cache = state.pipeline.initialize_cache(
                text=[state.prompt],
                height=state.desc.video_height // ratio,
                width=state.desc.video_width // ratio,
            )
        frames = state.pipeline.generate(
            autoregressive_index=state.index, cache=state.cache
        )
        metrics = state.pipeline.finalize(
            autoregressive_index=state.index, cache=state.cache
        )
        state.index += 1
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=int(frames.shape[0]),
                output_layout=state.desc.output_layout,
                metrics=metrics,
            )
        ]

    def reset(self) -> None:
        """Restart the current prompt after a runtime reset."""
        self.state.set_prompt(self.state.prompt)

    def close(self) -> None:
        """Release the per-prompt cache."""
        self.state.cache = None
