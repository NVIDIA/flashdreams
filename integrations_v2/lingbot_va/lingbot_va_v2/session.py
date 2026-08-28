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

"""Finite V2 session adapter for LingBot-VA Robotwin I2AV rollouts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from lingbot_va.constants import (
    ROBOTWIN_ACTION_PER_FRAME,
    ROBOTWIN_FRAME_CHUNK_SIZE,
    ROBOTWIN_USED_ACTION_CHANNEL_IDS,
)
from lingbot_va.engine import (
    LingbotVAEngineConfig,
    LingbotVAEngineOutput,
    expected_output_shape,
)

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.tensor_artifact import (
    TensorArtifactOutput,
    TensorArtifactSchema,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

ACTIONS_SCHEMA = TensorArtifactSchema(
    name="actions",
    dimension_names=("step", "channel"),
    concatenate_axis=0,
)
"""Tensor artifact schema for denormalized Robotwin actions."""


class LingbotVAEngineLike(Protocol):
    """Minimal engine boundary used by the V2 adapter and CPU stand-ins."""

    def run(self) -> LingbotVAEngineOutput:
        """Generate one complete fixed-input rollout."""
        ...

    def close(self) -> None:
        """Release partially or fully initialized model state."""
        ...


EngineFactory = Callable[[LingbotVAEngineConfig], LingbotVAEngineLike]
"""Create a session-owned engine from immutable application config."""


@dataclass(slots=True)
class LingbotVAModelState:
    """Mutable state owned exclusively by the model loop."""

    config: LingbotVAEngineConfig
    """Model, checkpoint, and fixed input settings."""

    session_desc: SessionDesc
    """Canonical output contract for the session."""

    engine_factory: EngineFactory
    """Factory used to construct the destructive one-run engine."""

    engine: LingbotVAEngineLike | None = None
    """Session-owned engine, created lazily on the model thread."""

    generated: bool = False
    """Whether this finite session has produced its only result."""


class LingbotVAModelLoop(IModelLoop[LingbotVAModelState]):
    """Generate one complete fixed-input video/action rollout in one step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Run the offline rollout; live user events are outside this adapter's scope."""
        del events
        if self.state.generated:
            raise RuntimeError("LingBot-VA has already generated this rollout.")
        if self.state.engine is None:
            self.state.engine = self.state.engine_factory(self.state.config)
        output = self.state.engine.run()
        _validate_engine_output(self.state.config, output)
        self.state.generated = True
        return [
            StepResult(
                step_index=step_index,
                output=output.video,
                frame_count=output.video.shape[0],
                output_layout=self.state.session_desc.output_layout,
                metrics=dict(output.metrics),
                tensor_artifacts=(
                    TensorArtifactOutput(
                        schema=ACTIONS_SCHEMA,
                        tensor=output.actions,
                    ),
                ),
            )
        ]

    def is_finished(self) -> bool:
        """Return whether the only rollout has completed."""
        return self.state.generated

    def reset(self) -> None:
        """Discard the destructive engine; the next step creates a new one."""
        self.close()
        self.state.generated = False

    def close(self) -> None:
        """Idempotently close the session-owned engine."""
        engine = self.state.engine
        self.state.engine = None
        if engine is not None:
            engine.close()


class LingbotVASession(ISession):
    """Own one isolated, resettable LingBot-VA rollout."""

    def __init__(
        self,
        config: LingbotVAEngineConfig,
        session_desc: SessionDesc,
        engine_factory: EngineFactory,
    ) -> None:
        """
        Args:
            config: Immutable model and input settings.
            session_desc: Canonical Robotwin output description.
            engine_factory: Factory used lazily on the model thread.
        """
        self._config = config
        self._session_desc = session_desc
        self._engine_factory = engine_factory
        self._model_loop: LingbotVAModelLoop | None = None

    def init(self) -> None:
        """Register one finite model loop; model loading remains lazy."""
        if self._model_loop is not None:
            raise RuntimeError("LingbotVASession.init() may only run once.")
        model_loop = self.register_model_loop(
            LingbotVAModelLoop,
            state=LingbotVAModelState(
                config=self._config,
                session_desc=self._session_desc,
                engine_factory=self._engine_factory,
            ),
        )
        assert isinstance(model_loop, LingbotVAModelLoop)
        self._model_loop = model_loop

    @property
    def session_desc(self) -> SessionDesc:
        """Return the immutable output contract for this session."""
        return self._session_desc

    def close(self) -> None:
        """Close a loop even when the runtime never started it."""
        if self._model_loop is not None:
            self._model_loop.close()


def _validate_engine_output(
    config: LingbotVAEngineConfig,
    output: LingbotVAEngineOutput,
) -> None:
    """Keep incorrect model shapes from reaching generic runtime sinks."""
    expected_video = expected_output_shape(config)
    if tuple(output.video.shape) != expected_video:
        raise ValueError(
            f"LingBot-VA engine returned video shape {tuple(output.video.shape)}; "
            f"expected {expected_video}."
        )
    expected_action_shape = (
        config.num_chunks * ROBOTWIN_FRAME_CHUNK_SIZE * ROBOTWIN_ACTION_PER_FRAME,
        len(ROBOTWIN_USED_ACTION_CHANNEL_IDS),
    )
    if tuple(output.actions.shape) != expected_action_shape:
        raise ValueError(
            f"LingBot-VA engine returned action shape {tuple(output.actions.shape)}; "
            f"expected {expected_action_shape}."
        )
    if not output.video.is_floating_point() or not output.actions.is_floating_point():
        raise TypeError("LingBot-VA video and action outputs must be floating point.")
    _validate_metrics(output.metrics)


def _validate_metrics(metrics: Mapping[str, float]) -> None:
    """Require numeric model metrics before constructing a StepResult."""
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in metrics.values()
    ):
        raise TypeError("LingBot-VA engine metrics must be numeric.")


__all__ = [
    "ACTIONS_SCHEMA",
    "EngineFactory",
    "LingbotVAModelLoop",
    "LingbotVASession",
]
