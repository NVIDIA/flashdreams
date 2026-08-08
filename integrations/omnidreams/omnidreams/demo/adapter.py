# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OmniDreams adapter for the shared demo API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omnidreams.config import OMNIDREAMS_CONFIGS, OMNIDREAMS_RUNNERS

from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InputCanonicalizer,
    InputField,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    Mp4OutputSpec,
    PreparedScenario,
)
from flashdreams.runtime.interfaces import InferenceRuntime

from .replay import (
    OmnidreamsReplayRuntime,
    OmnidreamsReplayRuntimeOptions,
    PipelineFactory,
)
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_MODEL_ID,
    resolve_replay_scenario,
)

ReplayRuntimeFactory = Callable[..., InferenceRuntime]


class OmnidreamsDemoAdapter:
    """Model-owned OmniDreams adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        replay_runtime_factory: ReplayRuntimeFactory = OmnidreamsReplayRuntime,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self._replay_runtime_factory = replay_runtime_factory
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()

    @property
    def model_id(self) -> str:
        return OMNIDREAMS_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return InferenceInputSchema(
            global_conditioning_fields=(
                InputField(
                    name="scenario",
                    input_modality="omnidreams/replay-scenario",
                    description="Resolved OmniDreams replay scenario.",
                ),
            )
        )

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return None

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4",)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "OmniDreams prepare_scenario currently supports only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if not isinstance(spec.output, Mp4OutputSpec):
            raise ValueError("OmniDreams replay demo currently requires MP4 output.")
        scenario = resolve_replay_scenario(
            spec.scenario,
            default_prompt=self._default_replay_prompt(spec.config),
        )
        return PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={"scenario": scenario},
            ),
            source_schema=UserInputSchema(description="fixed OmniDreams replay input"),
            canonicalizer=InputCanonicalizer(),
            mapping=self._mapping,
            metadata={
                "model_id": self.model_id,
                "preset_id": self._preset_id(spec.config),
                "num_views": len(scenario.camera_names),
            },
        )

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(
                f"OmniDreams adapter requires model_id={self.model_id!r}, "
                f"got {config.model_id!r}."
            )
        self._pipeline_config(config)

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self._replay_runtime_factory(
            config=config,
            options=OmnidreamsReplayRuntimeOptions(
                pipeline_config=self._pipeline_config(config),
                pipeline_factory=self._pipeline_factory,
            ),
        )

    def _preset_id(self, config: InferenceConfig | None) -> str:
        return (
            DEFAULT_OMNIDREAMS_PRESET
            if config is None or config.preset_id is None
            else config.preset_id
        )

    def _pipeline_config(self, config: InferenceConfig) -> Any:
        custom = config.runtime_options.get("pipeline_config")
        if custom is not None:
            return custom
        preset_id = self._preset_id(config)
        try:
            return OMNIDREAMS_CONFIGS[preset_id]
        except KeyError as exc:
            supported = ", ".join(sorted(OMNIDREAMS_CONFIGS))
            raise ValueError(
                f"Unsupported OmniDreams preset_id={preset_id!r}. "
                f"Supported presets: {supported}."
            ) from exc

    def _default_replay_prompt(self, config: InferenceConfig | None) -> str:
        runner = OMNIDREAMS_RUNNERS.get(self._preset_id(config))
        return "" if runner is None else str(getattr(runner, "prompt", ""))


__all__ = [
    "OmnidreamsDemoAdapter",
    "ReplayRuntimeFactory",
]
