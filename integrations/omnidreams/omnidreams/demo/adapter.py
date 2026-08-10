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
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoSpec,
    PreparedScenario,
)
from flashdreams.runtime.demo.session_inputs import ModelInputProvider
from flashdreams.runtime.interfaces import InferenceRuntime

from .providers import (
    LudusSceneConditioningProvider,
    PrecomputedHDMapProvider,
    keyboard_driving_user_input_schema,
    precomputed_hdmap_inference_input_schema,
)
from .runtime import (
    OmnidreamsRuntime,
    OmnidreamsRuntimeOptions,
    PipelineFactory,
)
from .spec import (
    DEFAULT_OMNIDREAMS_PRESET,
    OMNIDREAMS_CONDITIONING_LUDUS,
    OMNIDREAMS_CONDITIONING_MODES,
    OMNIDREAMS_CONDITIONING_PRECOMPUTED,
    OMNIDREAMS_MODEL_ID,
    conditioning_mode_from_scenario,
    resolve_ludus_replay_scenario,
    resolve_replay_scenario,
)

RuntimeFactory = Callable[..., InferenceRuntime]
ReplayRuntimeFactory = RuntimeFactory


class OmnidreamsDemoAdapter:
    """Model-owned OmniDreams adapter consumed by shared demo launchers."""

    def __init__(
        self,
        *,
        runtime_factory: RuntimeFactory | None = None,
        replay_runtime_factory: ReplayRuntimeFactory | None = None,
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        if runtime_factory is not None and replay_runtime_factory is not None:
            raise ValueError(
                "Specify either runtime_factory or replay_runtime_factory, not both."
            )
        self._runtime_factory = (
            runtime_factory
            if runtime_factory is not None
            else replay_runtime_factory or OmnidreamsRuntime
        )
        self._pipeline_factory = pipeline_factory
        self._mapping = IdentityInputMapping()

    @property
    def model_id(self) -> str:
        return OMNIDREAMS_MODEL_ID

    @property
    def inference_input_schema(self) -> InferenceInputSchema:
        return precomputed_hdmap_inference_input_schema()

    @property
    def canonical_input_schema(self) -> CanonicalInputSchema | None:
        return None

    def default_input_mapping(self) -> IdentityInputMapping:
        return self._mapping

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null")

    def supported_conditioning_modes(self) -> tuple[str, ...]:
        return OMNIDREAMS_CONDITIONING_MODES

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "OmniDreams prepare_scenario currently supports only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if spec.output.mode not in self.supported_output_modes():
            raise ValueError(
                "OmniDreams replay demo supports output modes "
                f"{self.supported_output_modes()}, got {spec.output.mode!r}."
            )
        conditioning_mode = conditioning_mode_from_scenario(spec.scenario)
        if conditioning_mode == OMNIDREAMS_CONDITIONING_PRECOMPUTED:
            scenario = resolve_replay_scenario(
                spec.scenario,
                default_prompt=self._default_replay_prompt(spec.config),
            )
            source_schema = UserInputSchema(description="fixed OmniDreams replay input")
        elif conditioning_mode == OMNIDREAMS_CONDITIONING_LUDUS:
            scenario = resolve_ludus_replay_scenario(spec.scenario)
            source_schema = keyboard_driving_user_input_schema()
        else:
            raise ValueError(
                f"Unsupported OmniDreams conditioning mode: {conditioning_mode!r}."
            )
        return PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={"scenario": scenario},
            ),
            source_schema=source_schema,
            canonicalizer=InputCanonicalizer(),
            mapping=self._mapping,
            metadata={
                "conditioning_mode": conditioning_mode,
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
        return self._runtime_factory(
            config=config,
            options=OmnidreamsRuntimeOptions(
                pipeline_config=self._pipeline_config(config),
                pipeline_factory=self._pipeline_factory,
            ),
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> ModelInputProvider:
        if spec.input_mode != "replay":
            raise ValueError(
                "OmniDreams replay providers currently support only "
                f"input_mode='replay', got {spec.input_mode!r}."
            )
        if spec.config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        conditioning_mode = str(
            scenario.metadata.get(
                "conditioning_mode",
                conditioning_mode_from_scenario(spec.scenario),
            )
        )
        if conditioning_mode == OMNIDREAMS_CONDITIONING_LUDUS:
            return LudusSceneConditioningProvider(
                scenario=scenario,
                config=spec.config,
            )
        if conditioning_mode != OMNIDREAMS_CONDITIONING_PRECOMPUTED:
            raise ValueError(
                f"Unsupported OmniDreams conditioning mode: {conditioning_mode!r}."
            )
        return PrecomputedHDMapProvider(
            scenario=scenario,
            config=spec.config,
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
    "RuntimeFactory",
]
