# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashVSR model/demo adapter for shared replay and WebRTC run modes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from flashdreams.runtime import InferenceInput, InputCanonicalizer, UserInputSchema
from flashdreams.runtime.demo import DemoSpec, PreparedScenario, WebRTCOutputSpec
from flashvsr.runtime import (
    FIELD_CHUNK_SIZE,
    FIELD_FPS,
    FIELD_INPUT_HEIGHT,
    FIELD_INPUT_WIDTH,
    FIELD_TAIL_POLICY,
    FIELD_TOTAL_FRAMES,
    FlashVSRModelAdapter,
)

from .providers import PREPARED_VIDEO_METADATA_KEY, FlashVSRVideoInputProvider
from .spec import (
    FlashVSRVideoScenario,
    PreparedFlashVSRVideo,
    prepare_video_source,
    resolve_video_scenario,
)


class FlashVSRDemoAdapter(FlashVSRModelAdapter):
    """Prepare decoded videos for native runtime API demo execution."""

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("mp4", "null", "webrtc")

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if spec.input_mode != "replay":
            raise ValueError(
                "FlashVSR demos require input_mode='replay'; WebRTC replays or "
                "loops that decoded source through the realtime transport."
            )
        if spec.output.mode not in self.supported_output_modes():
            raise ValueError(f"Unsupported FlashVSR output mode: {spec.output.mode!r}.")
        if spec.config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        self.validate_config(spec.config)

        if isinstance(spec.scenario, PreparedFlashVSRVideo):
            prepared = spec.scenario
            video_scenario = prepared.scenario
        else:
            video_scenario = resolve_video_scenario(spec.scenario)
            prepared = prepare_video_source(
                video_scenario,
                scale=self._scale_for_spec(spec),
            )
        if video_scenario.loop_input and not isinstance(spec.output, WebRTCOutputSpec):
            raise ValueError(
                "FlashVSR loop_input is supported only with WebRTC output."
            )

        global_conditioning: dict[str, Any] = {
            FIELD_INPUT_HEIGHT: prepared.input_height,
            FIELD_INPUT_WIDTH: prepared.input_width,
            FIELD_FPS: prepared.fps,
            FIELD_CHUNK_SIZE: video_scenario.chunk_size,
            FIELD_TAIL_POLICY: video_scenario.tail_policy,
        }
        if not video_scenario.loop_input:
            global_conditioning[FIELD_TOTAL_FRAMES] = prepared.total_frames
        return PreparedScenario(
            initial_inputs=InferenceInput(global_conditioning=global_conditioning),
            source_schema=UserInputSchema(description="decoded FlashVSR video source"),
            canonicalizer=InputCanonicalizer(),
            mapping=self.default_input_mapping(),
            metadata={
                PREPARED_VIDEO_METADATA_KEY: prepared,
                "model_id": self.model_id,
                "preset_id": self.preset_id(spec.config),
                "input_path": str(prepared.resolved_path),
                "target_height": prepared.target_height,
                "target_width": prepared.target_width,
            },
        )

    def prepare_uploaded_video(
        self,
        spec: DemoSpec,
        *,
        upload_path: Path,
        original_name: str,
    ) -> PreparedFlashVSRVideo:
        """Decode one uploaded WebRTC video at the server playback rate.

        Args:
            spec: Base WebRTC demo specification.
            upload_path: Server-generated temporary upload path.
            original_name: Sanitized browser filename used only for metadata.

        Returns:
            Decoded CPU video ready for a model input provider.

        Raises:
            ValueError: The specification is not WebRTC output.
        """
        if not isinstance(spec.output, WebRTCOutputSpec):
            raise ValueError("FlashVSR uploads require WebRTC output.")
        scenario = resolve_video_scenario(spec.scenario)
        uploaded_scenario = replace(
            scenario,
            input_path=upload_path,
            fps=float(spec.output.fps),
        )
        prepared = prepare_video_source(
            uploaded_scenario,
            scale=self._scale_for_spec(spec),
        )
        display_scenario = replace(
            uploaded_scenario,
            input_path=original_name,
        )
        return replace(
            prepared,
            scenario=display_scenario,
            resolved_path=Path(original_name),
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> FlashVSRVideoInputProvider:
        del spec
        return FlashVSRVideoInputProvider(
            scenario=scenario,
            inference_input_schema=self.inference_input_schema,
        )

    def _scale_for_spec(self, spec: DemoSpec) -> int:
        config = spec.config
        if config is None:
            raise RuntimeError("DemoSpec.config was not initialized.")
        pipeline_config = self.pipeline_config(config)
        configured_scale = getattr(
            getattr(pipeline_config, "encoder", None),
            "scale",
            2,
        )
        return int(config.runtime_options.get("scale", configured_scale))


__all__ = ["FlashVSRDemoAdapter", "FlashVSRVideoScenario"]
