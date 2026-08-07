# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental shared demo API data shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, ModelAdapter
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.output_schema import InferenceOutputSchema
from flashdreams.runtime.sources import UserInputSource


@dataclass(frozen=True, kw_only=True, slots=True)
class NullOutputSpec:
    """Headless/null replay output."""

    mode: Literal["null"] = "null"
    store_results: bool = False

    def __post_init__(self) -> None:
        if self.mode != "null":
            raise ValueError("NullOutputSpec.mode must be 'null'.")


@dataclass(frozen=True, kw_only=True, slots=True)
class Mp4OutputSpec:
    """MP4 replay output."""

    path: str | Path
    fps: int | float
    mode: Literal["mp4"] = "mp4"
    output_layout: VideoTensorLayout = "bvtchw"
    move_to_cpu: bool = True

    def __post_init__(self) -> None:
        if self.mode != "mp4":
            raise ValueError("Mp4OutputSpec.mode must be 'mp4'.")
        if float(self.fps) <= 0:
            raise ValueError("Mp4OutputSpec.fps must be > 0.")
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCOutputSpec:
    """Shared WebRTC serving output."""

    mode: Literal["webrtc"] = "webrtc"
    host: str = "127.0.0.1"
    port: int = 8080
    fps: int = 30
    video_width: int = 1280
    video_height: int = 720
    warmup_chunks: int = 0
    warmup_timeout_s: float = 30.0
    client_liveness_timeout_s: float = 30.0
    web_dir: str | Path | None = None
    request_session_path: str = "/request_session"
    preload_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode != "webrtc":
            raise ValueError("WebRTCOutputSpec.mode must be 'webrtc'.")
        if not self.host.strip():
            raise ValueError("WebRTCOutputSpec.host must be non-empty.")
        if not (0 < int(self.port) < 65536):
            raise ValueError("WebRTCOutputSpec.port must be between 1 and 65535.")
        if self.fps <= 0:
            raise ValueError("WebRTCOutputSpec.fps must be > 0.")
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("WebRTCOutputSpec video dimensions must be > 0.")
        if self.warmup_chunks < 0:
            raise ValueError("WebRTCOutputSpec.warmup_chunks must be >= 0.")
        if self.warmup_timeout_s <= 0:
            raise ValueError("WebRTCOutputSpec.warmup_timeout_s must be > 0.")
        if self.client_liveness_timeout_s <= 0:
            raise ValueError("WebRTCOutputSpec.client_liveness_timeout_s must be > 0.")
        if not self.request_session_path.startswith("/"):
            raise ValueError(
                "WebRTCOutputSpec.request_session_path must start with '/'."
            )
        if self.web_dir is not None:
            object.__setattr__(self, "web_dir", Path(self.web_dir))


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalWindowOutputSpec:
    """Native windowed output presented on the machine running the model.

    Deliberately thin: window geometry and whether chrome is drawn are the
    only presentation-level knobs shared across models. Scene, camera, and
    control settings are model-specific and belong in ``DemoSpec.scenario``.
    """

    mode: Literal["local-window"] = "local-window"
    width: int = 1920
    height: int = 1080
    title: str = "flashdreams"
    show_hud: bool = True

    def __post_init__(self) -> None:
        if self.mode != "local-window":
            raise ValueError("LocalWindowOutputSpec.mode must be 'local-window'.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("LocalWindowOutputSpec dimensions must be > 0.")
        if not self.title.strip():
            raise ValueError("LocalWindowOutputSpec.title must be non-empty.")


OutputSpec: TypeAlias = (
    NullOutputSpec | Mp4OutputSpec | WebRTCOutputSpec | LocalWindowOutputSpec
)


@dataclass(frozen=True, kw_only=True, slots=True)
class DemoRoute:
    """One supported demo input/output mode pair."""

    input_mode: str
    """Application input mode."""

    output_mode: str
    """Selected output mode."""

    def __post_init__(self) -> None:
        if not self.input_mode.strip():
            raise ValueError("DemoRoute.input_mode must be non-empty.")
        if not self.output_mode.strip():
            raise ValueError("DemoRoute.output_mode must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class DemoSpec:
    """User-facing shared demo run description."""

    __hash__ = None

    model_id: str
    input_mode: str
    output: OutputSpec
    preset_id: str | None = None
    scenario: Any | None = None
    config: InferenceConfig | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("DemoSpec.model_id must be non-empty.")
        if not self.input_mode.strip():
            raise ValueError("DemoSpec.input_mode must be non-empty.")
        config = self.config
        if config is None:
            config = InferenceConfig(
                model_id=self.model_id,
                preset_id=self.preset_id,
            )
        else:
            if config.model_id != self.model_id:
                raise ValueError(
                    "DemoSpec.model_id must match InferenceConfig.model_id."
                )
            if self.preset_id is None:
                object.__setattr__(self, "preset_id", config.preset_id)
            elif config.preset_id is None:
                config = replace(config, preset_id=self.preset_id)
            elif config.preset_id != self.preset_id:
                raise ValueError(
                    "DemoSpec.preset_id must match InferenceConfig.preset_id."
                )
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


@dataclass(frozen=True, kw_only=True, slots=True)
class PreparedSession:
    """Runtime-ready inputs prepared for one model session."""

    __hash__ = None

    initial_inputs: InferenceInput
    inference_input_schema: InferenceInputSchema | None = None
    """Route-specific model input schema; ``None`` uses the adapter default."""

    user_inputs: UserInputSource = field(default_factory=UserInputs)
    """Replay batch or live source consumed one requested window at a time."""
    source_schema: UserInputSchema = field(default_factory=UserInputSchema)
    canonicalizer: InputCanonicalizer = field(default_factory=InputCanonicalizer)
    mapping: InputMapping | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class DemoAdapter(ModelAdapter, Protocol):
    """Model-owned adapter surface consumed by shared demo launchers."""

    @property
    def inference_output_schema(self) -> InferenceOutputSchema:
        """Return the semantic result type produced by shared runtime sessions."""
        ...

    def supported_routes(self) -> tuple[DemoRoute, ...]:
        """Return supported demo input/output mode pairs."""
        ...

    def prepare_session(self, spec: DemoSpec) -> PreparedSession:
        """Materialize one route-independent model session from ``spec``."""
        ...

    def create_demo_runtime(self, spec: DemoSpec) -> InferenceRuntime:
        """Create the runtime implementation selected by this demo route."""
        ...

    def list_sessions(self, spec: DemoSpec) -> tuple[DemoSpec, ...]:
        """Return selectable session specs for this demo route."""
        ...

    def create_webrtc_runtime(self, spec: DemoSpec) -> Any:
        """Create the model-owned runtime consumed by the shared WebRTC manager."""
        ...


__all__ = [
    "DemoAdapter",
    "DemoRoute",
    "DemoSpec",
    "LocalWindowOutputSpec",
    "Mp4OutputSpec",
    "NullOutputSpec",
    "OutputSpec",
    "PreparedSession",
    "WebRTCOutputSpec",
]
