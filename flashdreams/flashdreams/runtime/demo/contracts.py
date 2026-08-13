# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by applications, IO handlers, and session drivers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from flashdreams.infra.postprocess import VideoTensorLayout
from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import InferenceInput, UserInputs, UserInputSchema
from flashdreams.runtime.interfaces import ModelAdapter
from flashdreams.runtime.mapping import InputMapping

from .host import WarmupSessionInputs
from .session_inputs import ModelInputProvider


@dataclass(frozen=True, kw_only=True, slots=True)
class NullOutputSpec:
    """Headless/null replay output."""

    mode: Literal["null"] = "null"
    store_results: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class Mp4OutputSpec:
    """MP4 replay output."""

    path: str | Path
    fps: int | float
    mode: Literal["mp4"] = "mp4"
    output_layout: VideoTensorLayout = "bvtchw"
    move_to_cpu: bool = True

    def __post_init__(self) -> None:
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
class NativeWindowOutputSpec:
    """Shared local native-window presentation output."""

    mode: Literal["local-window"] = "local-window"
    fps: int = 30
    video_width: int = 1280
    video_height: int = 720
    title: str = "FlashDreams"
    max_queued_chunks: int = 2
    close_timeout_s: float = 10.0
    batch_index: int = 0
    view_index: int = 0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("NativeWindowOutputSpec.fps must be > 0.")
        if self.video_width <= 0 or self.video_height <= 0:
            raise ValueError("NativeWindowOutputSpec dimensions must be > 0.")
        if not self.title.strip():
            raise ValueError("NativeWindowOutputSpec.title must be non-empty.")
        if self.max_queued_chunks <= 0:
            raise ValueError("NativeWindowOutputSpec.max_queued_chunks must be > 0.")
        if self.close_timeout_s <= 0:
            raise ValueError("NativeWindowOutputSpec.close_timeout_s must be > 0.")
        if self.batch_index < 0 or self.view_index < 0:
            raise ValueError("NativeWindowOutputSpec indices must be >= 0.")


class OutputSpec(Protocol):
    """IO-handler-owned output description."""

    @property
    def mode(self) -> str: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class WebRTCAppResources:
    """Application resources attached to the shared WebRTC server."""

    model_web_resource: Any | None = None
    configure_app: Callable[[Any], None] | None = None
    preload_name: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class DemoSpec:
    """One prepared application execution passed to shared session drivers."""

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
class PreparedScenario:
    """Runtime-ready scenario prepared by an application."""

    __hash__ = None

    initial_inputs: InferenceInput
    user_inputs: UserInputs = field(default_factory=UserInputs)
    source_schema: UserInputSchema = field(default_factory=UserInputSchema)
    canonicalizer: InputCanonicalizer = field(default_factory=InputCanonicalizer)
    mapping: InputMapping | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class DemoAdapter(ModelAdapter, Protocol):
    """Transport-neutral application adapter consumed by session drivers."""

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        """Validate and materialize scenario inputs before runtime creation."""
        ...

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> ModelInputProvider:
        """Create the model-facing provider for one prepared scenario."""
        ...


class ModelWarmupAdapter(Protocol):
    """Optional application hook for model-affine runtime warmup inputs."""

    def create_model_warmup_sessions(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> Sequence[WarmupSessionInputs]:
        """Return temporary synthetic or loopback sessions for model warmup."""
        ...


__all__ = [
    "DemoAdapter",
    "DemoSpec",
    "ModelWarmupAdapter",
    "Mp4OutputSpec",
    "NativeWindowOutputSpec",
    "NullOutputSpec",
    "OutputSpec",
    "PreparedScenario",
    "WebRTCAppResources",
    "WebRTCOutputSpec",
]
