# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-neutral text-to-video application built on the v2 loop API."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop, IUILoop, invoke_async
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.slangpy_ui_loop import SlangPyUILoop
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_UI_FPS = 60


@dataclass(frozen=True, kw_only=True, slots=True)
class TextInputSpec:
    """One integration-owned text field rendered by the shared SlangPy UI."""

    name: str
    label: str
    default: str = ""
    required: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.label.strip():
            raise ValueError("Text input names and labels must be non-empty.")


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VApplicationDefaults:
    """Model-owned defaults consumed by the reusable T2V application."""

    pipeline_config: Any
    total_blocks: int
    pixel_height: int
    pixel_width: int
    prompt: str = ""
    device: str = "cuda"
    fps: int = 16
    output_layout: VideoTensorLayout = VideoTensorLayout.tchw
    text_inputs: tuple[TextInputSpec, ...] = ()

@dataclass(frozen=True, kw_only=True, slots=True)
class T2VIntegrationHooks:
    """Optional model-wiring hooks consumed by the shared T2V application.

    Integrations use these callbacks for configuration differences without
    implementing an application, session, model loop, or UI loop of their own.
    """

    configure_argument_parser: Callable[[argparse.ArgumentParser], None] | None = None
    """Callback that adds integration-owned command-line arguments."""

    apply_parsed_arguments: Callable[[argparse.Namespace], None] | None = None
    """Callback that consumes validated integration-owned arguments."""

    text_values_from_arguments: (
        Callable[[argparse.Namespace], Mapping[str, str]] | None
    ) = None
    """Callback that maps parsed arguments to retained UI text fields."""

    validate_text_values: Callable[[Mapping[str, str]], None] | None = None
    """Callback that validates integration-owned text values."""

    validate_total_blocks: Callable[[int], None] | None = None
    """Callback that validates model-specific rollout length constraints."""

    apply_compile_override: Callable[[Any, bool], Any] | None = None
    """Callback that applies compilation to an integration's config shape."""

    apply_seed_override: Callable[[Any, int], Any] | None = None
    """Callback that applies a seed to an integration's config shape."""

    initialize_cache: Callable[["T2VModelState"], Any] | None = None
    """Callback that builds integration-specific per-rollout model state."""


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VSessionConfig:
    """Resolved application arguments used to create isolated sessions."""

    prompt: str
    device: str
    total_blocks: int
    text_values: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class T2VModelState:
    """Rollout state owned exclusively by the model loop."""

    pipeline_factory: Callable[[], Any]
    session_desc: SessionDesc
    prompt: str
    total_blocks: int
    text_values: dict[str, str]
    cache_factory: Callable[["T2VModelState"], Any]
    pipeline: Any = None
    cache: Any = None
    blocks_generated: int = 0
    ui_loop: IUILoop[Any] | None = None

    def request_generation(
        self,
        prompt: str,
        total_blocks: int,
        text_values: Mapping[str, str],
    ) -> None:
        """Apply a UI request on the model thread before its next step."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty.")
        if total_blocks <= 0:
            raise ValueError("total_blocks must be > 0.")
        self.prompt = prompt
        self.total_blocks = total_blocks
        self.text_values = dict(text_values)
        self.cache = None
        self.blocks_generated = 0

    def notify_ui(self, status: str) -> None:
        """Send immutable status text to UI-owned state."""
        if self.ui_loop is None:
            return
        invoke_async(
            self.ui_loop,
            lambda state, status=status: state.set_status(status),
        )


class T2VModelLoop(IModelLoop[T2VModelState]):
    """Initialize and advance one autoregressive rollout on the model thread."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate and finalize one model block."""
        del events
        state = self.state
        if state.pipeline is None:
            state.notify_ui("Loading the model pipeline...")
            state.pipeline = state.pipeline_factory()
        if state.cache is None:
            state.notify_ui("Encoding inputs and initializing the rollout...")
            state.cache = state.cache_factory(state)
        autoregressive_index = state.blocks_generated
        state.notify_ui(
            f"Generating block {autoregressive_index + 1}/{state.total_blocks}..."
        )
        frames = state.pipeline.generate(
            autoregressive_index=autoregressive_index,
            cache=state.cache,
        )
        metrics = state.pipeline.finalize(
            autoregressive_index=autoregressive_index,
            cache=state.cache,
        )
        state.blocks_generated += 1
        if state.blocks_generated >= state.total_blocks:
            state.notify_ui("Generation complete.")
        else:
            state.notify_ui(
                f"Generated block {state.blocks_generated}/{state.total_blocks}."
            )
        return [
            StepResult(
                step_index=step_index,
                output=frames.detach(),
                frame_count=_frame_count(frames, state.session_desc.output_layout),
                output_layout=state.session_desc.output_layout,
                metrics=dict(metrics or {}),
            )
        ]

    def is_finished(self) -> bool:
        """Finish after the requested number of model blocks."""
        return self.state.blocks_generated >= self.state.total_blocks

    def reset(self) -> None:
        """Rebuild cache state for the current prompt on the next step."""
        self.state.cache = None
        self.state.blocks_generated = 0
        self.state.notify_ui("Rollout reset.")

    def close(self) -> None:
        """Release session-owned model state."""
        self.state.cache = None


@dataclass(slots=True)
class T2VUIState:
    """Retained widget and form state owned exclusively by the UI loop."""

    model_loop: IModelLoop[T2VModelState]
    prompt: str
    total_blocks: int
    text_specs: tuple[TextInputSpec, ...]
    text_values: dict[str, str]
    status: str = "Ready."
    prompt_widget: Any | None = field(default=None, init=False, repr=False)
    block_widget: Any | None = field(default=None, init=False, repr=False)
    text_widgets: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    status_widget: Any | None = field(default=None, init=False, repr=False)
    generate_button: Any | None = field(default=None, init=False, repr=False)

    def set_status(self, status: str) -> None:
        """Apply model-loop status on the UI thread."""
        self.status = status
        if self.status_widget is not None:
            self.status_widget.text = status


class T2VSlangPyUILoop(SlangPyUILoop[T2VUIState]):
    """Composite model output beneath retained SlangPy generation controls."""

    def step_ui(
        self, ui: Any, step_index: int, events: UserInputEvents
    ) -> Tensor | None:
        """Create the controls once and present the latest model frame."""
        del step_index, events
        if self.state.prompt_widget is None:
            height = 250 + 32 * len(self.state.text_specs)
            window = ui.Window(
                ui.screen,
                "Text-to-video",
                position=(16, 16),
                size=(560, height),
            )
            self.state.prompt_widget = ui.InputText(
                window,
                "Prompt",
                self.state.prompt,
                self._set_prompt,
                multi_line=True,
            )
            for spec in self.state.text_specs:
                self.state.text_widgets[spec.name] = ui.InputText(
                    window,
                    spec.label,
                    self.state.text_values.get(spec.name, spec.default),
                    lambda value, name=spec.name: self._set_text_value(name, value),
                )
            self.state.block_widget = ui.InputInt(
                window,
                "Blocks",
                self.state.total_blocks,
                self._set_total_blocks,
                step=1,
                step_fast=10,
            )
            self.state.generate_button = ui.Button(
                window,
                "Generate",
                self._generate,
            )
            self.state.status_widget = ui.Text(window, self.state.status)
        return self.presented_model_frame()

    def _set_prompt(self, prompt: str) -> None:
        self.state.prompt = prompt

    def _set_total_blocks(self, total_blocks: int) -> None:
        self.state.total_blocks = int(total_blocks)

    def _set_text_value(self, name: str, value: str) -> None:
        self.state.text_values[name] = value

    def _generate(self) -> None:
        prompt = self.state.prompt.strip()
        if not prompt:
            self.state.set_status("Enter a prompt before generating.")
            return
        if self.state.total_blocks <= 0:
            self.state.set_status("Blocks must be greater than zero.")
            return
        missing = [
            spec.label
            for spec in self.state.text_specs
            if spec.required and not self.state.text_values.get(spec.name, "").strip()
        ]
        if missing:
            self.state.set_status(f"Required: {', '.join(missing)}.")
            return
        prompt_to_send = prompt
        blocks_to_send = self.state.total_blocks
        values_to_send = dict(self.state.text_values)
        self.state.set_status("Generation request queued.")
        invoke_async(
            self.state.model_loop,
            lambda state: state.request_generation(
                prompt_to_send,
                blocks_to_send,
                values_to_send,
            ),
        )

    def reset(self) -> None:
        """Reset renderer transients and show the reset status."""
        self.state.set_status("Rollout reset.")
        super().reset()


class T2VSession(ISession):
    """One cache-isolated rollout with independent UI and model loops."""

    def __init__(
        self,
        *,
        pipeline_factory: Callable[[], Any],
        config: T2VSessionConfig,
        session_desc: SessionDesc,
        cache_factory: Callable[[T2VModelState], Any],
        text_specs: tuple[TextInputSpec, ...] = (),
        ui_renderer: Any | None = None,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._config = config
        self._session_desc = session_desc
        self._cache_factory = cache_factory
        self._text_specs = text_specs
        self._ui_renderer = ui_renderer

    @property
    def session_desc(self) -> SessionDesc:
        """Return the accepted output geometry and loop rates."""
        return self._session_desc

    def init(self) -> None:
        """Register the model loop, then connect the UI through messages."""
        model_state = T2VModelState(
            pipeline_factory=self._pipeline_factory,
            session_desc=self._session_desc,
            prompt=self._config.prompt,
            total_blocks=self._config.total_blocks,
            text_values=dict(self._config.text_values),
            cache_factory=self._cache_factory,
        )
        model_loop = self.register_model_loop(T2VModelLoop, state=model_state)
        ui_kwargs: dict[str, Any]
        if self._ui_renderer is None:
            ui_kwargs = {
                "width": self._session_desc.video_width,
                "height": self._session_desc.video_height,
            }
        else:
            ui_kwargs = {"renderer": self._ui_renderer}
        ui_loop = self.register_ui_loop(
            T2VSlangPyUILoop,
            state=T2VUIState(
                model_loop=model_loop,
                prompt=self._config.prompt,
                total_blocks=self._config.total_blocks,
                text_specs=self._text_specs,
                text_values=dict(self._config.text_values),
            ),
            **ui_kwargs,
        )
        model_state.ui_loop = ui_loop


class T2VApplication(IApplication):
    """Reusable SlangPy T2V application configured by one model integration."""

    def __init__(
        self,
        *,
        defaults: T2VApplicationDefaults,
        pipeline_config: Any | None = None,
        hooks: T2VIntegrationHooks | None = None,
        ui_renderer_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        self.defaults = defaults
        self._pipeline_config = pipeline_config or defaults.pipeline_config
        self._hooks = hooks or T2VIntegrationHooks()
        self._ui_renderer_factory = ui_renderer_factory
        self._config: T2VSessionConfig | None = None
        self._pipeline: Any = None

    @property
    def pipeline_config(self) -> Any:
        """Return the model config after command-line overrides."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse model-neutral generation arguments."""
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 SLUG --",
            description="Generate video through a SlangPy control panel.",
        )
        parser.add_argument(
            "--prompt",
            default=self.defaults.prompt,
            help="Initial prompt shown in the UI.",
        )
        parser.add_argument(
            "--device", default=self.defaults.device, help="Model device."
        )
        parser.add_argument(
            "--total-blocks", type=int, default=self.defaults.total_blocks
        )
        parser.add_argument(
            "--compile",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument("--seed", type=int, default=None)
        self._configure_argument_parser(parser)
        args = parser.parse_args(list(commandline_args))
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("--prompt is required, and cannot be empty.")
        self._validate_total_blocks(args.total_blocks)
        text_values = self._text_values_from_arguments(args)
        self._validate_text_values(text_values)
        if args.compile is not None:
            self._pipeline_config = self._apply_compile_override(
                self._pipeline_config, args.compile
            )
        if args.seed is not None:
            self._pipeline_config = self._apply_seed_override(
                self._pipeline_config, args.seed
            )
        self._apply_parsed_arguments(args)
        self._config = T2VSessionConfig(
            prompt=prompt,
            device=args.device,
            total_blocks=args.total_blocks,
            text_values=text_values,
        )

    def session_desc(self) -> SessionDesc:
        """Describe the model's trained output size and rate."""
        return SessionDesc(
            output_layout=self.defaults.output_layout,
            frames_per_second_for_ui=_UI_FPS,
            frames_per_second_for_step=self.defaults.fps,
            video_width=self.defaults.pixel_width,
            video_height=self.defaults.pixel_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create a rollout whose model loop lazily loads the shared pipeline."""
        config = self._config
        if config is None:
            raise RuntimeError("init() must run before create_session().")
        self._validate_layout(session_desc)
        renderer = (
            None
            if self._ui_renderer_factory is None
            else self._ui_renderer_factory(
                session_desc.video_width, session_desc.video_height
            )
        )
        return T2VSession(
            pipeline_factory=lambda: self._load_pipeline(
                device=config.device,
                session_desc=session_desc,
            ),
            config=config,
            session_desc=session_desc,
            cache_factory=self._initialize_cache,
            text_specs=self.defaults.text_inputs,
            ui_renderer=renderer,
        )

    def close(self) -> None:
        """Release the application-owned model pipeline."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if callable(close):
            close()

    def _initialize_cache(self, state: T2VModelState) -> Any:
        if self._hooks.initialize_cache is not None:
            return self._hooks.initialize_cache(state)
        decoder = getattr(state.pipeline, "decoder", None)
        if decoder is None or not hasattr(decoder, "spatial_compression_ratio"):
            raise TypeError("T2V requires a video decoder compression ratio.")
        ratio = int(decoder.spatial_compression_ratio)
        height = state.session_desc.video_height
        width = state.session_desc.video_width
        if height % ratio or width % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got {width}x{height}."
            )
        return state.pipeline.initialize_cache(
            text=[state.prompt],
            image=None,
            height=height // ratio,
            width=width // ratio,
        )

    def _load_pipeline(self, *, device: str, session_desc: SessionDesc) -> Any:
        """Build and validate the shared pipeline from the model thread."""
        if self._pipeline is None:
            self._pipeline = self._pipeline_config.setup().to(device).eval()
        self._validate_frame_size(session_desc, self._pipeline)
        return self._pipeline

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add integration-specific arguments."""
        if self._hooks.configure_argument_parser is not None:
            self._hooks.configure_argument_parser(parser)

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Retain validated integration-specific arguments."""
        if self._hooks.apply_parsed_arguments is not None:
            self._hooks.apply_parsed_arguments(args)

    def _text_values_from_arguments(self, args: argparse.Namespace) -> dict[str, str]:
        """Return initial values for integration-owned SlangPy text fields."""
        if self._hooks.text_values_from_arguments is not None:
            return dict(self._hooks.text_values_from_arguments(args))
        del args
        return {spec.name: spec.default for spec in self.defaults.text_inputs}

    def _validate_text_values(self, values: Mapping[str, str]) -> None:
        missing = [
            spec.label
            for spec in self.defaults.text_inputs
            if spec.required and not values.get(spec.name, "").strip()
        ]
        if missing:
            raise ValueError(f"Required inputs are empty: {', '.join(missing)}.")
        if self._hooks.validate_text_values is not None:
            self._hooks.validate_text_values(values)

    def _validate_total_blocks(self, total_blocks: int) -> None:
        if total_blocks <= 0:
            raise ValueError("--total-blocks must be > 0.")
        if self._hooks.validate_total_blocks is not None:
            self._hooks.validate_total_blocks(total_blocks)

    def _apply_compile_override(self, pipeline_config: Any, enabled: bool) -> Any:
        if self._hooks.apply_compile_override is not None:
            return self._hooks.apply_compile_override(pipeline_config, enabled)
        return derive_config(
            pipeline_config,
            diffusion_model={"transformer": {"compile_network": enabled}},
        )

    def _apply_seed_override(self, pipeline_config: Any, seed: int) -> Any:
        if self._hooks.apply_seed_override is not None:
            return self._hooks.apply_seed_override(pipeline_config, seed)
        return derive_config(pipeline_config, diffusion_model={"seed": seed})

    def _validate_layout(self, session_desc: SessionDesc) -> None:
        if session_desc.output_layout is not self.defaults.output_layout:
            raise ValueError(
                f"This application produces {self.defaults.output_layout.value}, got "
                f"{session_desc.output_layout.value}."
            )

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        decoder = getattr(pipeline, "decoder", None)
        ratio = getattr(decoder, "spatial_compression_ratio", None)
        if ratio is None:
            raise TypeError("T2V requires a video decoder compression ratio.")
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got "
                f"{session_desc.video_width}x{session_desc.video_height}."
            )


def _frame_count(frames: Tensor, layout: VideoTensorLayout) -> int:
    if layout is VideoTensorLayout.tchw:
        return int(frames.shape[0])
    if layout is VideoTensorLayout.btchw:
        return int(frames.shape[1])
    if layout is VideoTensorLayout.bcthw:
        return int(frames.shape[2])
    if layout is VideoTensorLayout.bvtchw:
        return int(frames.shape[2])
    raise ValueError(f"Unsupported output layout: {layout}.")


__all__ = [
    "T2VApplication",
    "T2VApplicationDefaults",
    "T2VIntegrationHooks",
    "T2VModelLoop",
    "T2VModelState",
    "T2VSession",
    "T2VSessionConfig",
    "T2VSlangPyUILoop",
    "T2VUIState",
    "TextInputSpec",
]
