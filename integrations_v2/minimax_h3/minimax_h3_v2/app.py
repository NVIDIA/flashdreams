# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin V2 applications for native MiniMax H3 synchronized generation."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from PIL import Image

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.infra.runner_io import (
    read_audio_f32,
    read_image_rgb,
    read_optional_audio_f32,
    read_video_rgb_with_fps,
)
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from minimax_h3.constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    FPS,
    align_num_frames,
)
from minimax_h3.inference import (
    MiniMaxH3InferenceConfig,
    MiniMaxH3InferenceEngine,
    MiniMaxH3InferenceRequest,
    MiniMaxH3InferenceResult,
    MiniMaxH3Workflow,
)
from minimax_h3.latent_checkpoint import (
    MiniMaxH3AssetIdentity,
    MiniMaxH3LatentCheckpointStore,
)
from minimax_h3.reference_conditioning import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3Reference,
    MiniMaxH3VideoReference,
)

_DEFAULT_PROMPT = "Animate this scene with coherent natural motion."
_DEFAULT_SIZE = 768
_DEFAULT_DURATION = 5.0
_DEFAULT_STEPS = 30
_DEFAULT_SEED = 42


class MiniMaxH3Engine(Protocol):
    """Inference surface shared by the real engine and CPU stand-ins."""

    def generate(self, request: MiniMaxH3InferenceRequest) -> MiniMaxH3InferenceResult:
        """Generate one synchronized finite result."""
        ...

    def close(self) -> None:
        """Release shared model metadata."""
        ...


MiniMaxH3EngineFactory = Callable[[MiniMaxH3InferenceConfig], MiniMaxH3Engine]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ReferenceSpec:
    """One validated ordered local reference before media decoding."""

    kind: str
    path: Path


@dataclass(frozen=True, kw_only=True, slots=True)
class _ApplicationSettings:
    """Validated application arguments independent of output geometry."""

    prompt: str
    duration: float
    steps: int
    seed: int
    restart: bool
    work_dir: Path | None
    job_id: str | None
    lora_path: Path | None
    lora_scale: float


@dataclass(frozen=True, kw_only=True, slots=True)
class _DecodedInputs:
    """Decoded workflow media and immutable checkpoint identities."""

    first_image: Image.Image | None = None
    last_image: Image.Image | None = None
    references: tuple[MiniMaxH3Reference, ...] = ()
    identities: tuple[MiniMaxH3AssetIdentity, ...] = ()


@dataclass(slots=True)
class MiniMaxH3ModelState:
    """One finite request and completion bit owned by the model loop."""

    engine: MiniMaxH3Engine
    request: MiniMaxH3InferenceRequest
    finished: bool = False


class MiniMaxH3ModelLoop(IModelLoop[MiniMaxH3ModelState]):
    """Generate one native H3 result, then report finite completion."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Run the staged engine once and return synchronized typed media."""
        del events
        if self.state.finished:
            raise RuntimeError("MiniMax H3 already generated this session")
        generated = self.state.engine.generate(self.state.request)
        self.state.finished = True
        return [
            StepResult(
                step_index=step_index,
                output=generated.video,
                frame_count=generated.video.shape[0],
                output_layout=VideoTensorLayout.tchw,
                metrics=dict(generated.metrics),
                audio=generated.audio,
            )
        ]

    def is_finished(self) -> bool:
        """Return whether the single finite rollout completed."""
        return self.state.finished

    def reset(self) -> None:
        """Allow the same immutable request to be generated again."""
        self.state.finished = False


class MiniMaxH3Session(ISession):
    """Register one finite native H3 model loop for the V2 runtime."""

    def __init__(
        self,
        engine: MiniMaxH3Engine,
        request: MiniMaxH3InferenceRequest,
        session_desc: SessionDesc,
    ) -> None:
        """Retain application-owned inference and request state."""
        self._engine = engine
        self._request = request
        self._session_desc = session_desc

    def init(self) -> None:
        """Register the finite H3 model loop; the default UI presents it."""
        self.register_model_loop(
            MiniMaxH3ModelLoop,
            state=MiniMaxH3ModelState(
                engine=self._engine,
                request=self._request,
            ),
        )

    @property
    def session_desc(self) -> SessionDesc:
        """Return the synchronized media description validated by the app."""
        return self._session_desc


class MiniMaxH3Application(IApplication):
    """Decode inputs at the boundary and drive one native H3 workflow."""

    def __init__(
        self,
        workflow: MiniMaxH3Workflow,
        *,
        engine_factory: MiniMaxH3EngineFactory = MiniMaxH3InferenceEngine,
    ) -> None:
        """Create a weight-free application for one fixed workflow."""
        self.workflow = workflow
        self._engine_factory = engine_factory
        self._settings: _ApplicationSettings | None = None
        self._inputs: _DecodedInputs | None = None
        self._engine: MiniMaxH3Engine | None = None
        self._closed = False

    def session_desc(self) -> SessionDesc:
        """Describe fixed H3 media without parsing arguments or loading weights."""
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
            frames_per_second_for_ui=FPS,
            frames_per_second_for_step=FPS,
            video_width=_DEFAULT_SIZE,
            video_height=_DEFAULT_SIZE,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_channels=AUDIO_CHANNELS,
            metadata={"workflow": self.workflow},
        )

    def init(self, commandline_args: Sequence[str]) -> None:
        """Validate arguments, decode local inputs, and load shared metadata."""
        if self._closed:
            raise RuntimeError("MiniMaxH3Application is closed")
        if self._engine is not None:
            raise RuntimeError("MiniMaxH3Application is already initialized")
        namespace = _parse_arguments(self.workflow, commandline_args)
        settings = _settings_from_namespace(namespace)
        inputs = _decode_inputs(self.workflow, namespace)
        config = MiniMaxH3InferenceConfig(
            device=namespace.device,
            attention_backend=namespace.attention,
            cache_dir=namespace.cache_dir,
            checkpoint_min_free_gb=namespace.checkpoint_min_free_gb,
        )
        engine = self._engine_factory(config)
        self._settings = settings
        self._inputs = inputs
        self._engine = engine

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create an uninitialized finite session for validated H3 geometry."""
        if self._closed:
            raise RuntimeError("MiniMaxH3Application is closed")
        if self._settings is None or self._inputs is None or self._engine is None:
            raise RuntimeError(
                "MiniMaxH3Application.init() must run before create_session()"
            )
        _validate_session_desc(session_desc)
        store = None
        if self._settings.work_dir is not None:
            store = MiniMaxH3LatentCheckpointStore(
                work_dir=self._settings.work_dir,
                job_id=cast(str, self._settings.job_id),
            )
        request = MiniMaxH3InferenceRequest(
            workflow=self.workflow,
            prompt=self._settings.prompt,
            width=session_desc.video_width,
            height=session_desc.video_height,
            duration=self._settings.duration,
            num_inference_steps=self._settings.steps,
            seed=self._settings.seed,
            first_image=self._inputs.first_image,
            last_image=self._inputs.last_image,
            references=self._inputs.references,
            checkpoint_store=store,
            checkpoint_inputs=self._inputs.identities,
            restart=self._settings.restart,
            lora_path=self._settings.lora_path,
            lora_scale=self._settings.lora_scale,
        )
        return MiniMaxH3Session(self._engine, request, session_desc)

    def close(self) -> None:
        """Release application-owned shared inference metadata once."""
        if self._closed:
            return
        engine = self._engine
        if engine is not None:
            engine.close()
        self._engine = None
        self._closed = True


def _parse_arguments(
    workflow: MiniMaxH3Workflow,
    commandline_args: Sequence[str],
) -> argparse.Namespace:
    """Parse workflow-specific arguments passed after the V2 separator."""
    parser = argparse.ArgumentParser(
        prog=f"minimax-h3-{workflow}",
        description=f"Run native MiniMax H3 {workflow.upper()} inference.",
        allow_abbrev=False,
    )
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=_DEFAULT_DURATION)
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--attention",
        choices=("flash", "cudnn", "efficient", "math"),
        default="flash",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--checkpoint-min-free-gb",
        type=float,
        default=150.0,
    )
    if workflow == "fl2va":
        parser.add_argument("--image-path", type=Path)
        parser.add_argument("--last-image-path", type=Path)
    elif workflow == "ref2va":
        parser.add_argument("--reference", action="append", default=[])
    return parser.parse_args(list(commandline_args))


def _settings_from_namespace(namespace: argparse.Namespace) -> _ApplicationSettings:
    """Validate static request options before constructing engine resources."""
    prompt = namespace.prompt.strip()
    if not prompt:
        raise ValueError("--prompt must be non-empty")
    align_num_frames(namespace.duration)
    if namespace.steps < 2:
        raise ValueError("--steps must be at least 2")
    paired_resume = namespace.work_dir is not None and namespace.job_id is not None
    if (namespace.work_dir is None) != (namespace.job_id is None):
        raise ValueError("--work-dir and --job-id must be passed together")
    if namespace.restart and not paired_resume:
        raise ValueError("--restart requires --work-dir and --job-id")
    if not math.isfinite(namespace.lora_scale) or not 0 <= namespace.lora_scale <= 4:
        raise ValueError("--lora-scale must be finite and between 0 and 4")
    lora_path = None
    if namespace.lora is not None:
        lora_path = _resolve_file(namespace.lora, option="--lora")
    elif namespace.lora_scale != 1.0:
        raise ValueError("--lora-scale requires --lora")
    return _ApplicationSettings(
        prompt=prompt,
        duration=namespace.duration,
        steps=namespace.steps,
        seed=namespace.seed,
        restart=namespace.restart,
        work_dir=(
            None if namespace.work_dir is None else namespace.work_dir.expanduser()
        ),
        job_id=namespace.job_id,
        lora_path=lora_path,
        lora_scale=namespace.lora_scale,
    )


def _decode_inputs(
    workflow: MiniMaxH3Workflow,
    namespace: argparse.Namespace,
) -> _DecodedInputs:
    """Decode workflow media before allocating any model weights."""
    if workflow == "t2va":
        return _DecodedInputs()
    if workflow == "fl2va":
        first_path = _optional_file(namespace.image_path, option="--image-path")
        last_path = _optional_file(
            namespace.last_image_path, option="--last-image-path"
        )
        if first_path is None and last_path is None:
            raise ValueError("fl2va requires --image-path, --last-image-path, or both")
        identities = tuple(
            MiniMaxH3AssetIdentity.from_file(path, source=f"{anchor}:{path}")
            for anchor, path in (("first", first_path), ("last", last_path))
            if path is not None
        )
        return _DecodedInputs(
            first_image=None if first_path is None else _read_pil_image(first_path),
            last_image=None if last_path is None else _read_pil_image(last_path),
            identities=identities,
        )

    specs = _parse_reference_specs(namespace.reference)
    references = tuple(_decode_reference(spec) for spec in specs)
    identities = tuple(
        MiniMaxH3AssetIdentity.from_file(spec.path, source=f"{spec.kind}:{spec.path}")
        for spec in specs
    )
    return _DecodedInputs(references=references, identities=identities)


def _parse_reference_specs(entries: Sequence[str]) -> tuple[_ReferenceSpec, ...]:
    """Parse ordered local reference specs and enforce released limits."""
    specs: list[_ReferenceSpec] = []
    for entry in entries:
        kind, separator, value = entry.partition(":")
        if not separator or kind not in {"image", "video", "audio"} or not value:
            raise ValueError(
                f"invalid reference {entry!r}; expected image:path, video:path, "
                "or audio:path"
            )
        specs.append(
            _ReferenceSpec(
                kind=kind,
                path=_resolve_file(Path(value), option="--reference"),
            )
        )
    if not specs:
        raise ValueError("ref2va requires at least one --reference")
    for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
        count = sum(spec.kind == kind for spec in specs)
        if count > limit:
            raise ValueError(f"MiniMax H3 accepts at most {limit} {kind} references")
    if len(specs) > 12:
        raise ValueError("MiniMax H3 accepts at most 12 references in total")
    if all(spec.kind == "audio" for spec in specs):
        raise ValueError(
            "an audio reference must be paired with an image or video reference"
        )
    return tuple(specs)


def _decode_reference(spec: _ReferenceSpec) -> MiniMaxH3Reference:
    """Decode one reference through model-neutral host media helpers."""
    if spec.kind == "image":
        return MiniMaxH3ImageReference(_read_pil_image(spec.path))
    if spec.kind == "video":
        frames, fps = read_video_rgb_with_fps(spec.path)
        audio = read_optional_audio_f32(
            spec.path,
            sample_rate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
        )
        return MiniMaxH3VideoReference(
            frames=frames,
            fps=fps,
            audio=audio,
            sample_rate=None if audio is None else AUDIO_SAMPLE_RATE,
        )
    if spec.kind == "audio":
        return MiniMaxH3AudioReference(
            audio=read_audio_f32(
                spec.path,
                sample_rate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
            ),
            sample_rate=AUDIO_SAMPLE_RATE,
        )
    raise AssertionError(f"validated reference kind was not decoded: {spec.kind}")


def _read_pil_image(path: Path) -> Image.Image:
    """Read and validate one uint8 RGB image through the shared media helper."""
    pixels = np.asarray(read_image_rgb(path))
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError(
            f"decoded image must have uint8 shape [height, width, 3]: {path}"
        )
    return Image.fromarray(pixels, mode="RGB")


def _optional_file(path: Path | None, *, option: str) -> Path | None:
    """Resolve an optional local regular file."""
    return None if path is None else _resolve_file(path, option=option)


def _resolve_file(path: Path, *, option: str) -> Path:
    """Resolve one local regular input before any model allocation."""
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{option} file not found: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{option} must name a regular file: {resolved}")
    return resolved


def _validate_session_desc(session_desc: SessionDesc) -> None:
    """Require the synchronized finite V2 contract implemented by H3."""
    if session_desc.output_layout is not VideoTensorLayout.tchw:
        raise ValueError("MiniMax H3 requires tchw output layout")
    if session_desc.backpressure_mode is not BackpressureMode.BLOCK:
        raise ValueError("MiniMax H3 file generation requires block backpressure")
    if session_desc.presentation_mode is not PresentationMode.ONLY_PRESENT_NEW:
        raise ValueError("MiniMax H3 requires only_present_new presentation")
    if session_desc.frames_per_second_for_step != FPS:
        raise ValueError(f"MiniMax H3 requires {FPS} fps")
    if (
        session_desc.audio_sample_rate != AUDIO_SAMPLE_RATE
        or session_desc.audio_channels != AUDIO_CHANNELS
    ):
        raise ValueError(f"MiniMax H3 requires {AUDIO_SAMPLE_RATE} Hz stereo audio")


def create_t2va_app() -> IApplication:
    """Return a new native MiniMax H3 prompt-only V2 application."""
    return MiniMaxH3Application("t2va")


def create_fl2va_app() -> IApplication:
    """Return a new native MiniMax H3 keyframe V2 application."""
    return MiniMaxH3Application("fl2va")


def create_ref2va_app() -> IApplication:
    """Return a new native MiniMax H3 ordered-reference V2 application."""
    return MiniMaxH3Application("ref2va")


def create_app() -> IApplication:
    """Return T2VA for module-name discovery; registered slugs are explicit."""
    return create_t2va_app()


__all__ = [
    "MiniMaxH3Application",
    "MiniMaxH3Engine",
    "MiniMaxH3EngineFactory",
    "MiniMaxH3ModelLoop",
    "MiniMaxH3ModelState",
    "MiniMaxH3Session",
    "create_app",
    "create_fl2va_app",
    "create_ref2va_app",
    "create_t2va_app",
]
