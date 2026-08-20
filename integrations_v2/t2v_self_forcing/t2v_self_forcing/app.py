# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-Forcing text-to-video application, generating a clip from a prompt."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from self_forcing.config import PIPELINE_WAN21_T2V_1PT3B

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

_VIDEO_WIDTH = 832
"""Frame width the model was trained at."""

_VIDEO_HEIGHT = 480
"""Frame height it was trained at."""

_FRAMES_PER_SECOND = 16
"""Rate the frames it generates are meant to play at."""

_OUTPUT_LAYOUT = VideoTensorLayout.tchw
"""Layout the pipeline emits, one sequence of frames with no batch dimension."""


def default_session_desc() -> SessionDesc:
    """Return the session this model generates without being asked to resolve.

    A caller has to describe a session before one exists to describe it, so an
    application that only generates at certain sizes says so here. Passing this
    to :meth:`SelfForcingT2VApplication.create_session` is the case that needs
    no resolving.
    """
    return SessionDesc(
        output_layout=_OUTPUT_LAYOUT,
        frames_per_second_for_ui=60,
        frames_per_second_for_step=_FRAMES_PER_SECOND,
        video_width=_VIDEO_WIDTH,
        video_height=_VIDEO_HEIGHT,
    )


@dataclass(frozen=True, slots=True)
class SelfForcingT2VConfig:
    """Resolved settings shared by every session of one application."""

    prompt: str
    """Text every session generates from."""

    device: str
    """Device the pipeline is built on."""


## Session


class SelfForcingT2VSession(ISession):
    """One rollout: a prompt in, a chunk of frames per step out.

    A step is one autoregressive block. The model streams, so a run is as long
    as whatever drives it asks for, and each step continues the one before it
    rather than starting again. The first block decodes fewer frames than the
    rest, because the first latent frame of a causal decode covers one frame
    rather than a chunk of them.

    The pipeline belongs to the application and is shared with every other
    session. What belongs to a run is the cache this initializes, which holds
    the encoded prompt and the attention state the rollout builds up.
    """

    def __init__(self, pipeline: Any, prompt: str, session_desc: SessionDesc) -> None:
        """
        Args:
            pipeline: Loaded pipeline, owned by the application.
            prompt: Text to generate from.
            session_desc: Session the runtime asked for.

        Raises:
            ValueError: The description asks for output this cannot generate.
        """
        _validate(session_desc, pipeline)
        self._pipeline = pipeline
        self._prompt = prompt
        self._session_desc = session_desc
        self._cache: Any = None

    def init(self) -> None:
        """Encode the prompt and prepare the rollout's cache.

        This is where the text encoder runs, so it is slower than a step and
        happens once.
        """
        self._cache = self._new_cache()

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def step(self, step_index: int, events: UserInputEvents) -> StepResult:
        """Generate the next block of frames.

        Args:
            step_index: Zero-based index of this step, which is also the
                autoregressive index the rollout is up to. The pipeline rejects
                a step out of order.
            events: Ignored. This model takes its prompt at the start of a run
                and nothing after it.

        Returns:
            Result carrying ``[T, 3, H, W]`` frames as ``[-1, 1]`` floats, and
            whatever the pipeline measured while generating them.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._cache is None:
            raise RuntimeError("SelfForcingT2VSession.init() must run before step().")
        frames = self._pipeline.generate(
            autoregressive_index=step_index, cache=self._cache
        )
        # Advancing the attention state is what makes the next step continue
        # this one, and it reports what the step cost when the pipeline is
        # configured to measure it.
        metrics = self._pipeline.finalize(
            autoregressive_index=step_index, cache=self._cache
        )
        return StepResult(
            step_index=step_index,
            output=frames.detach(),
            frame_count=int(frames.shape[0]),
            output_layout=self._session_desc.output_layout,
            metrics=dict(metrics or {}),
        )

    def reset(self) -> None:
        """Start the rollout again from the same prompt.

        The cache is replaced rather than cleared, so nothing of the abandoned
        run reaches the new one.
        """
        self._cache = self._new_cache()

    def close(self) -> None:
        """Release the rollout's cache, leaving the loaded model alone."""
        self._cache = None

    def _new_cache(self) -> Any:
        """Encode the prompt into a cache for one rollout."""
        ratio = self._pipeline.decoder.spatial_compression_ratio
        return self._pipeline.initialize_cache(
            text=[self._prompt],
            image=None,
            height=self._session_desc.video_height // ratio,
            width=self._session_desc.video_width // ratio,
        )


## Application


class SelfForcingT2VApplication(IApplication):
    """Self-Forcing distilled Wan 2.1 1.3B, generating video from text.

    The model is loaded once, on the first session, and every session after
    that shares it. Loading means reading a checkpoint of several gigabytes and
    compiling the network, so a caller wanting several clips of one prompt
    should keep the application rather than build a second.
    """

    def __init__(self, pipeline_config: Any = PIPELINE_WAN21_T2V_1PT3B) -> None:
        """
        Args:
            pipeline_config: Model to run. The default is the distilled
                four-step checkpoint; a test passes a stand-in.
        """
        self._pipeline_config = pipeline_config
        self._config: SelfForcingT2VConfig | None = None
        self._pipeline: Any = None

    @property
    def pipeline_config(self) -> Any:
        """Model this will load, including whatever the command line changed."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse what to generate and where to generate it.

        The model itself is not loaded here. A caller can ask an application
        what it wants before paying for a checkpoint.

        Args:
            commandline_args: Application-specific arguments.

        Raises:
            ValueError: No prompt was given, or it is empty.
        """
        parser = argparse.ArgumentParser(
            prog="t2v-self-forcing",
            description="Generate video from text with Self-Forcing Wan 2.1 1.3B.",
        )
        parser.add_argument("--prompt", default="")
        parser.add_argument("--device", default="cuda")
        parser.add_argument(
            "--compile", action=argparse.BooleanOptionalAction, default=None
        )
        args = parser.parse_args(list(commandline_args))

        if not args.prompt.strip():
            raise ValueError("--prompt is required, and cannot be empty.")
        if args.compile is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"transformer": {"compile_network": args.compile}},
            )
        self._config = SelfForcingT2VConfig(prompt=args.prompt, device=args.device)

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized session, loading the model if needed.

        Args:
            session_desc: Session to generate. See :func:`default_session_desc`
                for what this model generates without being asked to resolve.

        Returns:
            Session sharing this application's loaded model.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
            ValueError: The description asks for output this cannot generate.
        """
        if self._config is None:
            raise RuntimeError(
                "SelfForcingT2VApplication.init() must run before create_session()."
            )
        if self._pipeline is None:
            self._pipeline = (
                self._pipeline_config.setup().to(self._config.device).eval()
            )
        return SelfForcingT2VSession(
            self._pipeline, self._config.prompt, session_desc
        )

    def close(self) -> None:
        """Release the model, and whatever memory it was holding."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if close is not None:
            close()


def create_app() -> IApplication:
    """Return a new Self-Forcing text-to-video application."""
    return SelfForcingT2VApplication()


def _validate(session_desc: SessionDesc, pipeline: Any) -> None:
    """Reject a session this model cannot generate.

    Rejecting rather than resolving: a caller that asked for one video and
    silently received another has no way to notice.

    Raises:
        ValueError: The layout or the frame size is one the model cannot emit.
    """
    if session_desc.output_layout is not _OUTPUT_LAYOUT:
        raise ValueError(
            f"Self-Forcing only produces {_OUTPUT_LAYOUT.value} output, got "
            f"{session_desc.output_layout.value}."
        )
    # A frame is decoded from a latent grid, so its size has to be a whole
    # number of latents across.
    ratio = pipeline.decoder.spatial_compression_ratio
    if session_desc.video_width % ratio or session_desc.video_height % ratio:
        raise ValueError(
            f"Frame dimensions must be multiples of {ratio}, got "
            f"{session_desc.video_width}x{session_desc.video_height}."
        )
