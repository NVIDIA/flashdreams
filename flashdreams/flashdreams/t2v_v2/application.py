# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video application every t2v integration configures rather than writes."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.t2v_v2.defaults import T2VApplicationDefaults
from flashdreams.t2v_v2.session import T2VSession

_FRAMES_PER_SECOND_FOR_UI = 60
"""Rate an interactive window would read input and present results at.

Nothing on the batch path reads it, since a file has no client to keep up with,
but a session has to declare one.
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class T2VSessionConfig:
    """What one command line resolved to, shared by every session it creates."""

    prompt: str
    """Text every session generates from."""

    device: str
    """Device the pipeline is built on."""

    pixel_width: int
    """Frame width to generate."""

    pixel_height: int
    """Frame height to generate."""

    fps: int
    """Rate the generated frames are meant to play at."""

    total_blocks: int
    """Steps a run generates when its caller did not say how many."""


class T2VApplication(IApplication):
    """Streaming text-to-video, configured by one integration's defaults.

    Every text-to-video model takes a prompt and generates blocks of frames at
    a size and rate it was trained for, so the command line for one is the
    command line for all of them. An integration supplies
    :class:`T2VApplicationDefaults` and inherits the rest, and adds a flag of
    its own through :meth:`_configure_argument_parser` when it has one.

    The model is loaded once, on the first session, and every session after
    that shares it. Loading means reading a checkpoint of several gigabytes and
    possibly compiling the network, so a caller wanting several clips of one
    prompt should keep the application rather than build a second.
    """

    session_type: type[T2VSession] = T2VSession
    """Session this creates. A model needing its own overrides this."""

    def __init__(self, *, defaults: T2VApplicationDefaults) -> None:
        """
        Args:
            defaults: What this integration generates when nobody asks for
                anything in particular.
        """
        self.defaults = defaults
        self._pipeline_config = defaults.pipeline_config
        self._config: T2VSessionConfig | None = None
        self._pipeline: Any = None

    @property
    def pipeline_config(self) -> Any:
        """Model this will load, including whatever the command line changed."""
        return self._pipeline_config

    @property
    def total_blocks(self) -> int:
        """Steps a run generates when its caller did not say how many.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        return self._resolved("total_blocks").total_blocks

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse what to generate, at what size, and where.

        The model itself is not loaded here. A caller can ask an application
        what it wants before paying for a checkpoint.

        Args:
            commandline_args: Application-specific arguments.

        Raises:
            ValueError: No prompt was given, or a size, rate, or rollout length
                is not positive.
        """
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 SLUG --",
            description="Generate video from text.",
        )
        parser.add_argument(
            "--prompt", default="", help="Text to generate from. Required."
        )
        parser.add_argument(
            "--device",
            default=self.defaults.device,
            help="Device to load the model on. Default: %(default)s.",
        )
        parser.add_argument(
            "--pixel-width",
            type=int,
            default=self.defaults.pixel_width,
            help="Frame width. Default: %(default)s, what this model generates.",
        )
        parser.add_argument(
            "--pixel-height",
            type=int,
            default=self.defaults.pixel_height,
            help="Frame height. Default: %(default)s.",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=self.defaults.fps,
            help="Rate the frames are meant to play at. Default: %(default)s.",
        )
        parser.add_argument(
            "--total-blocks",
            type=int,
            default=self.defaults.total_blocks,
            help=(
                "Steps a run generates when it was not told how many. "
                "Default: %(default)s."
            ),
        )
        parser.add_argument(
            "--compile",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=(
                "Compile the network, costing minutes once and saving "
                "milliseconds a step. Default: whatever the model's config says."
            ),
        )
        self._configure_argument_parser(parser)
        args = parser.parse_args(list(commandline_args))

        if not args.prompt.strip():
            raise ValueError("--prompt is required, and cannot be empty.")
        if args.pixel_width <= 0 or args.pixel_height <= 0:
            raise ValueError(
                "--pixel-width and --pixel-height must be > 0, got "
                f"{args.pixel_width}x{args.pixel_height}."
            )
        if args.fps <= 0:
            raise ValueError(f"--fps must be > 0, got {args.fps}.")
        self._validate_total_blocks(args.total_blocks)
        self._apply_parsed_arguments(args)

        if args.compile is not None:
            self._pipeline_config = self._apply_compile_override(
                self._pipeline_config, args.compile
            )
        self._config = T2VSessionConfig(
            prompt=args.prompt,
            device=args.device,
            pixel_width=args.pixel_width,
            pixel_height=args.pixel_height,
            fps=args.fps,
            total_blocks=args.total_blocks,
        )

    def session_desc(self) -> SessionDesc:
        """Return the session this application would generate as asked.

        A caller has to describe a session before one exists to describe it,
        and only the application knows what its model generates, so this is
        where a runner gets a description it can pass straight back to
        :meth:`create_session`.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        config = self._resolved("session_desc()")
        return SessionDesc(
            output_layout=self.defaults.output_layout,
            frames_per_second_for_ui=_FRAMES_PER_SECOND_FOR_UI,
            frames_per_second_for_step=config.fps,
            video_width=config.pixel_width,
            video_height=config.pixel_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized session, loading the model if needed.

        Args:
            session_desc: Session to generate. See :meth:`session_desc` for the
                one this application would choose for itself.

        Returns:
            Session sharing this application's loaded model.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
            ValueError: The description asks for output this cannot generate.
        """
        config = self._resolved("create_session()")
        # Before loading rather than after: a checkpoint of several gigabytes is
        # a long wait for a layout this was never going to accept.
        self._validate_layout(session_desc)
        if self._pipeline is None:
            self._pipeline = self._pipeline_config.setup().to(config.device).eval()
        self._validate_frame_size(session_desc, self._pipeline)
        return self.session_type(self._pipeline, config.prompt, session_desc)

    def close(self) -> None:
        """Release the model, and whatever memory it was holding."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if close is not None:
            close()

    ## Integration hooks

    def _configure_argument_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments this integration takes beyond the shared ones."""

    def _apply_parsed_arguments(self, args: argparse.Namespace) -> None:
        """Keep whatever :meth:`_configure_argument_parser` added."""

    def _validate_total_blocks(self, total_blocks: int) -> None:
        """Reject a rollout length this model cannot generate.

        Raises:
            ValueError: ``total_blocks`` is not positive. A model generating its
                whole clip in one rollout overrides this to require exactly one.
        """
        if total_blocks <= 0:
            raise ValueError(f"--total-blocks must be > 0, got {total_blocks}.")

    def _apply_compile_override(self, pipeline_config: Any, enabled: bool) -> Any:
        """Return ``pipeline_config`` with network compilation turned on or off."""
        return derive_config(
            pipeline_config,
            diffusion_model={"transformer": {"compile_network": enabled}},
        )

    ## Internals

    def _resolved(self, called: str) -> T2VSessionConfig:
        """Return what the command line resolved to.

        Args:
            called: What the caller was doing, for the error.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}.init() must run before {called}."
            )
        return self._config

    def _validate_layout(self, session_desc: SessionDesc) -> None:
        """Reject a layout this model does not emit.

        Rejecting rather than resolving: a caller that asked for one video and
        silently received another has no way to notice.

        Raises:
            ValueError: The layout is not the one this model emits.
        """
        layout = self.defaults.output_layout
        if session_desc.output_layout is not layout:
            raise ValueError(
                f"This application only produces {layout.value} output, got "
                f"{session_desc.output_layout.value}."
            )

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        """Reject a frame size this model cannot decode.

        Raises:
            ValueError: The frame size is not a whole number of latents.
        """
        # A frame is decoded from a latent grid, so its size has to be a whole
        # number of latents across.
        ratio = pipeline.decoder.spatial_compression_ratio
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got "
                f"{session_desc.video_width}x{session_desc.video_height}."
            )
