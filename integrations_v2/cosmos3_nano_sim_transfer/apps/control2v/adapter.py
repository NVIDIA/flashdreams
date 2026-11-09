# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control video to video with Cosmos3-Nano-Sim-Transfer (``control2v-cosmos3-nano-sim-transfer``).

::

    flashdreams-run-v2 control2v-cosmos3-nano-sim-transfer --output-path out.mp4 --timeout unbound -- \\
        --control seg.mp4 --hint seg --prompt "A city street ..." --seed 0 --history long-rollout

The application reads the control clip, generates it one latent per model step
(one frame, then four) with :class:`~cosmos3_nano_sim_transfer.impl.stream.TransferStream`,
and finishes after the clip's last latent. As in cosmos-framework, a clip that
is not ``1 + 4n`` frames long is padded to whole latents for the encoder, the
prompt states the clip's own length, and the output is trimmed to it.

Every step's result carries the stage timings of
:attr:`TransferStream.last_timing
<cosmos3_nano_sim_transfer.impl.stream.TransferStream.last_timing>`, which the
runtime's ``--stats-path`` records; ``--timing-path`` writes them per latent
with their steady-state medians and the peak GPU memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.infra.runner_io import read_video_window_rgb, resolve_prompt_value
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from ...config import (
    CONTROL_HINTS,
    HF_REPO_ID,
    HISTORY_PRESETS,
    HistoryProfile,
    steady_from,
)
from ...impl.control import latent_frame_spans, padded_frame_indices, read_control_fps
from ...impl.decode import DECODERS
from ...impl.prompt import HEIGHT, WIDTH
from ...impl.stream import TransferStream

SLUG = "control2v-cosmos3-nano-sim-transfer"
"""Name the application is registered under (``flashdreams.applications_v2``)."""

DEFAULT_FPS = 30
"""Frame rate described before the control clip is read."""


@dataclass(frozen=True)
class Control2VArgs:
    """The application's own arguments (after ``--``)."""

    control: Path
    """Control video: segmentation, depth, edge or blur rendered as RGB."""
    prompt: str
    """Scene description."""
    hint: str = "seg"
    """Kind of control video."""
    emphasize_control_in_prompt: bool = False
    """Append the Framework control-adherence sentence to the prompt."""
    prompt_aspect_ratio: str | None = None
    """Optional bucket label injected into JSON prompts; does not resize controls."""
    seed: int = 0
    """Noise seed."""
    history: str = "long-rollout"
    """History preset (``export``, ``long-rollout``) or ``W/S``."""
    fps: float | None = None
    """Frame rate of the control clip; read from the file when ``None``."""
    max_frames: int | None = None
    """Use at most this many control frames."""
    prompt_frames: int | None = None
    """Clip length the prompt states; the control clip's when ``None``."""
    decoder: str = "wan2.2"
    """Latent decoder."""
    release: Path | None = None
    """Release snapshot folder; the pinned release is downloaded when ``None``."""
    vae: Path | None = None
    """Wan 2.2 VAE weights for the encoder and the ``wan2.2`` decoder."""
    taehv: Path | None = None
    """TAEHV-super weights for the ``taehv-super`` decoder."""
    device: str = "cuda"
    """Where the model runs."""
    timing_path: Path | None = None
    """JSON file for per-latent stage timings."""


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {number}")
    return number


def _seed(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not number > 0 or number == float("inf"):
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {value}")
    return number


def _history(value: str) -> str:
    if value not in HISTORY_PRESETS:
        try:
            HistoryProfile.parse(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from None
    return value


def build_parser() -> argparse.ArgumentParser:
    """Parser of the application's arguments (what ``-- --help`` prints)."""
    parser = argparse.ArgumentParser(
        prog=f"flashdreams-run-v2 {SLUG} --",
        description=(
            "Generate an 832x480 video from a control video and a caption with "
            f"{HF_REPO_ID}, one latent (four frames) per model step."
        ),
    )
    parser.add_argument(
        "--control", type=Path, required=True, help="control video file"
    )
    parser.add_argument(
        "--hint",
        choices=CONTROL_HINTS,
        default="seg",
        help="kind of control video; selects the resize of other sizes (default: seg)",
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="scene description")
    prompt.add_argument(
        "--prompt-file",
        type=Path,
        help="text file whose first non-empty line is the scene description",
    )
    parser.add_argument("--seed", type=_seed, default=0, help="noise seed (default: 0)")
    parser.add_argument(
        "--emphasize-control-in-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="append the Framework control-adherence sentence (default: disabled)",
    )
    parser.add_argument(
        "--prompt-aspect-ratio",
        help="nominal aspect ratio for JSON prompt metadata, e.g. 16,9; no resize effect",
    )
    parser.add_argument(
        "--history",
        type=_history,
        default="long-rollout",
        help=(
            "latents in view of each new latent: 'long-rollout' (30/3, the "
            "release's profile for long clips), 'export' (the checkpoint's own, "
            "51/1) or W/S (window/pinned latents) (default: long-rollout)"
        ),
    )
    parser.add_argument(
        "--fps",
        type=_positive_float,
        default=None,
        help="frame rate of the control video (default: read from the file)",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=None,
        help="use at most this many control frames (default: all)",
    )
    parser.add_argument(
        "--prompt-frames",
        type=_positive_int,
        default=None,
        help="clip length the prompt's duration sentence states (default: the clip's)",
    )
    parser.add_argument(
        "--decoder",
        choices=DECODERS,
        default="wan2.2",
        help="latent decoder: wan2.2 (reference) or taehv-super (fast) (default: wan2.2)",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=None,
        help="release snapshot folder (default: download the pinned release)",
    )
    parser.add_argument(
        "--vae", type=Path, default=None, help="Wan 2.2 VAE weights (Wan2.2_VAE.pth)"
    )
    parser.add_argument("--taehv", type=Path, default=None, help="TAEHV-super weights")
    parser.add_argument("--device", default="cuda", help="device (default: cuda)")
    parser.add_argument(
        "--timing-path",
        type=Path,
        default=None,
        help="write per-latent stage timings and their medians to this JSON file",
    )
    return parser


def parse_args(commandline_args: Sequence[str]) -> Control2VArgs:
    """Parse the application's arguments.

    Raises:
        SystemExit: An argument is missing or invalid (argparse's behavior).
        ValueError: The prompt file holds no text.
    """
    args = build_parser().parse_args(list(commandline_args))
    prompt = (
        resolve_prompt_value(args.prompt_file)
        if args.prompt_file is not None
        else resolve_prompt_value(args.prompt)
    )
    return Control2VArgs(
        control=args.control,
        prompt=prompt,
        hint=args.hint,
        emphasize_control_in_prompt=args.emphasize_control_in_prompt,
        prompt_aspect_ratio=args.prompt_aspect_ratio,
        seed=args.seed,
        history=args.history,
        fps=args.fps,
        max_frames=args.max_frames,
        prompt_frames=args.prompt_frames,
        decoder=args.decoder,
        release=args.release,
        vae=args.vae,
        taehv=args.taehv,
        device=args.device,
        timing_path=args.timing_path,
    )


def read_control_clip(path: Path, max_frames: int | None = None) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 frames of a control video, at most ``max_frames``.

    Raises:
        FileNotFoundError: ``path`` is not a file.
        ValueError: The video has no frames.
    """
    if not path.is_file():
        raise FileNotFoundError(f"control video does not exist: {path}")
    frames = read_video_window_rgb(path, frame_count=max_frames)
    if frames.shape[0] == 0:
        raise ValueError(f"control video has no frames: {path}")
    return frames


def load_stream(args: Control2VArgs, fps: float) -> TransferStream:
    """Load the model for ``args`` (the application's default stream loader)."""
    return TransferStream.load(
        args.release,
        hint=args.hint,
        emphasize_control_in_prompt=args.emphasize_control_in_prompt,
        prompt_aspect_ratio=args.prompt_aspect_ratio,
        history=args.history,
        fps=fps,
        decoder=args.decoder,
        vae=args.vae,
        taehv=args.taehv,
        device=args.device,
    )


StreamLoader = Callable[[Control2VArgs, float], TransferStream]
"""Builds the stream from the parsed arguments and the clip's frame rate."""


def timing_summary(timings: Sequence[dict[str, float]], start: int) -> dict[str, float]:
    """Median of every timing key over ``timings[start:]`` (all of them when shorter)."""
    window = list(timings[start:]) or list(timings)
    keys = sorted({key for record in window for key in record if key != "latent"})
    return {key: statistics.median(r[key] for r in window if key in r) for key in keys}


## Session


@dataclass
class Control2VState:
    """What the model loop owns: the stream and the clip it generates."""

    stream: TransferStream
    frames: np.ndarray
    """``[T, H, W, 3]`` uint8 control frames."""
    indices: list[int]
    """Frame indices of the clip padded to whole latents."""
    spans: list[slice]
    """Positions in ``indices`` of each latent's control frames."""
    timings: list[dict[str, float]] = field(default_factory=list)
    """Stage timings per generated latent."""
    generated: int = 0
    """Latents generated since the session (re)started."""


class Control2VModelLoop(IModelLoop[Control2VState]):
    """Generate one latent of the clip per step."""

    def step(self, step_index: int, events: UserInputEvents) -> list[StepResult]:
        """Generate the frames of latent ``step_index``.

        Raises:
            RuntimeError: Steps arrive out of order.
        """
        del events
        state = self.state
        if step_index != state.generated:
            raise RuntimeError(
                f"control2v step out of sequence: expected {state.generated}, got {step_index}"
            )
        span = state.spans[step_index]
        last = step_index == len(state.spans) - 1
        output = state.stream.step_tensor(state.frames[state.indices[span]], last=last)
        # Frames generated from padding are not part of the clip.
        output = output[: max(0, min(span.stop, len(state.frames)) - span.start)]
        timing = state.stream.last_timing
        state.timings.append({"latent": step_index, **timing})
        state.generated += 1
        return [
            StepResult(
                step_index=step_index,
                output=output,
                frame_count=int(output.shape[0]),
                output_layout=VideoTensorLayout.tchw,
                metrics=dict(timing),
            )
        ]

    def is_finished(self) -> bool:
        return self.state.generated >= len(self.state.spans)

    def reset(self) -> None:
        """Start the clip over with the same prompt and seed."""
        self.state.stream.reset()
        self.state.generated = 0
        self.state.timings.clear()


@dataclass(frozen=True)
class _Resolved:
    """What one application run shares with its sessions."""

    args: Control2VArgs
    stream: TransferStream
    frames: np.ndarray
    fps: float
    history: HistoryProfile


class Control2VSession(ISession):
    """One pass over the control clip."""

    def __init__(self, resolved: _Resolved, session_desc: SessionDesc) -> None:
        self._resolved = resolved
        self._session_desc = session_desc
        self._state: Control2VState | None = None

    @property
    def session_desc(self) -> SessionDesc:
        return self._session_desc

    def init(self) -> None:
        """Start a stream session and register the model loop."""
        resolved = self._resolved
        frames = resolved.frames
        resolved.stream.start(
            resolved.args.prompt,
            resolved.args.seed,
            history=resolved.history,
            num_frames=resolved.args.prompt_frames or int(frames.shape[0]),
        )
        if resolved.stream.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved.stream.device)
        indices = padded_frame_indices(int(frames.shape[0]))
        self._state = Control2VState(
            stream=resolved.stream,
            frames=frames,
            indices=indices,
            spans=latent_frame_spans(len(indices)),
        )
        self.register_model_loop(Control2VModelLoop, state=self._state)

    def close(self) -> None:
        """Write the timing file, if one was asked for."""
        state, path = self._state, self._resolved.args.timing_path
        self._state = None
        if state is None or path is None or not state.timings:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.timing_report(state), indent=1))

    def timing_report(self, state: Control2VState) -> dict[str, Any]:
        """Settings, load times, stage timings per latent and their steady medians."""
        resolved = self._resolved
        stream = resolved.stream
        start = steady_from(resolved.history, len(state.timings))
        report: dict[str, Any] = {
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(resolved.args).items()
            },
            "fps": resolved.fps,
            "history": str(resolved.history),
            "frames": int(resolved.frames.shape[0]),
            "latents": len(state.timings),
            "load_s": stream.load_s,
            "steady_from_latent": start,
            "median_ms": timing_summary(state.timings, start),
            "cached_tokens": stream.cached_tokens,
            "per_latent_ms": state.timings,
        }
        if stream.device.type == "cuda":
            report["peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(stream.device) / 2**30
            )
            report["reserved_gib"] = torch.cuda.memory_reserved(stream.device) / 2**30
        return report


## Application


class Control2VApplication(IApplication):
    """Generate a video from a control clip and a caption, then finish."""

    def __init__(self, *, stream_loader: StreamLoader | None = None) -> None:
        """
        Args:
            stream_loader: Builds the model; :func:`load_stream` by default
                (a test passes a small one).
        """
        self._stream_loader = stream_loader or load_stream
        self._resolved: _Resolved | None = None

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse the arguments, read the control clip, then load the model.

        Raises:
            FileNotFoundError: The control video does not exist.
            ValueError: An argument is invalid.
        """
        args = parse_args(commandline_args)
        frames = read_control_clip(args.control, args.max_frames)
        fps = args.fps if args.fps is not None else read_control_fps(args.control)
        stream = self._stream_loader(args, fps)
        self._resolved = _Resolved(
            args=args,
            stream=stream,
            frames=frames,
            fps=fps,
            history=stream.model.history(args.history),
        )

    def session_desc(self) -> SessionDesc:
        """Every frame once, in order, at the control clip's frame rate."""
        resolved = self._resolved
        fps = DEFAULT_FPS if resolved is None else max(1, round(resolved.fps))
        metadata: dict[str, Any] = {"application": "control2v", "model": HF_REPO_ID}
        if resolved is not None:
            metadata.update(
                control=resolved.args.control.name,
                hint=resolved.args.hint,
                history=str(resolved.history),
                seed=resolved.args.seed,
            )
        return SessionDesc(
            output_layout=VideoTensorLayout.tchw,
            backpressure_mode=BackpressureMode.BLOCK,
            presentation_mode=PresentationMode.ON_DEMAND,
            frames_per_second_for_ui=fps,
            frames_per_second_for_step=fps,
            video_width=WIDTH,
            video_height=HEIGHT,
            metadata=metadata,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one pass over the clip.

        Raises:
            RuntimeError: :meth:`init` has not run.
            ValueError: Another layout or frame size is asked for.
        """
        if self._resolved is None:
            raise RuntimeError("Control2VApplication.init() must run first.")
        expected = self.session_desc()
        if session_desc.output_layout is not VideoTensorLayout.tchw:
            raise ValueError(
                f"control2v produces tchw output, got {session_desc.output_layout.value}"
            )
        size = (session_desc.video_width, session_desc.video_height)
        if size != (WIDTH, HEIGHT):
            raise ValueError(
                f"control2v produces {WIDTH}x{HEIGHT} frames, got {size[0]}x{size[1]}"
            )
        resolved = replace(
            session_desc,
            frames_per_second_for_ui=expected.frames_per_second_for_ui,
            frames_per_second_for_step=expected.frames_per_second_for_step,
            metadata={**session_desc.metadata, **expected.metadata},
        )
        return Control2VSession(self._resolved, resolved)

    def close(self) -> None:
        """Release the model."""
        if self._resolved is not None:
            self._resolved.stream.close()
            self._resolved = None


def create_app() -> IApplication:
    """Create the control-video-to-video application."""
    return Control2VApplication()


__all__ = [
    "SLUG",
    "Control2VApplication",
    "Control2VArgs",
    "Control2VModelLoop",
    "Control2VSession",
    "create_app",
    "load_stream",
    "parse_args",
    "read_control_clip",
    "timing_summary",
]
