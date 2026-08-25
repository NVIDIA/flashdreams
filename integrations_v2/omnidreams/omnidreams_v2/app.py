# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omnidreams application, generating driving video from a road layout."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnidreams.config import RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
from omnidreams.interactive_drive.math3d import normalize_camera_name

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.session import ISession
from flashdreams.infra.config import derive_config
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

from .conditioning import HDMapSource, PrecomputedHDMapSource, RenderedHDMapSource
from .ludus import LudusSceneRenderer
from .samples import DEFAULT_HDMAP_SAMPLE, fetch_hdmap_sample
from .scenes import (
    DEFAULT_SCENE,
    DEFAULT_SCENE_CAMERA,
    fetch_scene,
    read_prompt,
    read_seed_frame,
)
from .session import OmnidreamsSession

_RUNNER_CONFIG = RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE
"""Shipped single-view distilled model, and the defaults that come with it."""

_OUTPUT_LAYOUT = VideoTensorLayout.bvtchw
"""What this model emits: a batch of cameras, each a sequence of frames."""

_FRAMES_PER_SECOND_FOR_UI = 60
"""Rate an interactive window would read input and present results at."""

_DEFAULT_RUN_SECONDS = 60
"""How much video a run produces when it was not told how much to produce.

Long enough to show whether a drive holds together, short enough to wait for.
A run generates in real time at best, so an hour of road is an hour of waiting,
which is not what someone who named no length was asking for.
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class SceneDrive:
    """A scene to draw the road from, and where along it a run starts."""

    archive: Path
    """Scene on disk, holding the road layout and the drive recorded along it."""

    camera: str
    """Camera to draw from, spelled the way the scene spells it."""

    first_frame_path: Path
    """Frame the run continues from: the one unpacked from the scene, or one the
    command line named in its place."""

    view_start_us: int
    """When the scene's own frame was captured, which is where the drawing
    starts whichever frame the run continues from."""


@dataclass(frozen=True, kw_only=True, slots=True)
class OmnidreamsSessionConfig:
    """What one command line resolved to, shared by every session it creates."""

    prompt: str
    """Text describing the drive, applied to every camera."""

    device: str
    """Device the pipeline is built on."""

    max_blocks: int | None
    """Blocks to stop after: a count, ``0`` for no limit, or ``None`` for the
    default runtime, which takes the model's block sizes to work out."""

    hdmap_video_paths: tuple[Path, ...]
    """HDMap video per camera, in camera order. Empty for a drawn run."""

    first_frame_paths: tuple[Path, ...]
    """Frame to continue from per camera, in the same order. Empty for a drawn
    run, which takes its frame from the scene instead."""

    view_names: tuple[str, ...]
    """Camera labels, in the same order."""

    scene: SceneDrive | None
    """Scene to draw the road from, or ``None`` to replay a recording of it."""


class OmnidreamsApplication(IApplication):
    """Omnidreams: driving video generated from a rendered road layout.

    The model continues from a first frame and follows the HDMap it is given,
    so a run is described by where that layout comes from rather than by a
    prompt alone. It comes either from a scene drawn by the Ludus rasterizer as
    the run goes, which is what a run gets by default, or from recorded video,
    which is what ``--hdmap`` asks for instead.

    Drawing is the default because it is what a run that eventually steers
    needs: a recording cannot show a road nobody drove down. Nothing steers
    yet, so a drawn run follows the drive its scene recorded, which is as
    repeatable as replaying a video of it.

    A scene supplies the frame to continue from and the prompt as well as the
    layout, and ``--first-frame`` and ``--prompt`` each replace what it supplied
    without giving up the drawing. That is how the one road is driven under a
    sky or a season the scene never recorded.

    The model is loaded once, on the first session, and shared by every session
    after it, since loading reads a checkpoint of several gigabytes.
    """

    def __init__(
        self,
        *,
        pipeline_config: Any | None = None,
        source_factory: Callable[[OmnidreamsSessionConfig, SessionDesc], HDMapSource]
        | None = None,
    ) -> None:
        """
        Args:
            pipeline_config: Model to run, in place of the shipped distilled
                checkpoint. A test passes a stand-in.
            source_factory: Where conditioning comes from, in place of the
                recording or the scene the command line named. A test passes a
                stand-in for both.
        """
        self._pipeline_config = (
            _RUNNER_CONFIG.pipeline if pipeline_config is None else pipeline_config
        )
        self._source_factory = source_factory or _configured_source
        self._config: OmnidreamsSessionConfig | None = None
        self._pipeline: Any = None

    @property
    def pipeline_config(self) -> Any:
        """Model this will load, including whatever the command line changed."""
        return self._pipeline_config

    def init(self, commandline_args: Sequence[str]) -> None:
        """Parse what road to drive, how far along it, and where the road is.

        Also where the road layout comes from, which is what a scene and a
        recording are two answers to. Not what size or rate to generate at: that
        describes the session, which the caller asks for. The model is not
        loaded here either, though a scene is fetched so that a run that cannot
        find its road says so before waiting on a checkpoint.

        Raises:
            ValueError: The conditioning is described twice or by halves, or the
                rollout length is not one this model can generate.
            FileNotFoundError: The named scene or recording does not exist.
        """
        parser = argparse.ArgumentParser(
            prog="flashdreams-run-v2 omnidreams --",
            description="Generate driving video from a road layout.",
        )
        parser.add_argument(
            "--scene",
            default=None,
            metavar="ID_OR_PATH",
            help=(
                "Scene to draw the road layout from, as an id to download or a "
                "path to an archive. Drawing is what a run that names no "
                f"source of its own does. Default: {DEFAULT_SCENE}."
            ),
        )
        parser.add_argument(
            "--hdmap",
            nargs="*",
            default=None,
            metavar="ID_OR_PATH",
            help=(
                "Replay an already-rendered HDMap rather than drawing one: a "
                "sample id to download, or one video per camera. Bare, the "
                f"default sample ({DEFAULT_HDMAP_SAMPLE})."
            ),
        )
        parser.add_argument(
            "--first-frame",
            type=Path,
            nargs="+",
            default=(),
            metavar="PATH",
            help=(
                "Image or video to continue from, one per camera. Frame zero of "
                "a video is taken. Default: the frame the scene recorded, or "
                "the one a sample carries. Required alongside HDMap videos of "
                "your own, which carry none."
            ),
        )
        parser.add_argument(
            "--view-names",
            nargs="+",
            default=(),
            metavar="NAME",
            help=(
                "Camera labels, one per camera. Default: indexed placeholders, "
                "which is all a single-camera model needs."
            ),
        )
        parser.add_argument(
            "--prompt",
            default=None,
            help=(
                "Text describing the drive. Default: the scene's own "
                "description of it, or failing that the model's."
            ),
        )
        parser.add_argument(
            "--device",
            default="cuda",
            help="Device to load the model on. Default: %(default)s.",
        )
        parser.add_argument(
            "--max-blocks",
            type=int,
            default=None,
            help=(
                "Stop after this many blocks, or 0 to drive to the end of the "
                f"road however long that takes. Default: about "
                f"{_DEFAULT_RUN_SECONDS} seconds of video."
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
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help=(
                "Seed the noise a run samples from, so the same command "
                "generates the same clip. Default: whatever the model's config "
                "says, which is usually a seed of its own."
            ),
        )
        args = parser.parse_args(list(commandline_args))

        if args.max_blocks is not None and args.max_blocks < 0:
            raise ValueError(f"--max-blocks cannot be negative, got {args.max_blocks}.")
        scene = _resolve_scene(args)
        if scene is None:
            hdmap_videos, first_frames = _resolve_conditioning(
                args.hdmap, tuple(args.first_frame)
            )
            defaults = tuple(f"view_{index}" for index in range(len(hdmap_videos)))
        else:
            hdmap_videos, first_frames = (), ()
            # A scene names its cameras, so a drawn run can label its own view
            # rather than leaving a placeholder where the label goes.
            defaults = (normalize_camera_name(scene.camera)[1],)
        view_names = _resolve_view_names(tuple(args.view_names), defaults)

        if args.compile is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config,
                diffusion_model={"transformer": {"compile_network": args.compile}},
            )
        if args.seed is not None:
            self._pipeline_config = derive_config(
                self._pipeline_config, diffusion_model={"seed": args.seed}
            )
        self._config = OmnidreamsSessionConfig(
            prompt=_resolve_prompt(args.prompt, scene),
            device=args.device,
            max_blocks=args.max_blocks,
            hdmap_video_paths=hdmap_videos,
            first_frame_paths=first_frames,
            view_names=view_names,
            scene=scene,
        )

    def session_desc(self) -> SessionDesc:
        """Return the description of a session this application uses.

        The model generates its best video at the size and rate it was trained
        at, so that is what a caller with none of its own in mind gets.
        """
        return SessionDesc(
            output_layout=_OUTPUT_LAYOUT,
            frames_per_second_for_ui=_FRAMES_PER_SECOND_FOR_UI,
            frames_per_second_for_step=_RUNNER_CONFIG.output_fps,
            video_width=_RUNNER_CONFIG.pixel_width,
            video_height=_RUNNER_CONFIG.pixel_height,
        )

    def create_session(self, session_desc: SessionDesc) -> ISession:
        """Create one uninitialized session, loading the model if needed.

        Raises:
            RuntimeError: :meth:`init` has not run yet.
            ValueError: The description asks for output this cannot generate.
        """
        config = self._config
        if config is None:
            raise RuntimeError(
                f"{type(self).__name__}.init() must run before create_session()."
            )
        # Before loading rather than after: a checkpoint of several gigabytes is
        # a long wait for a layout this was never going to accept.
        self._validate_layout(session_desc)
        if self._pipeline is None:
            self._pipeline = self._pipeline_config.setup().to(config.device).eval()
        self._validate_frame_size(session_desc, self._pipeline)
        return OmnidreamsSession(
            self._pipeline,
            config.prompt,
            self._source_factory(config, session_desc),
            session_desc,
            self._resolve_max_blocks(config, session_desc),
        )

    def _resolve_max_blocks(
        self, config: OmnidreamsSessionConfig, session_desc: SessionDesc
    ) -> int | None:
        """Return the block count to stop a run after, or ``None`` for no limit.

        A default has to be worked out here rather than while parsing, because
        how many blocks a length of video takes is something only the loaded
        model can say.
        """
        if config.max_blocks == 0:
            return None
        if config.max_blocks is not None:
            return config.max_blocks
        return _blocks_for_seconds(
            self._pipeline,
            frames_per_second=session_desc.frames_per_second_for_step,
            seconds=_DEFAULT_RUN_SECONDS,
        )

    def close(self) -> None:
        """Release the model, and whatever memory it was holding."""
        pipeline = self._pipeline
        self._pipeline = None
        self._config = None
        close = getattr(pipeline, "close", None)
        if close is not None:
            close()

    def _validate_layout(self, session_desc: SessionDesc) -> None:
        """Reject a layout this model does not emit.

        Rejecting rather than resolving: a caller that asked for one video and
        silently received another has no way to notice.
        """
        if session_desc.output_layout is not _OUTPUT_LAYOUT:
            raise ValueError(
                f"This application only produces {_OUTPUT_LAYOUT.value} output, "
                f"got {session_desc.output_layout.value}."
            )

    def _validate_frame_size(self, session_desc: SessionDesc, pipeline: Any) -> None:
        """Reject a frame size that is not a whole number of latents across."""
        ratio = pipeline.decoder.spatial_compression_ratio
        if session_desc.video_width % ratio or session_desc.video_height % ratio:
            raise ValueError(
                f"Frame dimensions must be multiples of {ratio}, got "
                f"{session_desc.video_width}x{session_desc.video_height}."
            )


def create_app() -> IApplication:
    """Return a new Omnidreams application."""
    return OmnidreamsApplication()


def _configured_source(
    config: OmnidreamsSessionConfig, session_desc: SessionDesc
) -> HDMapSource:
    """Return conditioning for a run, drawn or replayed as it asked."""
    if config.scene is not None:
        return _rendered_source(config, session_desc)
    return _recorded_source(config, session_desc)


def _recorded_source(
    config: OmnidreamsSessionConfig, session_desc: SessionDesc
) -> HDMapSource:
    """Return conditioning read from the recordings the command line named."""
    return PrecomputedHDMapSource(
        hdmap_video_paths=config.hdmap_video_paths,
        first_frame_paths=config.first_frame_paths,
        view_names=config.view_names,
        pixel_width=session_desc.video_width,
        pixel_height=session_desc.video_height,
        device=config.device,
    )


def _rendered_source(
    config: OmnidreamsSessionConfig, session_desc: SessionDesc
) -> HDMapSource:
    """Return conditioning drawn from the scene the command line named.

    Raises:
        RuntimeError: The run has no scene, so there is nothing to draw.
    """
    scene = config.scene
    if scene is None:
        raise RuntimeError("A drawn run needs a scene to draw.")
    return RenderedHDMapSource(
        renderer=LudusSceneRenderer(
            scene_path=scene.archive,
            camera=scene.camera,
            view_start_us=scene.view_start_us,
            # Drawn at the rate the model generates at, so one drawn frame is
            # one generated frame and the drive runs at the speed it recorded.
            frames_per_second=session_desc.frames_per_second_for_step,
            pixel_width=session_desc.video_width,
            pixel_height=session_desc.video_height,
            device=config.device,
        ),
        first_frame_path=scene.first_frame_path,
        view_name=config.view_names[0],
        pixel_width=session_desc.video_width,
        pixel_height=session_desc.video_height,
        device=config.device,
    )


def _resolve_scene(args: argparse.Namespace) -> SceneDrive | None:
    """Return the scene to draw the road from, or ``None`` to replay a recording.

    Drawing is what a run that named no source of its own does, a scene being
    the source that has a road in it rather than a picture of one, and so the
    only one a run could eventually be steered through.

    ``--first-frame`` is not what decides: it says what the run continues from,
    which is a question a drawn run answers too, and answering it with a frame
    of your own is how a scene's road is driven under a sky it never saw.

    Raises:
        ValueError: A scene was named alongside a recording, which are two
            answers to the one question of where the layout comes from. Or more
            than one frame was named for the one camera drawn.
    """
    if args.hdmap is not None:
        if args.scene is not None:
            raise ValueError(
                "--scene draws the road layout as the run goes, so it cannot "
                "be combined with --hdmap, which replays a recording of one."
            )
        return None
    first_frames = tuple(args.first_frame)
    if len(first_frames) > 1:
        raise ValueError(
            f"Got {len(first_frames)} first frames for a drawn run, which draws "
            "the one camera the scene is drawn from. Pass one."
        )
    archive = fetch_scene(args.scene or DEFAULT_SCENE)
    # Read even when overridden: the timestamp is where the drawing starts, and
    # a frame of your own carries no such moment of its own.
    recorded_frame, view_start_us = read_seed_frame(
        archive, camera=DEFAULT_SCENE_CAMERA
    )
    return SceneDrive(
        archive=archive,
        camera=DEFAULT_SCENE_CAMERA,
        first_frame_path=first_frames[0] if first_frames else recorded_frame,
        view_start_us=view_start_us,
    )


def _resolve_prompt(prompt: str | None, scene: SceneDrive | None) -> str:
    """Return the text describing the drive.

    What the command line said, else what the scene says about itself, else the
    model's own, which describes the sort of drive it was trained on rather than
    this one in particular.
    """
    if prompt is not None:
        return prompt
    if scene is not None:
        described = read_prompt(scene.archive)
        if described is not None:
            return described
    return _RUNNER_CONFIG.prompt


def _blocks_for_seconds(pipeline: Any, *, frames_per_second: int, seconds: int) -> int:
    """Return the blocks whose frames add up to ``seconds`` of video.

    Counted off the model rather than divided out, because the blocks are not
    all the same length: a causal decoder's first one is usually shorter, and
    how it divides up a rollout is its own business.

    Raises:
        ValueError: The model reports a block with no frames in it, so no
            number of them would ever amount to a run.
    """
    wanted = frames_per_second * seconds
    frames = 0
    blocks = 0
    while frames < wanted:
        length = pipeline.get_num_frames(blocks)
        if length <= 0:
            raise ValueError(
                f"The model reports {length} frames in block {blocks}, so a "
                "run length cannot be worked out from it. Pass --max-blocks."
            )
        frames += length
        blocks += 1
    return blocks


def _resolve_conditioning(
    hdmap: list[str],
    first_frames: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return the recording and the frame to continue from, per camera.

    Only reached by a run that named ``--hdmap``, which is what asking to replay
    something means. What it was given decides which kind: nothing at all is the
    default sample, a lone bare id is a sample to download, and anything else is
    one video per camera.

    Raises:
        ValueError: A sample was named alongside a frame it already carries, or
            videos were named without one, or a different number of each was
            named. The model rejects a camera count it was not trained for, but
            only once it has loaded, so the two are counted against each other
            here.
        FileNotFoundError: A lone ``--hdmap`` word is neither a path nor a
            sample the dataset has.
    """
    sample = _named_sample(hdmap)
    if sample is not None:
        if first_frames:
            raise ValueError(
                f"--hdmap {sample} is a sample to download, which carries its "
                "own frame to continue from, so it cannot be combined with "
                "--first-frame."
            )
        try:
            recording, first_frame = fetch_hdmap_sample(sample)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No HDMap {sample!r}: there is no such path, and it is no "
                f"sample either ({exc})."
            ) from exc
        return (recording,), (first_frame,)
    hdmap_videos = tuple(Path(named) for named in hdmap)
    if not first_frames:
        raise ValueError("--first-frame is required alongside HDMap videos.")
    if len(first_frames) != len(hdmap_videos):
        raise ValueError(
            f"Got {len(hdmap_videos)} HDMap video(s) and {len(first_frames)} "
            "first frame(s): pass one of each per camera."
        )
    return hdmap_videos, first_frames


def _named_sample(hdmap: list[str]) -> str | None:
    """Return the sample id ``--hdmap`` named, or ``None`` if it named videos.

    A sample is named the way it is listed, by a bare id with no file extension
    on it. Telling them apart by extension rather than by what exists means a
    misspelled video is reported as the missing video it is, instead of being
    looked for in a dataset it was never in. A file that is sitting right there
    still wins, extension or not.
    """
    if not hdmap:
        return DEFAULT_HDMAP_SAMPLE
    if len(hdmap) > 1:
        return None
    lone = Path(hdmap[0])
    return None if lone.suffix or lone.exists() else hdmap[0]


def _resolve_view_names(
    view_names: tuple[str, ...], defaults: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the camera labels, which only a multi-camera model reads.

    Args:
        view_names: What the command line asked for, if anything.
        defaults: A label per camera to fall back on, which is also how many
            cameras this run has.

    Raises:
        ValueError: Some cameras were labelled and others were not.
    """
    if not view_names:
        return defaults
    if len(view_names) != len(defaults):
        raise ValueError(
            f"Got {len(defaults)} camera(s) and {len(view_names)} view "
            "name(s): pass one name per camera, or none at all."
        )
    return view_names
