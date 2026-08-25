# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Omnidreams application, against stand-ins.

What is particular to a model conditioned on a road layout: every step is
conditioned on exactly the frames it generates, and a run ends when the layout
does. Neither needs a checkpoint to cover. A run against the real one, which
the other v2 integrations have as ``test_real_model.py``, is not written yet:
it needs a recorded drive to condition on as well as the checkpoint.
"""

import shutil
from pathlib import Path

import pytest
import torch
from omnidreams.config import RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE as _RUNNER
from omnidreams_v2 import OmnidreamsApplication, OmnidreamsSessionConfig
from omnidreams_v2 import app as app_module
from omnidreams_v2.samples import DEFAULT_HDMAP_SAMPLE

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_NO_EVENTS = UserInputEvents([])
"""What a window with nobody on it reports."""

_CONDITIONING_ARGS = [
    "--hdmap",
    "drive_hdmap.mp4",
    "--first-frame",
    "drive_first_frame.png",
    "--device",
    "cpu",
]
"""Arguments naming conditioning that a stand-in source stands in for."""


def test_the_model_says_what_it_generates_without_being_told() -> None:
    """The numbers are the checkpoint's, read off the runner config this
    integration already ships rather than written down again."""
    app = OmnidreamsApplication(pipeline_config=FakePipelineConfig())

    desc = app.session_desc()

    assert (desc.video_width, desc.video_height) == (
        _RUNNER.pixel_width,
        _RUNNER.pixel_height,
    )
    assert desc.frames_per_second_for_step == _RUNNER.output_fps
    assert desc.output_layout is VideoTensorLayout.bvtchw


def test_every_step_is_conditioned_on_the_frames_it_is_about_to_generate() -> None:
    """The chunk handed to the model covers the block it generates and no more,
    and consecutive blocks read consecutive stretches of the drive."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=24, pipeline=pipeline)
    window = RecordingClientWindow()

    _run(pipeline, source, window)

    assert source.chunks_read == [(0, 5), (5, 13), (13, 21)]
    assert pipeline.conditioned_frames == [5, 8, 8]
    assert [result.frame_count for result in window.results] == [5, 8, 8]


def test_a_run_ends_when_the_road_does() -> None:
    """Three frames are left over, which is less than a block, so the run stops
    rather than generating a partly conditioned one. Nothing asked it to stop:
    the recording is what says how long a drive is."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=24, pipeline=pipeline)

    _run(pipeline, source, RecordingClientWindow())

    assert source.frames_left == 3


def test_a_run_can_be_cut_short_of_the_road() -> None:
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=1000, pipeline=pipeline)
    window = RecordingClientWindow()

    _run(pipeline, source, window, max_blocks=2)

    assert [result.frame_count for result in window.results] == [5, 8]


def test_a_run_told_no_length_produces_about_a_minute() -> None:
    """Rather than driving to the end of the road, which on a long scene is an
    hour of generating for someone who only asked to see it work. It stops on
    the block that reaches a minute, a block being the smallest thing a run can
    produce."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=100_000, pipeline=pipeline)
    window = RecordingClientWindow()

    _run(pipeline, source, window)

    a_minute = _RUNNER.output_fps * 60
    generated = sum(result.frame_count for result in window.results)
    assert generated >= a_minute
    assert generated - window.results[-1].frame_count < a_minute


def test_a_run_can_be_told_to_drive_to_the_end_of_the_road() -> None:
    """Which is what a session someone is sitting in front of wants, and what
    the default minute would otherwise cut off. This road is longer than a
    minute, so the two answers are told apart."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=2000, pipeline=pipeline)

    _run(pipeline, source, RecordingClientWindow(), max_blocks=0)

    assert source.frames_left == 3


def test_a_reset_drives_the_same_route_again_from_the_start() -> None:
    """Both halves of a run start over: the cache the model generates from, and
    the source's place in the drive."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=1000, pipeline=pipeline)
    app = _application(pipeline, source)
    app.init(_CONDITIONING_ARGS)
    session = app.create_session(_stand_in_session_desc(pipeline))
    session.init()
    session.step(0, _NO_EVENTS)

    session.reset()
    session.step(0, _NO_EVENTS)

    assert source.chunks_read == [(0, 5), (0, 5)]
    assert len(pipeline.caches) == 2


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (["--hdmap", "a.mp4"], "--first-frame is required"),
        (
            ["--hdmap", "a.mp4", "b.mp4", "--first-frame", "a.png"],
            "one of each per camera",
        ),
        (
            ["--hdmap", "5f0e1a2b-drive", "--first-frame", "a.png"],
            "cannot be combined with --first-frame",
        ),
        (
            [*_CONDITIONING_ARGS, "--view-names", "left", "right"],
            "one name per camera",
        ),
        ([*_CONDITIONING_ARGS, "--max-blocks", "-1"], "cannot be negative"),
    ],
)
def test_a_run_that_describes_its_drive_incompletely_is_refused(
    arguments: list[str], expected: str
) -> None:
    """Before the model loads, since a checkpoint of several gigabytes is a long
    wait for a typo."""
    app = OmnidreamsApplication(pipeline_config=FakePipelineConfig())

    with pytest.raises(ValueError, match=expected):
        app.init(arguments)


@pytest.mark.parametrize(
    ("arguments", "expected_sample"),
    [
        (["--hdmap"], DEFAULT_HDMAP_SAMPLE),
        (["--hdmap", "5f0e1a2b-drive"], "5f0e1a2b-drive"),
    ],
)
def test_a_run_that_asks_to_replay_without_naming_what_gets_a_sample(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str], expected_sample: str
) -> None:
    """Bare ``--hdmap`` is how you replay something without having a recording
    to hand, and a bare id is how you replay one of the listed samples: neither
    has a file extension on it, which is what tells them from a video. The
    download itself is stubbed, since a unit test that reaches Hugging Face is
    not one."""
    recording, first_frame = Path("road_hdmap.mp4"), Path("road_first_frame.png")
    requested: list[str] = []

    def fetch(sample_id: str) -> tuple[Path, Path]:
        requested.append(sample_id)
        return recording, first_frame

    monkeypatch.setattr(app_module, "fetch_hdmap_sample", fetch)
    pipeline = FakePipeline()
    configs: list[OmnidreamsSessionConfig] = []

    def factory(
        config: OmnidreamsSessionConfig, session_desc: SessionDesc
    ) -> FakeHDMapSource:
        del session_desc
        configs.append(config)
        return FakeHDMapSource(total_frames=1000, pipeline=pipeline)

    app = OmnidreamsApplication(
        pipeline_config=FakePipelineConfig(pipeline), source_factory=factory
    )
    app.init([*arguments, "--device", "cpu"])
    app.create_session(_stand_in_session_desc(pipeline))

    assert requested == [expected_sample]
    assert configs[0].hdmap_video_paths == (recording,)
    assert configs[0].first_frame_paths == (first_frame,)


def test_compilation_can_be_turned_off_for_a_run() -> None:
    """Run against the real config rather than a stand-in, since what this
    covers is the override landing where this model keeps the setting. No model
    is loaded to answer it."""
    app = OmnidreamsApplication()

    app.init([*_CONDITIONING_ARGS, "--no-compile"])

    assert app.pipeline_config.diffusion_model.transformer.compile_network is False


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="writing an MP4 needs ffmpeg on PATH"
)
def test_a_run_writes_every_generated_frame_to_an_mp4(tmp_path: Path) -> None:
    """This model emits a camera dimension the others do not, so the run to a
    file is worth covering rather than assuming."""
    pipeline = FakePipeline()
    source = FakeHDMapSource(total_frames=1000, pipeline=pipeline)
    path = tmp_path / "drive.mp4"

    _run(pipeline, source, Mp4ClientWindow(path), max_blocks=3)

    assert path.stat().st_size > 0


## Stand-ins


class FakePipeline:
    """A model's worth of behaviour, without a model.

    Generates frames of the shape and range the real pipeline does, including
    the camera dimension, so the seam a checkpoint plugs into is covered on a
    CPU. Every call is recorded, so a test can assert the rollout was driven in
    order and conditioned on the right frames.
    """

    def __init__(
        self,
        *,
        width: int = 128,
        height: int = 64,
        compression_ratio: int = 8,
        first_block_frames: int = 5,
        block_frames: int = 8,
    ) -> None:
        """
        Args:
            width: Frame width to generate. Not square by default, so a
                transposed frame cannot pass unnoticed.
            height: Frame height to generate.
            compression_ratio: Pixels one latent covers in each direction.
            first_block_frames: Frames the first block decodes, which a causal
                decoder has fewer of than the rest.
            block_frames: Frames every block after the first decodes.
        """
        self.decoder = FakeDecoder(compression_ratio)
        self.width = width
        self.height = height
        self.first_block_frames = first_block_frames
        self.block_frames = block_frames
        self.device: str | None = None
        self.caches: list[dict[str, object]] = []
        self.generated: list[int] = []
        self.conditioned_frames: list[int] = []
        self.closed = False
        self._frames_generated = 0

    def to(self, device: str) -> "FakePipeline":
        self.device = device
        return self

    def eval(self) -> "FakePipeline":
        return self

    def initialize_cache(self, **kwargs: object) -> object:
        self.caches.append(kwargs)
        self._frames_generated = 0
        return object()

    def get_num_frames(self, autoregressive_index: int) -> int:
        if autoregressive_index == 0:
            return self.first_block_frames
        return self.block_frames

    def generate(
        self, *, autoregressive_index: int, cache: object, hdmap: torch.Tensor
    ) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        self.conditioned_frames.append(int(hdmap.shape[2]))
        count = self.get_num_frames(autoregressive_index)
        frames = torch.stack(
            [self._frame(self._frames_generated + index) for index in range(count)]
        )
        self._frames_generated += count
        # [T, C, H, W] as one camera of one batch: [B, V, T, C, H, W].
        return frames.unsqueeze(0).unsqueeze(0)

    def finalize(self, *, autoregressive_index: int, cache: object) -> dict[str, float]:
        del autoregressive_index, cache
        return {"total_ms": 1.5}

    def close(self) -> None:
        self.closed = True

    def _frame(self, frame_index: int) -> torch.Tensor:
        """Return a grey frame whose shade moves with time, so a check made of a
        real video is meaningful here too."""
        shade = -0.5 + (frame_index % 8) / 8.0
        return torch.full((3, self.height, self.width), shade, dtype=torch.float32)


class FakePipelineConfig:
    """A pipeline config that builds a stand-in rather than loading a model."""

    def __init__(self, pipeline: FakePipeline | None = None) -> None:
        self.pipeline = pipeline if pipeline is not None else FakePipeline()

    def setup(self) -> FakePipeline:
        return self.pipeline


class FakeDecoder:
    """The one thing an application asks a decoder for."""

    def __init__(self, spatial_compression_ratio: int) -> None:
        self.spatial_compression_ratio = spatial_compression_ratio


class FakeHDMapSource:
    """A drive of a known length, without a recording of one."""

    def __init__(
        self,
        *,
        total_frames: int,
        pipeline: FakePipeline,
        view_names: tuple[str, ...] = ("view_0",),
    ) -> None:
        """
        Args:
            total_frames: Frames of conditioning this drive has in it.
            pipeline: Stand-in whose frame size the chunks match.
            view_names: Cameras this supplies.
        """
        self._total_frames = total_frames
        self._pipeline = pipeline
        self._view_names = view_names
        self._cursor = 0
        self.chunks_read: list[tuple[int, int]] = []
        self.opened = 0
        self.closed = 0

    @property
    def view_names(self) -> tuple[str, ...]:
        return self._view_names

    @property
    def frames_left(self) -> int:
        """Conditioning the run never reached."""
        return self._total_frames - self._cursor

    def open(self) -> None:
        self.opened += 1

    def first_frame(self) -> torch.Tensor:
        return torch.zeros(
            1, len(self._view_names), 1, 3, self._pipeline.height, self._pipeline.width
        )

    def has_frames(self, frame_count: int) -> bool:
        return self._cursor + frame_count <= self._total_frames

    def next_chunk(self, frame_count: int, events: UserInputEvents) -> torch.Tensor:
        del events
        end = self._cursor + frame_count
        self.chunks_read.append((self._cursor, end))
        self._cursor = end
        return torch.zeros(
            1,
            len(self._view_names),
            frame_count,
            3,
            self._pipeline.height,
            self._pipeline.width,
        )

    def reset(self) -> None:
        self._cursor = 0

    def close(self) -> None:
        self.closed += 1


class RecordingClientWindow(IClientWindow):
    """Keep what a run generated, and report no input."""

    def __init__(self) -> None:
        self.results: list[StepResult] = []

    def get_user_input_events(self) -> UserInputEvents:
        return _NO_EVENTS

    def open(self, session_desc: SessionDesc) -> None:
        del session_desc

    def write(self, result: StepResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        return


## Helpers


def _application(
    pipeline: FakePipeline, source: FakeHDMapSource
) -> OmnidreamsApplication:
    """Return an application over both stand-ins."""
    return OmnidreamsApplication(
        pipeline_config=FakePipelineConfig(pipeline),
        source_factory=lambda config, session_desc: source,
    )


def _stand_in_session_desc(pipeline: FakePipeline) -> SessionDesc:
    """Return the session the stand-in generates, rather than the checkpoint's."""
    return SessionDesc(
        output_layout=VideoTensorLayout.bvtchw,
        frames_per_second_for_step=_RUNNER.output_fps,
        video_width=pipeline.width,
        video_height=pipeline.height,
    )


def _run(
    pipeline: FakePipeline,
    source: FakeHDMapSource,
    window: IClientWindow,
    *,
    max_blocks: int | None = None,
) -> None:
    """Run one application to completion against ``window``."""
    limit = [] if max_blocks is None else ["--max-blocks", str(max_blocks)]
    ApplicationRunner(_application(pipeline, source), window).run(
        _stand_in_session_desc(pipeline), [*_CONDITIONING_ARGS, *limit]
    )
