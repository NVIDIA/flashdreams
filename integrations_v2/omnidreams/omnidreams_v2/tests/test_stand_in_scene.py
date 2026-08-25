# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for a renderer rather than precomputed run.

The rasterizer needs a GPU and a scene of real road, so what a stand-in covers
is everything around it. Three things in particular, each of which would produce
a plausible-looking but wrong drive rather than an error: that a run draws its
way along a scene consecutively, that what comes back reaches the model as
pixels of the range and layout it reads, and that the frame a run continues from
and the layout it is shown begin at the same moment.
"""

import zipfile
from pathlib import Path

import omnidreams.scenes
import pytest
import torch
from omnidreams_v2 import (
    LudusSceneRenderer,
    OmnidreamsApplication,
    OmnidreamsSessionConfig,
    RenderedHDMapSource,
)
from omnidreams_v2 import app as app_module
from omnidreams_v2 import scenes as scenes_module
from omnidreams_v2.scenes import DEFAULT_SCENE
from PIL import Image

from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu

_NO_EVENTS = UserInputEvents([])
"""What a window with nobody on it reports."""

_SCENE_CAMERA = "camera_front_wide_120fov"
"""The camera a scene records, in the spelling the model uses for it."""

_CAPTURED_US = 1_700_000_000_000_000
"""When the frame a run continues from was captured, as scenes count time."""


def test_a_drawn_run_works_its_way_along_the_scene() -> None:
    """Consecutively and without gaps, since a gap is a jump in the road the
    model is shown while the video it generates stays continuous."""
    renderer = StubRenderer(frame_count=64)
    source = _source(renderer)
    source.open()

    source.next_chunk(5, _NO_EVENTS)
    source.next_chunk(8, _NO_EVENTS)
    source.next_chunk(8, _NO_EVENTS)

    assert renderer.drawn == [(0, 5), (5, 8), (13, 8)]


def test_what_is_drawn_reaches_the_model_as_pixels_it_reads() -> None:
    """A rasterizer produces bytes laid out for a screen; the model reads signed
    pixels laid out for a convolution. Getting this wrong leaves a run
    conditioned on a washed-out or channel-swapped road."""
    renderer = StubRenderer(frame_count=8, height=4, width=6)
    source = _source(renderer)
    source.open()

    chunk = source.next_chunk(2, _NO_EVENTS)

    # [B, V, T, C, H, W], one camera of one batch.
    assert chunk.shape == (1, 1, 2, 3, 4, 6)
    assert chunk.dtype is torch.bfloat16
    # The stub draws the darkest and brightest byte it can, which are the ends
    # of the range the model expects.
    assert float(chunk.min()) == -1.0
    assert float(chunk.max()) == 1.0


def test_a_drawn_run_ends_when_the_scene_does() -> None:
    """The scene is what says how long a drawn run is, the same way a recording
    does for a replayed one."""
    renderer = StubRenderer(frame_count=10)
    source = _source(renderer)
    source.open()

    source.next_chunk(5, _NO_EVENTS)

    assert source.has_frames(5)
    assert not source.has_frames(6)


def test_a_drawn_run_refuses_to_draw_past_the_end_of_the_scene() -> None:
    renderer = StubRenderer(frame_count=10)
    source = _source(renderer)
    source.open()

    with pytest.raises(RuntimeError, match="10 frames long"):
        source.next_chunk(11, _NO_EVENTS)


def test_a_reset_returns_a_drawn_run_to_the_start_of_the_scene() -> None:
    renderer = StubRenderer(frame_count=64)
    source = _source(renderer)
    source.open()
    source.next_chunk(5, _NO_EVENTS)

    source.reset()
    source.next_chunk(5, _NO_EVENTS)

    assert renderer.drawn == [(0, 5), (0, 5)]


def test_a_drawn_run_continues_from_the_scenes_own_recorded_frame(
    tmp_path: Path,
) -> None:
    """In the layout the model reads it in, which is the same one the replayed
    path hands over, since it is the same model reading it."""
    source = _source(StubRenderer(frame_count=8), first_frame=_a_png(tmp_path, 8, 6))
    source.open()

    frame = source.first_frame()

    assert frame.shape == (1, 1, 1, 3, 4, 6)


def test_the_layout_is_drawn_from_the_moment_the_run_continues_from() -> None:
    """The one alignment a drawn run cannot get wrong quietly: shown the road
    from a different moment than the frame it continues from, the model would be
    asked to drive a corner that is not in front of it. Reaching for the private
    timeline because the alternative is covering it only on a GPU."""
    renderer = _renderer(view_start_us=100_000, frames_per_second=30)
    recorded = torch.tensor([0, 500_000, 1_000_000], dtype=torch.int64)

    timeline = renderer._timeline(recorded)

    assert int(timeline[0]) == 100_000
    # Drawn at the rate the model generates at, not the rate the drive was
    # recorded at, so one drawn frame is one generated frame.
    assert int(timeline[1] - timeline[0]) == round(1_000_000 / 30)
    assert int(timeline[-1]) <= 1_000_000


def test_a_run_starting_before_its_scene_was_recorded_starts_where_it_was() -> None:
    """A scene that disagrees with itself about when its drive began would
    otherwise be drawn as empty road."""
    renderer = _renderer(view_start_us=0, frames_per_second=30)
    recorded = torch.tensor([400_000, 900_000], dtype=torch.int64)

    timeline = renderer._timeline(recorded)

    assert int(timeline[0]) == 400_000


def test_a_scene_whose_drive_ends_before_the_run_starts_is_refused() -> None:
    renderer = _renderer(view_start_us=900_000, frames_per_second=30)
    recorded = torch.tensor([0, 500_000], dtype=torch.int64)

    with pytest.raises(ValueError, match="nothing to draw"):
        renderer._timeline(recorded)


def test_a_drawn_run_takes_its_prompt_and_first_frame_from_the_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scene describes the road it holds, which is a better description of the
    drive than the model's own general one."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path, prompt="A quiet suburban boulevard at dusk.")
    app = OmnidreamsApplication(pipeline_config=object())

    app.init(["--scene", str(archive), "--device", "cpu"])

    config = _config_of(app)
    assert config.prompt == "A quiet suburban boulevard at dusk."
    assert config.scene is not None
    assert config.scene.view_start_us == _CAPTURED_US
    assert config.scene.first_frame_path.read_bytes() == b"a recorded frame"
    # The scene names its camera, so a drawn run labels its view rather than
    # leaving the placeholder a nameless recording gets.
    assert config.view_names == (_SCENE_CAMERA,)


@pytest.mark.parametrize("arguments", [[], ["--scene", "a-named-scene"]])
def test_a_run_that_names_no_source_draws_the_default_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    """Including a run given no arguments at all, so generating something takes
    none. Drawing is the default because it is the source a run could one day
    be steered through."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path)
    requested: list[str] = []

    def fetch(scene: str) -> Path:
        requested.append(scene)
        return archive

    monkeypatch.setattr(app_module, "fetch_scene", fetch)
    app = OmnidreamsApplication(pipeline_config=object())

    app.init([*arguments, "--device", "cpu"])

    expected = "a-named-scene" if arguments else DEFAULT_SCENE
    assert requested == [expected]
    assert _config_of(app).scene is not None


def test_a_prompt_on_the_command_line_beats_the_scenes_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path, prompt="A quiet suburban boulevard at dusk.")
    app = OmnidreamsApplication(pipeline_config=object())

    app.init(["--scene", str(archive), "--prompt", "Heavy rain.", "--device", "cpu"])

    assert _config_of(app).prompt == "Heavy rain."


def test_a_drawn_run_can_continue_from_a_frame_of_your_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which is what drives the one road under a sky it never recorded: the
    layout is still drawn, only the picture the model continues from changes."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path)
    mine = _a_png(tmp_path, 6, 4)
    app = OmnidreamsApplication(pipeline_config=object())

    app.init(["--scene", str(archive), "--first-frame", str(mine), "--device", "cpu"])

    scene = _config_of(app).scene
    assert scene is not None
    assert scene.first_frame_path == mine
    # Still drawn, and still drawn from the moment the scene recorded: a frame
    # of your own says nothing about where along the road it was taken.
    assert scene.view_start_us == _CAPTURED_US


def test_a_frame_of_your_own_is_enough_to_ask_for_a_drawn_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming a frame is not asking to replay anything, so it leaves the default
    scene being drawn rather than demanding a recording to go with it."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path)
    monkeypatch.setattr(app_module, "fetch_scene", lambda scene: archive)
    mine = _a_png(tmp_path, 6, 4)
    app = OmnidreamsApplication(pipeline_config=object())

    app.init(["--first-frame", str(mine), "--device", "cpu"])

    scene = _config_of(app).scene
    assert scene is not None
    assert scene.first_frame_path == mine


def test_a_drawn_run_cannot_continue_from_more_frames_than_it_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One camera is drawn, so a second frame is a camera that never gets one."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path)
    app = OmnidreamsApplication(pipeline_config=object())

    with pytest.raises(ValueError, match="Pass one"):
        app.init(
            [
                "--scene",
                str(archive),
                "--first-frame",
                "a.png",
                "b.png",
                "--device",
                "cpu",
            ]
        )


@pytest.mark.parametrize(
    "replaying",
    [
        ["--hdmap"],
        ["--hdmap", "road.mp4"],
        ["--hdmap", "a-recorded-drive"],
    ],
)
def test_a_scene_and_a_recording_cannot_both_say_where_the_layout_comes_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replaying: list[str]
) -> None:
    """They are two answers to one question, and quietly preferring either one
    would leave a run doing something other than what it was asked to."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = _a_scene(tmp_path)
    app = OmnidreamsApplication(pipeline_config=object())

    with pytest.raises(ValueError, match="cannot be combined"):
        app.init(["--scene", str(archive), *replaying, "--device", "cpu"])


def test_a_scene_with_nothing_recorded_to_continue_from_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before the model loads, since a checkpoint of several gigabytes is a long
    wait for a scene that was never going to work."""
    monkeypatch.setattr(omnidreams.scenes, "FLASHDREAMS_CACHE_DIR", tmp_path / "cache")
    archive = tmp_path / "empty.usdz"
    with zipfile.ZipFile(archive, "w") as scene:
        scene.writestr("prompt.txt", "A road with no frames of it.")
    app = OmnidreamsApplication(pipeline_config=object())

    with pytest.raises(FileNotFoundError, match="no timestamped frames"):
        app.init(["--scene", str(archive), "--device", "cpu"])


def test_a_scene_that_is_neither_a_path_nor_a_known_id_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named as one message rather than two, since someone who mistyped a path
    does not want to be told their typo is not a scene id either."""

    def missing(scene_uuid: str) -> Path:
        raise OSError(f"no scene {scene_uuid}")

    monkeypatch.setattr(scenes_module, "hf_hub_download_scene", missing)
    app = OmnidreamsApplication(pipeline_config=object())

    with pytest.raises(FileNotFoundError, match="no such path"):
        app.init(["--scene", "not-a-scene", "--device", "cpu"])


## Stand-ins


class StubRenderer:
    """A scene of a known length, without a scene or a rasterizer.

    Draws the darkest and brightest bytes a rasterizer can, so a test can tell
    whether the conversion to model pixels covered the range.
    """

    def __init__(self, *, frame_count: int, height: int = 4, width: int = 6) -> None:
        self._frame_count = frame_count
        self._height = height
        self._width = width
        self.drawn: list[tuple[int, int]] = []
        self.opened = 0
        self.closed = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def open(self) -> None:
        self.opened += 1

    def render(self, start: int, count: int) -> torch.Tensor:
        self.drawn.append((start, count))
        frames = torch.zeros(count, self._height, self._width, 3, dtype=torch.uint8)
        frames[..., 0, :] = 255
        return frames

    def close(self) -> None:
        self.closed += 1


## Helpers


def _source(
    renderer: StubRenderer, *, first_frame: Path | None = None
) -> RenderedHDMapSource:
    """Return a drawn source over a stand-in renderer."""
    return RenderedHDMapSource(
        renderer=renderer,
        first_frame_path=first_frame or Path("unused.png"),
        view_name=_SCENE_CAMERA,
        pixel_width=6,
        pixel_height=4,
        device="cpu",
    )


def _renderer(*, view_start_us: int, frames_per_second: int) -> LudusSceneRenderer:
    """Return a renderer that has not loaded anything, for its timeline alone."""
    return LudusSceneRenderer(
        scene_path=Path("unused.usdz"),
        camera=_SCENE_CAMERA,
        view_start_us=view_start_us,
        frames_per_second=frames_per_second,
        pixel_width=6,
        pixel_height=4,
        device="cpu",
    )


def _a_scene(tmp_path: Path, *, prompt: str | None = None) -> Path:
    """Write a scene archive holding one recorded frame, and maybe a prompt.

    The frame is not a real image: nothing here decodes one, and a test that
    said otherwise would be claiming to cover more than it does.
    """
    archive = tmp_path / "clipgt-a-scene.usdz"
    with zipfile.ZipFile(archive, "w") as scene:
        for captured_us in (_CAPTURED_US + 33_333, _CAPTURED_US):
            scene.writestr(
                f"frames/{_SCENE_CAMERA}/{captured_us}.jpeg", b"a recorded frame"
            )
        if prompt is not None:
            scene.writestr("prompt.txt", prompt)
    return archive


def _a_png(tmp_path: Path, width: int, height: int) -> Path:
    """Write an image for the one test that decodes what it continues from."""
    path = tmp_path / "first_frame.png"
    Image.new("RGB", (width, height), color=(20, 40, 60)).save(path)
    return path


def _config_of(app: OmnidreamsApplication) -> OmnidreamsSessionConfig:
    """Return what an application's command line resolved to."""
    config = app._config
    assert config is not None
    return config
