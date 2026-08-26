# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the Waypoint V2 application and session."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from numpy import uint64
from torch import Tensor
from waypoint import WaypointControl
from waypoint.pipeline import WaypointInferencePipeline
from waypoint_v2.app import WaypointApplication, load_seed_display_frames
from waypoint_v2.control_events import ControlEventAdapter
from waypoint_v2.session import WaypointModelLoop, WaypointSession

from flashdreams.api_v2.user_input_event_data import UserInputEventData
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEventData,
    KeyboardInputState,
    KeyboardUserInputEventData,
    MouseUserInputEventData,
    ResetUserInputEventData,
    UserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _FakeDiffusionModel:
    dtype = torch.float32

    def __init__(self, seed: int) -> None:
        self.rng = torch.Generator().manual_seed(seed)


class _FakePipeline:
    device = torch.device("cpu")

    def __init__(self, seed: int = 7) -> None:
        self.diffusion_model = _FakeDiffusionModel(seed)
        self.initialized_caches: list[dict[str, Any]] = []
        self.generate_calls: list[tuple[int, WaypointControl]] = []

    def initialize_cache(self, *, seed_pixels: Tensor) -> dict[str, Any]:
        assert seed_pixels.shape == (1, 4, 3, 512, 1024)
        assert seed_pixels.dtype is torch.float32
        cache: dict[str, Any] = {"autoregressive_index": 0}
        self.initialized_caches.append(cache)
        return cache

    def generate(
        self,
        autoregressive_index: int,
        cache: dict[str, Any],
        control: WaypointControl,
    ) -> Tensor:
        assert autoregressive_index == cache["autoregressive_index"] + 1
        cache["autoregressive_index"] = autoregressive_index
        self.generate_calls.append((autoregressive_index, control))
        random_value = torch.rand((), generator=self.diffusion_model.rng)
        control_value = sum(control.buttons) / 1000
        return (random_value + control_value).expand(1, 4, 3, 2, 4).clone()

    def finalize(
        self, autoregressive_index: int, cache: dict[str, Any]
    ) -> dict[str, float]:
        assert cache["autoregressive_index"] == autoregressive_index
        return {"diffuse_ms": 1.25, "finalize_ms": 0.25}


def _pipeline(seed: int = 7) -> WaypointInferencePipeline:
    return cast(WaypointInferencePipeline, _FakePipeline(seed))


def _desc(*, width: int = 8, height: int = 4) -> SessionDesc:
    return SessionDesc(
        backpressure_mode=BackpressureMode.BLOCK,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEW,
        output_layout=VideoTensorLayout.tchw,
        video_width=width,
        video_height=height,
    )


def _seed_frames(session_desc: SessionDesc) -> Tensor:
    return torch.zeros(
        4,
        3,
        session_desc.video_height,
        session_desc.video_width,
        dtype=torch.float32,
    )


def _session(
    pipeline: WaypointInferencePipeline,
    *,
    controls: tuple[WaypointControl, ...] | None,
    seed: int = 7,
) -> WaypointSession:
    session_desc = _desc()
    return WaypointSession(
        pipeline=pipeline,
        pipeline_lock=threading.Lock(),
        session_desc=session_desc,
        seed_frames=_seed_frames(session_desc),
        seed=seed,
        controls=controls,
        mouse_sensitivity=1.0,
    )


def _events(*data: UserInputEventData) -> UserInputEvents:
    return UserInputEvents(
        [
            UserInputEvent(timestamp=uint64(index), event_data=event_data)
            for index, event_data in enumerate(data)
        ]
    )


def _empty_events() -> UserInputEvents:
    return UserInputEvents([])


def test_application_description_is_cheap_and_mp4_complete() -> None:
    """Session metadata is available before args, downloads, or model loading."""
    app = WaypointApplication()
    session_desc = app.session_desc()
    assert session_desc.output_layout is VideoTensorLayout.tchw
    assert session_desc.video_width == 1280
    assert session_desc.video_height == 720
    assert session_desc.frames_per_second_for_step == 60
    assert session_desc.backpressure_mode.value == "block"
    assert session_desc.presentation_mode.value == "only_present_new"


def test_application_requires_a_seed_source_and_actions_require_a_file() -> None:
    """Invalid argument combinations fail without constructing model state."""
    app = WaypointApplication()
    with pytest.raises(ValueError, match="seed-image"):
        app.init([])
    with pytest.raises(ValueError, match="actions requires"):
        app.init(["--seed-image", "seed.png", "--actions", "2"])


def test_invalid_session_contract_precedes_image_or_model_work() -> None:
    """Layout and size rejection happen before image decode or checkpoint setup."""
    calls: list[str] = []
    app = WaypointApplication(
        seed_loader=lambda path: calls.append(f"seed:{path}") or torch.empty(0),
        pipeline_factory=lambda seed, device, profile: calls.append("pipeline")
        or _pipeline(seed),
    )
    app.init(["--seed-image", "missing.png", "--seed", "11"])
    with pytest.raises(ValueError, match="tchw"):
        app.create_session(SessionDesc(output_layout=VideoTensorLayout.bcthw))
    with pytest.raises(ValueError, match="1280x720"):
        app.create_session(
            SessionDesc(
                output_layout=VideoTensorLayout.tchw,
                video_width=640,
                video_height=360,
            )
        )
    assert calls == []


def test_application_loads_one_pipeline_for_two_sessions(tmp_path: Path) -> None:
    """One application shares model modules while each session stays distinct."""
    controls_path = tmp_path / "controls.json"
    controls_path.write_text(
        '{"schema_version":1,"actions":[{"buttons":[87]},{}]}',
        encoding="utf-8",
    )
    factory_calls: list[tuple[int, torch.device, bool]] = []
    fake_pipeline = _pipeline(19)
    seed_frames = torch.zeros(1).expand(4, 3, 720, 1280)

    def pipeline_factory(
        seed: int, device: torch.device, profile: bool
    ) -> WaypointInferencePipeline:
        factory_calls.append((seed, device, profile))
        return fake_pipeline

    app = WaypointApplication(
        pipeline_factory=pipeline_factory,
        seed_loader=lambda path: seed_frames,
    )
    app.init(
        [
            "--seed-image",
            "seed.png",
            "--controls-file",
            str(controls_path),
            "--actions",
            "1",
            "--seed",
            "19",
            "--device",
            "cpu",
        ]
    )
    first = app.create_session(app.session_desc())
    second = app.create_session(app.session_desc())
    assert first is not second
    assert first.session_desc == second.session_desc
    assert factory_calls == [(19, torch.device("cpu"), False)]


def test_seed_loader_normalizes_rgb_and_repeats_four_frames(tmp_path: Path) -> None:
    """Pillow input becomes the exact normalized 720p seed display contract."""
    from PIL import Image

    path = tmp_path / "seed.png"
    Image.new("RGB", (2, 1), color=(255, 0, 127)).save(path)
    frames = load_seed_display_frames(path)
    assert frames.shape == (4, 3, 720, 1280)
    assert frames.dtype is torch.float32
    assert torch.equal(frames[0], frames[3])
    assert frames[0, 0, 0, 0].item() == pytest.approx(1.0)
    assert frames[0, 1, 0, 0].item() == pytest.approx(-1.0)


def test_file_session_emits_seed_then_exactly_four_frames_per_action() -> None:
    """Finite mode maps V2 step zero to seed and generated steps to AR 1..N."""
    fake = _FakePipeline()
    controls = (
        WaypointControl(buttons=frozenset({87})),
        WaypointControl(mouse_dx=2.5),
    )
    session = _session(cast(WaypointInferencePipeline, fake), controls=controls)
    session.init()
    loop = cast(WaypointModelLoop, session.model_loop)

    seed = loop.step(0, _empty_events())[0]
    first = loop.step(
        1,
        _events(KeyboardUserInputEventData(key="d", state=KeyboardInputState.PRESSED)),
    )[0]
    second = loop.step(2, _empty_events())[0]

    assert (
        seed.output.shape == first.output.shape == second.output.shape == (4, 3, 4, 8)
    )
    assert [seed.frame_count, first.frame_count, second.frame_count] == [4, 4, 4]
    assert [call[0] for call in fake.generate_calls] == [1, 2]
    assert [call[1] for call in fake.generate_calls] == list(controls)
    assert first.metrics == {
        "diffuse_ms": 1.25,
        "finalize_ms": 0.25,
        "autoregressive_index": 1,
        "generated_frames": 4,
    }
    assert loop.is_finished()


def test_reset_rebuilds_cache_and_replays_first_action_deterministically() -> None:
    """A fixed seed/control reproduces its first generated result after reset."""
    fake = _FakePipeline(seed=31)
    session = _session(
        cast(WaypointInferencePipeline, fake),
        controls=(WaypointControl(buttons=frozenset({32})),),
        seed=31,
    )
    session.init()
    loop = cast(WaypointModelLoop, session.model_loop)
    loop.step(0, _empty_events())
    first = loop.step(1, _empty_events())[0].output.clone()
    first_cache = fake.initialized_caches[-1]

    loop.reset()
    loop.step(0, _empty_events())
    replay = loop.step(1, _empty_events())[0].output
    second_cache = fake.initialized_caches[-1]

    assert torch.equal(first, replay)
    assert first_cache is not second_cache
    loop.close()
    assert loop.state.cache is None


def test_two_sessions_share_modules_but_keep_cache_and_rng_state_isolated() -> None:
    """Interleaved sessions replay the same seeded sequence independently."""
    fake = _FakePipeline(seed=43)
    pipeline = cast(WaypointInferencePipeline, fake)
    shared_lock = threading.Lock()
    controls = (WaypointControl(), WaypointControl())
    first = WaypointSession(
        pipeline=pipeline,
        pipeline_lock=shared_lock,
        session_desc=_desc(),
        seed_frames=_seed_frames(_desc()),
        seed=43,
        controls=controls,
        mouse_sensitivity=1.0,
    )
    second = WaypointSession(
        pipeline=pipeline,
        pipeline_lock=shared_lock,
        session_desc=_desc(),
        seed_frames=_seed_frames(_desc()),
        seed=43,
        controls=controls,
        mouse_sensitivity=1.0,
    )
    first.init()
    second.init()
    first_loop = cast(WaypointModelLoop, first.model_loop)
    second_loop = cast(WaypointModelLoop, second.model_loop)
    first_loop.step(0, _empty_events())
    second_loop.step(0, _empty_events())

    first_one = first_loop.step(1, _empty_events())[0].output
    second_one = second_loop.step(1, _empty_events())[0].output
    first_two = first_loop.step(2, _empty_events())[0].output
    second_two = second_loop.step(2, _empty_events())[0].output

    assert first_loop.state.cache is not second_loop.state.cache
    assert torch.equal(first_one, second_one)
    assert torch.equal(first_two, second_two)


def test_live_session_coalesces_events_for_each_generated_action() -> None:
    """Live mode samples held state and transient motion once per model step."""
    fake = _FakePipeline()
    session = _session(cast(WaypointInferencePipeline, fake), controls=None)
    session.init()
    loop = cast(WaypointModelLoop, session.model_loop)
    loop.step(0, _empty_events())
    loop.step(
        1,
        _events(
            KeyboardUserInputEventData(key="w", state=KeyboardInputState.PRESSED),
            MouseUserInputEventData(action="move", x=0.25, y=0.5),
            MouseUserInputEventData(action="move", x=0.5, y=0.25),
        ),
    )
    loop.step(2, _empty_events())

    first_control = fake.generate_calls[0][1]
    second_control = fake.generate_calls[1][1]
    assert first_control == WaypointControl(
        buttons=frozenset({87}),
        mouse_dx=2.0,
        mouse_dy=-1.0,
    )
    assert second_control == WaypointControl(buttons=frozenset({87}))
    assert not loop.is_finished()


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("a", 65),
        ("W", 87),
        ("9", 57),
        ("ArrowLeft", 0x25),
        ("Shift", 0x10),
        ("Control", 0x11),
        (" ", 0x20),
        ("Enter", 0x0D),
    ],
)
def test_control_adapter_uses_canonical_waypoint_keycodes(
    key: str, expected: int
) -> None:
    """Browser key strings map to the official Biome/Waypoint numeric IDs."""
    adapter = ControlEventAdapter(video_width=100, video_height=50)
    pressed = adapter.consume(
        _events(KeyboardUserInputEventData(key=key, state=KeyboardInputState.PRESSED))
    )
    held = adapter.consume(_empty_events())
    released = adapter.consume(
        _events(
            KeyboardUserInputEventData(
                key=key.swapcase(), state=KeyboardInputState.RELEASED
            )
        )
    )
    assert pressed.buttons == held.buttons == frozenset({expected})
    assert released.buttons == frozenset()


def test_control_adapter_maps_mouse_motion_buttons_and_wheel() -> None:
    """Absolute pointer events become accumulated deltas and canonical button IDs."""
    adapter = ControlEventAdapter(
        video_width=100, video_height=50, mouse_sensitivity=2.0
    )
    first = adapter.consume(
        _events(MouseUserInputEventData(action="move", x=0.25, y=0.5))
    )
    second = adapter.consume(
        _events(
            MouseUserInputEventData(
                action="button", x=0.25, y=0.5, button=1, pressed=True
            ),
            MouseUserInputEventData(action="move", x=0.5, y=0.25),
            MouseUserInputEventData(action="move", x=0.6, y=0.5),
            MouseUserInputEventData(action="wheel", x=0.6, y=0.5, wheel_y=1.0),
            MouseUserInputEventData(action="wheel", x=0.6, y=0.5, wheel_y=-0.25),
        )
    )
    assert first == WaypointControl()
    assert second == WaypointControl(
        buttons=frozenset({0x04}),
        mouse_dx=70.0,
        mouse_dy=0.0,
        scroll_wheel=1,
    )


def test_focus_loss_and_reset_clear_held_state_and_pointer_origin() -> None:
    """Lifecycle edges prevent stuck controls and discard pending pointer history."""
    adapter = ControlEventAdapter(video_width=100, video_height=50)
    adapter.consume(
        _events(
            KeyboardUserInputEventData(key="w", state=KeyboardInputState.PRESSED),
            MouseUserInputEventData(action="button", button=0, pressed=True),
            MouseUserInputEventData(action="move", x=0.2, y=0.3),
        )
    )
    unfocused = adapter.consume(
        _events(
            MouseUserInputEventData(action="move", x=0.4, y=0.5),
            FocusUserInputEventData(focused=False),
        )
    )
    after_focus = adapter.consume(
        _events(MouseUserInputEventData(action="move", x=0.8, y=0.9))
    )
    reset = adapter.consume(
        _events(
            KeyboardUserInputEventData(key="d", state=KeyboardInputState.PRESSED),
            ResetUserInputEventData(),
        )
    )
    assert unfocused == after_focus == reset == WaypointControl()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="writing an MP4 needs ffmpeg on PATH"
)
def test_v2_runtime_writes_seed_plus_four_frames_per_action(
    tmp_path: Path,
) -> None:
    """The generic MP4 window preserves every frame without Waypoint branches."""
    fake = _FakePipeline(seed=53)
    session = _session(
        cast(WaypointInferencePipeline, fake),
        controls=(WaypointControl(), WaypointControl(buttons=frozenset({32}))),
        seed=53,
    )
    path = tmp_path / "waypoint.mp4"

    run_session(session, Mp4ClientWindow(path), steps=None)

    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert len(raw) // (8 * 4 * 3) == 12
    assert path.stat().st_size > 0
