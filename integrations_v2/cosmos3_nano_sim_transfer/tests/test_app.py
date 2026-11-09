# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``control2v-cosmos3-nano-sim-transfer`` application: arguments, session, and a run to MP4.

The end-to-end run drives the application through the v2 runtime into an MP4
file, as ``flashdreams-run-v2 --output-path`` does, on the tiny stack of
``conftest.py``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_transfer.apps.control2v.adapter import (
    SLUG,
    Control2VApplication,
    Control2VArgs,
    create_app,
    parse_args,
    read_control_clip,
    timing_summary,
)
from cosmos3_nano_sim_transfer.config import HistoryProfile
from cosmos3_nano_sim_transfer.impl.prompt import HEIGHT, WIDTH
from cosmos3_nano_sim_transfer.impl.stream import TransferStream

from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.mp4_client_window import Mp4ClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="writing and reading MP4 files needs ffmpeg"
)

MakeStream = Callable[..., TransferStream]


def _write_control(path: Path, frames: int, fps: int = 10) -> np.ndarray:
    """A segmentation-like control clip (flat color blocks), 96x160."""
    import mediapy

    video = np.zeros((frames, 96, 160, 3), np.uint8)
    for t in range(frames):
        video[t, :, : 40 + 8 * t] = (200, 40, 40)
        video[t, 48:, :] = (40, 40, 200)
    mediapy.write_video(str(path), video, fps=fps, qp=0)
    return video


def test_arguments_and_defaults(tmp_path: Path) -> None:
    args = parse_args(["--control", "seg.mp4", "--prompt", "A street."])
    assert args == Control2VArgs(control=Path("seg.mp4"), prompt="A street.")
    assert not args.emphasize_control_in_prompt
    assert args.prompt_aspect_ratio is None
    assert (args.hint, args.seed, args.history, args.decoder) == (
        "seg",
        0,
        "long-rollout",
        "wan2.2",
    )
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("\n  A city at noon.  \nsecond line\n")
    args = parse_args(
        [
            "--control", "depth.mp4", "--hint", "depth", "--prompt-file", str(prompt_file),
            "--seed", "7", "--history", "12/2", "--fps", "20", "--max-frames", "41",
            "--prompt-frames", "401", "--decoder", "taehv-super",
            "--timing-path", "t.json", "--emphasize-control-in-prompt",
            "--prompt-aspect-ratio", "16,9",
        ]
    )  # fmt: skip
    assert args.prompt == "A city at noon."
    assert (args.hint, args.seed, args.history, args.fps) == ("depth", 7, "12/2", 20.0)
    assert (args.max_frames, args.prompt_frames, args.decoder) == (
        41,
        401,
        "taehv-super",
    )
    assert args.timing_path == Path("t.json")
    assert args.emphasize_control_in_prompt
    assert args.prompt_aspect_ratio == "16,9"
    for preset in ("export", "long-rollout"):
        assert (
            parse_args(["--control", "c", "--prompt", "p", "--history", preset]).history
            == preset
        )


def test_batch_forwards_structured_prompt_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos3_nano_sim_transfer.apps.control2v import adapter

    seen = []
    sentinel = object()

    def load(*args, **kwargs):
        seen.append(kwargs)
        return sentinel

    monkeypatch.setattr(adapter.TransferStream, "load", load)
    args = parse_args(
        [
            "--control",
            "seg.mp4",
            "--prompt",
            '{"scene":"A street."}',
            "--prompt-aspect-ratio",
            "16,9",
            "--emphasize-control-in-prompt",
        ]
    )
    assert adapter.load_stream(args, 30.0) is sentinel
    assert seen[0]["prompt_aspect_ratio"] == "16,9"
    assert seen[0]["emphasize_control_in_prompt"] is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--prompt", "p"],  # no control
        ["--control", "c"],  # no prompt
        ["--control", "c", "--prompt", "p", "--prompt-file", "f"],
        ["--control", "c", "--prompt", "p", "--hint", "normals"],
        ["--control", "c", "--prompt", "p", "--history", "30-3"],
        ["--control", "c", "--prompt", "p", "--history", "3/3"],
        ["--control", "c", "--prompt", "p", "--seed", "-1"],
        ["--control", "c", "--prompt", "p", "--max-frames", "0"],
        ["--control", "c", "--prompt", "p", "--fps", "0"],
        ["--control", "c", "--prompt", "p", "--decoder", "vae"],
    ],
    ids=[
        "no-control", "no-prompt", "two-prompts", "hint", "history-syntax",
        "history-window", "seed", "max-frames", "fps", "decoder",
    ],
)  # fmt: skip
def test_invalid_arguments_are_refused(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_history_specs_resolve_against_the_export(tiny_model) -> None:
    assert tiny_model.history("export") == HistoryProfile(51, 1)
    assert tiny_model.history("long-rollout") == HistoryProfile(30, 3)
    assert tiny_model.history("12/2") == HistoryProfile(12, 2)
    assert str(HistoryProfile(30, 3)) == "30/3"


def test_timing_medians_from_the_steady_state() -> None:
    timings = [{"latent": k, "step_ms": float(k), "model_ms": 10.0} for k in range(6)]
    assert timing_summary(timings, 3) == {"model_ms": 10.0, "step_ms": 4.0}
    assert timing_summary(timings[:2], 3) == {"model_ms": 10.0, "step_ms": 0.5}


@needs_ffmpeg
def test_control_clip_is_read_whole(tmp_path: Path) -> None:
    path = tmp_path / "seg.mp4"
    _write_control(path, 11)
    whole = read_control_clip(path)
    assert whole.shape == (11, 96, 160, 3)
    np.testing.assert_array_equal(read_control_clip(path, max_frames=6), whole[:6])
    assert read_control_clip(path, max_frames=1).shape[0] == 1
    with pytest.raises(FileNotFoundError):
        read_control_clip(tmp_path / "missing.mp4")


def test_session_description_and_guards() -> None:
    app = create_app()
    assert isinstance(app, Control2VApplication)
    desc = app.session_desc()
    assert desc.output_layout is VideoTensorLayout.tchw
    assert (desc.video_width, desc.video_height) == (WIDTH, HEIGHT)
    assert desc.presentation_mode is PresentationMode.ON_DEMAND
    assert desc.backpressure_mode is BackpressureMode.BLOCK
    with pytest.raises(RuntimeError):
        app.create_session(desc)


def test_registered_under_its_slug() -> None:
    found = [
        e for e in entry_points(group="flashdreams.applications_v2") if e.name == SLUG
    ]
    assert [e.value for e in found] == [
        "cosmos3_nano_sim_transfer.apps.control2v.adapter:create_app"
    ]


@needs_ffmpeg
def test_runs_a_control_clip_to_mp4(
    tmp_path: Path, make_stream: MakeStream, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mediapy

    control = tmp_path / "seg.mp4"
    clip = _write_control(control, 13, fps=10)
    loaded: list[tuple[Control2VArgs, float]] = []
    streams: list[TransferStream] = []
    controls: list[np.ndarray] = []

    def loader(args: Control2VArgs, fps: float) -> TransferStream:
        loaded.append((args, fps))
        stream = make_stream(hint=args.hint, history=args.history, fps=fps)
        step_tensor = stream.step_tensor

        def record(control: np.ndarray, *, last: bool = False) -> torch.Tensor:
            controls.append(np.array(control))
            return step_tensor(control, last=last)

        monkeypatch.setattr(stream, "step_tensor", record)
        streams.append(stream)
        return stream

    app = Control2VApplication(stream_loader=loader)
    output, timing = tmp_path / "out.mp4", tmp_path / "timing.json"
    argv = [
        "--control", str(control), "--prompt", "A street.", "--seed", "5",
        "--history", "4/1", "--max-frames", "11", "--timing-path", str(timing),
    ]  # fmt: skip
    ApplicationRunner(app, Mp4ClientWindow(output)).run(app.session_desc(), argv)

    assert loaded[0][1] == pytest.approx(10.0)  # the clip's own frame rate
    # 11 frames: 1 + 4 + 4 + 2, the last latent completed by padding (as
    # cosmos-framework pads it), the prompt stating 11 frames, 11 frames out.
    assert streams[0].prompt_frames == 11
    assert [len(c) for c in controls] == [1, 4, 4, 4]
    read_back = mediapy.read_video(str(control))
    np.testing.assert_array_equal(controls[3], read_back[[9, 10, 10, 9]])
    assert clip.shape[0] == 13
    video = mediapy.read_video(str(output))
    assert video.shape == (11, HEIGHT, WIDTH, 3)
    assert video.metadata is not None and video.metadata.fps == pytest.approx(10.0)

    report = json.loads(timing.read_text())
    assert report["latents"] == 4 and report["frames"] == 11
    assert report["history"] == "4/1"
    assert report["args"]["seed"] == 5 and report["args"]["control"] == str(control)
    assert [r["latent"] for r in report["per_latent_ms"]] == [0, 1, 2, 3]
    assert {"encode_ms", "model_ms", "decode_ms", "step_ms"} <= set(report["median_ms"])


def test_other_layouts_and_sizes_are_refused(
    tmp_path: Path, make_stream: MakeStream
) -> None:
    from dataclasses import replace

    app = Control2VApplication(stream_loader=lambda args, fps: make_stream())
    app.init(
        ["--control", str(_control_file(tmp_path)), "--prompt", "p", "--fps", "30"]
    )
    desc = app.session_desc()
    assert desc.metadata["history"] == "30/3" and desc.metadata["hint"] == "seg"
    with pytest.raises(ValueError):
        app.create_session(replace(desc, output_layout=VideoTensorLayout.bcthw))
    with pytest.raises(ValueError):
        app.create_session(replace(desc, video_width=1280, video_height=720))
    session = app.create_session(desc)
    assert session.session_desc.frames_per_second_for_step == 30
    app.close()


def _control_file(tmp_path: Path) -> Path:
    path = tmp_path / "seg.mp4"
    if shutil.which("ffmpeg") is None:
        pytest.skip("writing an MP4 needs ffmpeg")
    _write_control(path, 5)
    return path
