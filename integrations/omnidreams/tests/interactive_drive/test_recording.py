# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import omnidreams.interactive_drive.recording as recording_module
from omnidreams.interactive_drive._pipeline_fakes import minimal_scene
from omnidreams.interactive_drive.recording import (
    InteractiveDriveRecorder,
    RecordingConfig,
    normalize_recording_hotkey,
    recording_hotkey_collides_with_controls,
    recording_hotkey_matches,
    slangpy_key_name_candidates,
)
from omnidreams.interactive_drive.types import PresentedFrame
from PIL import Image


def test_hotkey_normalization_and_matching() -> None:
    assert normalize_recording_hotkey("R") == "r"
    assert normalize_recording_hotkey("ArrowUp") == "up"
    assert recording_hotkey_matches("f9", "F9") is True
    assert recording_hotkey_collides_with_controls("R") is True
    assert recording_hotkey_collides_with_controls("F9") is False
    assert slangpy_key_name_candidates("1") == ("key1", "digit1", "num_1")


def test_recorder_writes_bundle_on_stop(tmp_path, monkeypatch) -> None:
    writes: list[tuple[Path, int, tuple[int, ...]]] = []

    def fake_write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
        writes.append((path, fps, np.stack(frames, axis=0).shape))
        path.write_bytes(b"fake mp4")

    monkeypatch.setattr(recording_module, "_write_video", fake_write_video)
    scene = minimal_scene()
    scene = replace(
        scene,
        scene_id="recording scene",
        initial_rgb=np.full((4, 4, 3), 3, dtype=np.uint8),
        prompt="drive through a bright test scene",
    )
    recorder = InteractiveDriveRecorder(
        RecordingConfig(enabled=True, output_dir=tmp_path, hotkey="r"),
        scene=scene,
        fps=30,
    )

    recorder.start()
    recorder.record_frame(
        PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=np.full((4, 4, 3), 7, dtype=np.uint8),
            depth_host_f32=None,
            model_rgb_host_uint8=np.full((4, 4, 3), 11, dtype=np.uint8),
        )
    )
    output_dir = recorder.stop(reason="hotkey")

    assert output_dir is not None
    assert (output_dir / "first_frame.png").exists()
    assert np.array_equal(
        np.asarray(Image.open(output_dir / "first_frame.png")),
        np.full((4, 4, 3), 11, dtype=np.uint8),
    )
    assert (output_dir / "prompt.txt").read_text(encoding="utf-8") == scene.prompt
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "hdmap.mp4").read_bytes() == b"fake mp4"
    assert (output_dir / "inferred.mp4").read_bytes() == b"fake mp4"
    assert writes == [
        (output_dir / "hdmap.mp4", 30, (1, 4, 4, 3)),
        (output_dir / "inferred.mp4", 30, (1, 4, 4, 3)),
    ]


def test_recorder_video_write_failure_is_nonfatal(tmp_path, monkeypatch) -> None:
    def fail_write_video(
        frames: list[np.ndarray],
        path: Path,
        fps: int,
    ) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(recording_module, "_write_video", fail_write_video)
    scene = replace(minimal_scene(), scene_id="recording scene")
    recorder = InteractiveDriveRecorder(
        RecordingConfig(enabled=True, output_dir=tmp_path, hotkey="F9"),
        scene=scene,
        fps=30,
    )

    recorder.start()
    recorder.record_frame(
        PresentedFrame(
            timestamp_us=0,
            rgb_host_uint8=np.full((4, 4, 3), 7, dtype=np.uint8),
            depth_host_f32=None,
            model_rgb_host_uint8=np.full((4, 4, 3), 11, dtype=np.uint8),
        )
    )

    assert recorder.stop(reason="hotkey") is None
    assert recorder.is_recording is False
    assert recorder.closed_paths == ()


def test_recorder_start_failure_is_nonfatal(tmp_path) -> None:
    output_root = tmp_path / "not-a-directory"
    output_root.write_text("plain file", encoding="utf-8")
    recorder = InteractiveDriveRecorder(
        RecordingConfig(enabled=True, output_dir=output_root, hotkey="F9"),
        scene=minimal_scene(),
        fps=30,
    )

    recorder.start()

    assert recorder.is_recording is False
    assert recorder.closed_paths == ()


def test_recorder_caps_frame_buffers(tmp_path, monkeypatch) -> None:
    writes: list[tuple[Path, list[int]]] = []

    def fake_write_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
        del fps
        writes.append((path, [int(frame[0, 0, 0]) for frame in frames]))
        path.write_bytes(b"fake mp4")

    monkeypatch.setattr(recording_module, "_write_video", fake_write_video)
    scene = replace(minimal_scene(), scene_id="recording scene")
    recorder = InteractiveDriveRecorder(
        RecordingConfig(
            enabled=True,
            output_dir=tmp_path,
            hotkey="F9",
            max_buffer_frames=2,
        ),
        scene=scene,
        fps=30,
    )

    recorder.start()
    for value in (1, 2, 3):
        recorder.record_frame(
            PresentedFrame(
                timestamp_us=value,
                rgb_host_uint8=np.full((4, 4, 3), value, dtype=np.uint8),
                depth_host_f32=None,
                model_rgb_host_uint8=np.full((4, 4, 3), value + 10, dtype=np.uint8),
            )
        )
    output_dir = recorder.stop(reason="hotkey")

    assert output_dir is not None
    assert writes == [
        (output_dir / "hdmap.mp4", [2, 3]),
        (output_dir / "inferred.mp4", [12, 13]),
    ]
    assert np.array_equal(
        np.asarray(Image.open(output_dir / "first_frame.png")),
        np.full((4, 4, 3), 12, dtype=np.uint8),
    )
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["hdmap_frames"] == 2
    assert metadata["inferred_frames"] == 2
    assert metadata["dropped_hdmap_frames"] == 1
    assert metadata["dropped_inferred_frames"] == 1
    assert metadata["max_buffer_frames"] == 2


def test_recorder_collision_suffix_starts_at_one(tmp_path, monkeypatch) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls, tz):
            return datetime(2026, 6, 10, 1, 2, 3, tzinfo=tz)

    monkeypatch.setattr(recording_module, "datetime", FixedDateTime)
    scene = replace(minimal_scene(), scene_id="recording scene")
    recorder = InteractiveDriveRecorder(
        RecordingConfig(enabled=True, output_dir=tmp_path, hotkey="F9"),
        scene=scene,
        fps=30,
    )
    base_name = "20260610-010203-recording-scene-001"
    (tmp_path / base_name).mkdir()

    recorder.start()

    assert recorder.is_recording is True
    assert (tmp_path / f"{base_name}-01").is_dir()
