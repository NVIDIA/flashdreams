# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the way drawn pixels reach an mp4.

The writer takes ``[-1, 1]`` and truncates on the way back to bytes, so the
scaling into it has half a step to spare or every shade lands one darker than
it was drawn. That reads later as the drawing being dim rather than as the
conversion being wrong, and the drawing is the thing under examination.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.controls import (
    available_pixel_frames,
    writable,
    write_control_clip,
)

from flashdreams.infra.runner_io import video_tensor_to_uint8

pytestmark = pytest.mark.ci_cpu


def _clip(path, frames: int):
    """A tiny mp4 of ``frames`` frames, written where a control track would be."""
    return write_control_clip(
        torch.zeros(frames, 3, 16, 16, dtype=torch.uint8), path, fps=30.0
    )


def test_every_shade_comes_back_the_shade_it_went_in_as() -> None:
    shades = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16)
    drawn = shades.expand(1, 3, 16, 16).contiguous()

    written = video_tensor_to_uint8(writable(drawn), layout="tchw")

    assert (written == drawn.permute(0, 2, 3, 1).numpy()).all()


def test_pixels_that_are_not_bytes_are_refused() -> None:
    """Anything already scaled would be scaled twice, and quietly."""
    with pytest.raises(ValueError, match="uint8"):
        writable(torch.zeros(1, 3, 4, 4))


def test_a_length_the_encoder_can_take_is_what_comes_back(tmp_path) -> None:
    """21 frames on disk is 21 the encoder cannot have: it wants ``1 + 4n``."""
    track = _clip(tmp_path / "front.mp4", 21)

    assert available_pixel_frames([track]) == 21
    assert available_pixel_frames([_clip(tmp_path / "odd.mp4", 23)]) == 21


def test_the_shortest_track_decides(tmp_path) -> None:
    """A rig is read as one window, so the camera that ends first ends the run."""
    tracks = [_clip(tmp_path / "front.mp4", 33), _clip(tmp_path / "rear.mp4", 17)]

    assert available_pixel_frames(tracks) == 17


def test_the_start_frame_comes_off_the_front(tmp_path) -> None:
    track = _clip(tmp_path / "front.mp4", 33)

    assert available_pixel_frames([track], start_frame=8) == 25


def test_a_start_past_the_end_is_refused(tmp_path) -> None:
    track = _clip(tmp_path / "front.mp4", 9)

    with pytest.raises(ValueError, match="nothing at or after frame 9"):
        available_pixel_frames([track], start_frame=9)


def test_a_track_that_is_not_there_is_refused(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="not local files"):
        available_pixel_frames([tmp_path / "missing.mp4"])


def test_a_looped_track_repeats_from_its_start(monkeypatch, tmp_path) -> None:
    """--loop-controls tiles what the clip holds, in order, out to the run."""
    from cosmos3_nano_sim_auto.impl.conditioning import controls

    import flashdreams.infra.runner_io as runner_io

    source = torch.arange(5.0).view(5, 1, 1, 1)
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    monkeypatch.setattr(
        controls, "available_pixel_frames", lambda paths, start_frame=0: 5
    )
    monkeypatch.setattr(
        runner_io,
        "load_video_tensor",
        lambda path, *, frame_count, **_: source[:frame_count],
    )
    [looped] = controls.loaded_control_clips(
        [clip], pixel_frames=12, pixel_height=1, pixel_width=1, loop=True
    )
    assert looped.flatten().tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]


def test_looped_cameras_share_one_seam(monkeypatch, tmp_path) -> None:
    """Tracks of different lengths loop at the shortest, or the rig drifts apart."""
    from cosmos3_nano_sim_auto.impl.conditioning import controls

    import flashdreams.infra.runner_io as runner_io

    lengths = {"short.mp4": 5, "long.mp4": 9}
    paths = [tmp_path / name for name in lengths]
    for path in paths:
        path.write_bytes(b"")
    monkeypatch.setattr(
        controls,
        "available_pixel_frames",
        lambda paths, start_frame=0: min(lengths[p.name] for p in paths),
    )
    monkeypatch.setattr(
        runner_io,
        "load_video_tensor",
        lambda path, *, frame_count, **_: torch.arange(
            float(min(frame_count, lengths[path.name]))
        ).view(-1, 1, 1, 1),
    )
    load: dict[str, Any] = dict(
        pixel_frames=12, pixel_height=1, pixel_width=1, loop=True
    )
    short, long = controls.loaded_control_clips(paths, **load)
    assert torch.equal(short, long)
    # A rank reading only the long camera still loops at the rig's length.
    [alone] = controls.loaded_control_clips(paths[1:], loop_frames=5, **load)
    assert torch.equal(alone, short)


def test_every_missing_file_is_named_at_once(tmp_path) -> None:
    from cosmos3_nano_sim_auto.impl.conditioning.controls import require_local_files

    present = tmp_path / "here.mp4"
    present.write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="a.mp4.*b.mp4"):
        require_local_files([tmp_path / "a.mp4", present, tmp_path / "b.mp4"])
