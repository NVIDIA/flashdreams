# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the control seam.

The provider's whole job is to hand back the right patches in the right order,
and both mistakes are quiet: a wrong range conditions a chunk on the wrong
stretch of road, and a wrong order puts one camera's map on another camera.
Neither raises, and both look like the model having a bad day.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
    ControlProvider,
    EncodedControls,
    RenderedControls,
    pixel_span_for_latents,
    rendered_clips,
)
from cosmos3_nano_sim_auto.impl.conditioning.control_worker import ControlWorker
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.inputs import control_tokens

pytestmark = pytest.mark.ci_cpu

WIDTH = 8
CLIP = ClipGeometry(
    num_views=3,
    frames_per_view=5,
    patch_h=1,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=0,
)


def track(geometry: ClipGeometry = CLIP) -> torch.Tensor:
    """A control track whose every token says which view and frame it came from.

    So a misplaced slice is an assertion about numbers rather than a picture
    someone has to look at.
    """
    assert geometry.frames_per_view is not None
    tokens = torch.zeros(
        geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens, WIDTH
    )
    index = 0
    for view in range(geometry.num_views):
        for frame in range(geometry.frames_per_view):
            for _ in range(geometry.spatial_tokens):
                tokens[index, 0] = view
                tokens[index, 1] = frame
                index += 1
    return tokens


def views_and_frames(patches: torch.Tensor) -> list[tuple[int, int]]:
    return [(int(row[0]), int(row[1])) for row in patches]


def test_a_range_is_taken_from_inside_every_view_not_off_the_front() -> None:
    """The track is view-outer, so frames 1-3 are a stride, not a prefix.

    Slicing off the front would hand back the first camera's whole clip and
    call it three frames of the rig -- which has the right token count, so
    nothing downstream would notice.
    """
    provider = EncodedControls(track(), CLIP)

    patches = provider.tokens(1, 3)

    assert views_and_frames(patches) == [
        (view, frame) for view in range(3) for frame in (1, 2) for _ in range(2)
    ]


def test_the_whole_track_comes_back_as_it_went_in() -> None:
    """The prefill's own call when nothing is streamed, so it must not copy or reorder."""
    tokens = track()
    provider = EncodedControls(tokens, CLIP)

    assert CLIP.frames_per_view is not None
    assert provider.tokens(0, CLIP.frames_per_view) is tokens


def test_consecutive_ranges_rebuild_the_track_they_came_from() -> None:
    """Streaming a chunk at a time has to add up to encoding the clip at once."""
    provider = EncodedControls(track(), CLIP)

    assert CLIP.frames_per_view is not None
    pieces = [
        provider.tokens(start, start + 1) for start in range(CLIP.frames_per_view)
    ]

    assert views_and_frames(torch.cat(pieces)) != views_and_frames(
        provider.tokens(0, CLIP.frames_per_view)
    )
    # Not equal as concatenated: a chunk is view-outer within itself, so the
    # pieces interleave the views rather than following one camera through.
    # Frame-by-frame is what agrees.
    for start in range(CLIP.frames_per_view):
        assert views_and_frames(pieces[start]) == [
            (view, start) for view in range(3) for _ in range(2)
        ]


def test_a_range_past_the_track_is_refused() -> None:
    """The provider knows how much it has; the rollout does not."""
    provider = EncodedControls(track(), CLIP)

    with pytest.raises(ValueError, match="run outside the clip"):
        provider.tokens(3, 6)


def test_an_empty_range_is_refused() -> None:
    provider = EncodedControls(track(), CLIP)

    with pytest.raises(ValueError, match="run outside the clip"):
        provider.tokens(2, 2)


def test_a_track_that_does_not_match_the_geometry_is_refused_on_the_way_in() -> None:
    """Caught here rather than as a shape error deep in an attention call."""
    with pytest.raises(ValueError, match="control tokens"):
        EncodedControls(track()[:-2], CLIP)


def test_the_resident_track_satisfies_the_protocol() -> None:
    """What a ludus provider will be checked against, so it has to fit today's."""
    assert isinstance(EncodedControls(track(), CLIP), ControlProvider)


# --- rendered on demand ----------------------------------------------------

RIG = ClipGeometry(
    num_views=2,
    frames_per_view=5,
    patch_h=2,
    patch_w=2,
    frames_per_chunk=4,
    condition_frames=0,
)
"""Two cameras and 17 pixel frames, at a size the arithmetic is legible in.

``fake_encode`` does not compress space, so a 4x4 frame stays a 4x4 latent and
patch size 2 makes the 2x2 grid of four spatial tokens this names.
"""

PIXEL_H, PIXEL_W = 4, 4


def latent_count(pixel_frames: int) -> int:
    return 1 + (pixel_frames - 1) // 4


def make_fake_encode_stream() -> Callable[[torch.Tensor], torch.Tensor]:
    """A stand-in for the VAE: causal in time, 4x compressed, and it looks back.

    Latent frame 0 is pixel frame 0; each later one averages its four pixel
    frames *together with the frame before them*, so its receptive field reaches
    past the group it covers. Space is left alone, which the geometry above
    accounts for.

    That backwards reach is the whole point of the stand-in. A stream that keeps
    its history encodes a span exactly as it would have encoded the whole track,
    and one that starts fresh each call cannot -- so the comparison below can
    tell the two apart, which it could not if the fake's receptive field stopped
    at its own group.
    """
    seen: list[torch.Tensor] = []

    def encode(clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 4 or clip.shape[1] != 3:
            raise ValueError(f"expected a [T, 3, H, W] clip, got {tuple(clip.shape)}.")
        first_new = len(seen)
        seen.extend(clip)
        timeline = torch.stack(seen)
        latents = []
        for frame in range(latent_count(len(seen))):
            start = 0 if frame == 0 else 4 * frame - 3
            end = 1 if frame == 0 else 4 * frame + 1
            if end <= first_new:
                continue  # handed back on an earlier call
            latents.append(timeline[max(0, start - 1) : end].mean(dim=0, keepdim=True))
        return torch.cat(latents).permute(1, 0, 2, 3)  # [C, T_lat, H, W]

    return encode


def rig_render(start: int, end: int) -> torch.Tensor:
    """Frames whose every pixel is the absolute pixel-frame index, plus the view.

    So a span rendered for the wrong frames, or views taken in the wrong order,
    comes out as the wrong numbers rather than as a picture.
    """
    frames = []
    for view in range(RIG.num_views):
        for frame in range(start, end):
            frames.append(torch.full((PIXEL_H, PIXEL_W, 4), float(frame + 100 * view)))
    return torch.stack(frames).clamp(0, 255).to(torch.uint8)


def rendered() -> RenderedControls:
    return RenderedControls(
        rig_render,
        [make_fake_encode_stream() for _ in range(RIG.num_views)],
        geometry=RIG,
        latent_patch_size=2,
    )


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 1, (0, 1)),  # frame 0 stands alone
        (0, 5, (0, 17)),
        (1, 5, (1, 17)),
        (1, 2, (1, 5)),
        (3, 5, (9, 17)),
    ],
)
def test_a_latent_range_names_the_pixel_frames_it_came_from(
    start, end, expected
) -> None:
    """Off by four here conditions a chunk on the road it just drove past."""
    assert pixel_span_for_latents(start, end) == expected


def test_a_span_away_from_the_start_covers_a_multiple_of_four_pixel_frames() -> None:
    """What the encoder asserts after its opening call, so the spans have to hold it."""
    first, last = pixel_span_for_latents(2, 4)

    assert (last - first) % 4 == 0
    assert (last - first - 1) % 4 != 0


def test_consecutive_spans_tile_the_pixel_timeline() -> None:
    """A gap or an overlap would feed the streaming encoder the wrong footage."""
    spans = [pixel_span_for_latents(start, start + 1) for start in range(6)]

    assert spans[0] == (0, 1)
    for earlier, later in zip(spans, spans[1:], strict=False):
        assert earlier[1] == later[0]


def test_rendering_a_chunk_gives_the_patches_the_rollout_expects() -> None:
    """Right count, right width, so the token stream lines up with the positions."""
    provider = rendered()
    provider.tokens(0, 1)

    patches = provider.tokens(1, 5)

    frames = 4
    channels = 3  # the fake encoder keeps RGB
    assert patches.shape == (
        RIG.num_views * frames * RIG.spatial_tokens,
        channels * 2 * 2,  # one 2x2 patch of each channel
    )


def test_only_the_frames_asked_for_come_back() -> None:
    """Each call hands back its own span and none of the history behind it."""
    provider = rendered()

    for start, end in [(0, 1), (1, 5), (5, 9)]:
        patches = provider.tokens(start, end)
        assert patches.shape[0] == RIG.num_views * (end - start) * RIG.spatial_tokens


def test_a_run_can_carry_on_past_the_clip_it_was_cut_to() -> None:
    """The reason to render rather than read: there is no track to run out of."""
    provider = rendered()
    provider.tokens(0, 1)

    assert RIG.frames_per_view is not None
    assert provider.tokens(1, RIG.frames_per_view + 8).shape[0] == (
        RIG.num_views * (RIG.frames_per_view + 7) * RIG.spatial_tokens
    )


def test_a_chunk_at_a_time_matches_the_whole_track_exactly() -> None:
    """The claim the streaming path rests on, and the reason the streams exist.

    The stand-in encoder looks one frame further back than the group it covers,
    so this only holds if the provider's streams really carry their temporal
    state between calls; encoding each span from scratch would miss that frame
    at every seam. Same reason it holds for the real Wan encoder, whose
    ``WanVAECache`` is the state in question and whose reach is longer still.
    """
    resident = EncodedControls(whole_track(), RIG)
    provider = rendered()

    for start, end in [(0, 1), (1, 5)]:
        torch.testing.assert_close(
            provider.tokens(start, end), resident.tokens(start, end)
        )


def test_a_stream_that_forgets_its_history_does_not_match() -> None:
    """That the test above can fail, so it is worth something.

    A fresh stream per call is what ``make_control_encode_fn`` would give this
    class, and the mistake it is easiest to make. Frame 0 still agrees -- there
    is no history to lose yet -- so a comparison that stopped at the opening
    window would pass either way.
    """
    resident = EncodedControls(whole_track(), RIG)
    forgetful = RenderedControls(
        rig_render,
        [_ForgetfulStream() for _ in range(RIG.num_views)],
        geometry=RIG,
        latent_patch_size=2,
    )

    torch.testing.assert_close(forgetful.tokens(0, 1), resident.tokens(0, 1))
    assert not torch.allclose(forgetful.tokens(1, 5), resident.tokens(1, 5))


class _ForgetfulStream:
    """Encodes each span from an empty state, as a single-shot encoder would."""

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return make_fake_encode_stream()(clip)


def whole_track() -> torch.Tensor:
    """The track encoded in one go, built without going through the provider.

    A second, independent path to the same answer -- otherwise the comparison
    above would be the provider agreeing with itself.
    """
    frames = rig_render(0, 17)
    clips = (
        frames[..., :3]
        .reshape(RIG.num_views, 17, PIXEL_H, PIXEL_W, 3)
        .permute(0, 1, 4, 2, 3)
        .to(torch.float32)
        .div(127.5)
        .sub(1.0)
    )
    latents = torch.stack(
        [make_fake_encode_stream()(clips[view]) for view in range(RIG.num_views)]
    )
    return control_tokens(latents, latent_patch_size=2)


def with_render(render) -> RenderedControls:
    return RenderedControls(
        render,
        [make_fake_encode_stream() for _ in range(RIG.num_views)],
        geometry=RIG,
        latent_patch_size=2,
    )


def test_a_render_that_hands_back_the_wrong_frame_count_is_refused() -> None:
    """Otherwise the reshape would silently reinterpret views as frames."""

    def short(start: int, end: int) -> torch.Tensor:
        del start, end
        return torch.zeros(3, PIXEL_H, PIXEL_W, 4, dtype=torch.uint8)

    with pytest.raises(ValueError, match="rendered frames"):
        with_render(short).tokens(0, 5)


def test_a_render_without_colour_is_refused() -> None:
    def grey(start: int, end: int) -> torch.Tensor:
        return torch.zeros(
            RIG.num_views * (end - start), PIXEL_H, PIXEL_W, 1, dtype=torch.uint8
        )

    with pytest.raises(ValueError, match="3 channels"):
        with_render(grey).tokens(0, 5)


def test_a_range_that_skips_ahead_is_refused() -> None:
    """The streams have consumed the footage behind it and cannot be wound on.

    Left to itself this is silent: the range would come back the right shape,
    encoded against whatever the streams happened to hold.
    """
    provider = rendered()
    provider.tokens(0, 1)

    with pytest.raises(ValueError, match="carry on from 1"):
        provider.tokens(2, 5)


def test_a_range_already_handed_out_is_refused() -> None:
    """A provider that renders cannot be rewound, unlike one that slices."""
    provider = rendered()
    provider.tokens(0, 1)
    provider.tokens(1, 5)

    with pytest.raises(ValueError, match="cannot be rewound"):
        provider.tokens(1, 5)


def test_the_provider_says_where_the_next_range_starts() -> None:
    """So a driver picking a rollout back up does not have to guess."""
    provider = rendered()
    assert provider.next_frame == 0

    provider.tokens(0, 1)
    provider.tokens(1, 5)

    assert provider.next_frame == 5


def test_a_stream_missing_for_a_camera_is_refused_on_the_way_in() -> None:
    """Caught here rather than as an IndexError partway through a chunk."""
    with pytest.raises(ValueError, match="one encode stream per camera"):
        RenderedControls(
            rig_render,
            [make_fake_encode_stream()],
            geometry=RIG,
            latent_patch_size=2,
        )


def test_the_rendered_provider_satisfies_the_protocol() -> None:
    """The seam's point: ludus arrives without the rollout learning anything."""
    assert isinstance(rendered(), ControlProvider)


# --- drawn ahead, in a worker ---------------------------------------------


class _FakePipe:
    """The worker's end of the pipe, in-process: draws with the fakes above on ``recv``."""

    def __init__(self) -> None:
        self.streams = [make_fake_encode_stream() for _ in range(RIG.num_views)]
        self.asked: list[tuple[int, int]] = []

    def send(self, message: tuple[int, int]) -> None:
        self.asked.append(message)

    def recv(self) -> tuple[list[torch.Tensor], int]:
        start, end = self.asked[-1]
        first, last = pixel_span_for_latents(start, end)
        clips = rendered_clips(
            rig_render(first, last), pixel_frames=last - first, count=RIG.num_views
        )
        return [
            stream(clip) for stream, clip in zip(self.streams, clips, strict=True)
        ], last - first


def rendered_ahead() -> tuple[RenderedControls, _FakePipe]:
    pipe = _FakePipe()
    worker = ControlWorker(cast(Any, pipe), device=torch.device("cpu"))
    provider = RenderedControls(
        rig_render,
        [make_fake_encode_stream() for _ in range(RIG.num_views)],
        geometry=RIG,
        latent_patch_size=2,
        worker=worker,
    )
    return provider, pipe


def test_a_prefetched_chunk_is_the_chunk_drawn_on_demand() -> None:
    """Same numbers whether the worker drew it early or the provider drew it now.

    Asking twice for the range in flight is a no-op, so the rollout can say it
    every chunk; the opening window needs no request at all.
    """
    ahead, pipe = rendered_ahead()
    on_demand = rendered()

    torch.testing.assert_close(ahead.tokens(0, 1), on_demand.tokens(0, 1))
    ahead.prefetch(1, 3)
    ahead.prefetch(1, 3)
    torch.testing.assert_close(ahead.tokens(1, 3), on_demand.tokens(1, 3))
    ahead.prefetch(3, 5)
    torch.testing.assert_close(ahead.tokens(3, 5), on_demand.tokens(3, 5))
    assert pipe.asked == [(0, 1), (1, 3), (3, 5)]


def test_asking_for_a_range_other_than_the_prefetched_one_is_refused() -> None:
    """The worker's encoder has consumed the prefetched span; there is no serving another."""
    ahead, _pipe = rendered_ahead()
    ahead.tokens(0, 1)

    ahead.prefetch(1, 5)
    with pytest.raises(ValueError, match="in flight"):
        ahead.tokens(1, 3)
    with pytest.raises(ValueError, match="already in flight"):
        ahead.prefetch(1, 3)
    with pytest.raises(ValueError, match="carry on from 1"):
        ahead.prefetch(2, 5)


def test_prefetch_is_a_no_op_without_a_worker() -> None:
    provider = rendered()
    provider.prefetch(0, 1)
    assert provider.tokens(0, 1).shape[0] == RIG.num_views * RIG.spatial_tokens


def _dies_when_asked(conn, spec) -> None:  # noqa: ANN001 - a spawn target
    """A worker that comes up, then dies hard on its first request, as an OOM would."""
    import os

    conn.send("ready")
    conn.recv()
    os._exit(3)


def _echoes(conn, spec) -> None:  # noqa: ANN001 - a spawn target
    """A worker that answers each range with no cameras, until the pipe closes."""
    conn.send("ready")
    while True:
        try:
            start, end, *_poses = conn.recv()
        except EOFError:
            return
        conn.send(([], end - start))


def _ignores_the_pipe(conn, spec) -> None:  # noqa: ANN001 - a spawn target
    """A worker stuck in a kernel: it never reads the pipe closing."""
    import time

    conn.send("ready")
    time.sleep(600)


def _spawned(target) -> ControlWorker:  # noqa: ANN001
    return ControlWorker.spawn(None, device=torch.device("cpu"), target=target)  # ty: ignore[invalid-argument-type]  # The custom worker target does not consume a WorkerSpec.


def test_a_worker_that_dies_hard_fails_the_take_instead_of_hanging_it() -> None:
    """The parent must not hold the child's end of the pipe, or recv waits for ever."""
    import threading

    worker = _spawned(_dies_when_asked)
    outcome: list[BaseException | None] = []

    def take() -> None:
        try:
            worker.take(0, 1)
            outcome.append(None)
        except BaseException as failed:  # noqa: BLE001 - asserted on below
            outcome.append(failed)

    thread = threading.Thread(target=take, daemon=True)
    thread.start()
    thread.join(timeout=60)

    assert not thread.is_alive(), "take() hung on a dead worker"
    assert isinstance(outcome[0], RuntimeError)
    assert "exited" in str(outcome[0])


def test_close_stops_the_worker_and_a_restart_gets_a_new_one() -> None:
    first = _spawned(_echoes)
    assert first.take(0, 1) == ([], 1)

    first.close()
    assert first._process is not None
    assert not first._process.is_alive()

    second = _spawned(_echoes)
    assert second.take(0, 1) == ([], 1)
    second.close()


def test_close_terminates_a_worker_that_does_not_stop_when_asked() -> None:
    worker = _spawned(_ignores_the_pipe)

    worker.close()

    assert worker._process is not None
    assert not worker._process.is_alive()


def test_a_driven_worker_draws_along_exactly_the_poses_its_request_carried() -> None:
    import numpy as np
    from cosmos3_nano_sim_auto.impl.conditioning.control_worker import _GivenPoses

    given = _GivenPoses()
    given.next = np.zeros((4, 4, 4), dtype=np.float32)

    assert given.poses(1, 5).shape == (4, 4, 4)
    with pytest.raises(ValueError, match="no poses were sent"):
        given.poses(5, 9)
    given.next = np.zeros((3, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="no poses were sent"):
        given.poses(5, 9)
