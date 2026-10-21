# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the decoder, against a stand-in for TAEHV-super.

The real decoder is a checkpoint download and a GPU, so what stands in for it
here is a few lines that copy the two behaviours the wrapper is built around: it
carries temporal state between calls, and its first call comes up 3 frames
short. Those are the parts that can be got wrong, and they do not need real
weights to test.

The test worth reading is ``test_the_views_do_not_bleed_into_each_other``. The
whole reason this module exists is that the stock chunk decoder pins the batch
axis to 1, and the tempting fix -- call it once per camera against the one cache
it already has -- gives every camera the previous camera's last frame as its
history. The output is the right shape and looks like video, so the stand-in
records what the batch axis actually carried and the test pins that the eleven
cameras went through as eleven rows of one call.
"""

from __future__ import annotations

import pytest
import torch
from cosmos3_nano_sim_auto.impl.decode import (
    FRAMES_TO_TRIM,
    TIME_UPSCALE,
    decoded_pixel_frames,
    multiview_chunk_decoder,
)
from cosmos3_nano_sim_auto.inputs import pixel_frames_for_latents

pytestmark = pytest.mark.ci_cpu

VIEWS = 3
CHANNELS = 4
HEIGHT = 2
WIDTH = 3


class FakeTaehv:
    """A causal decoder's shape and warm-up, and nothing else.

    Upscales time by 4 and space by 16 by repeating, trims 3 frames on the first
    call only, and records the batch and length of every call so a test can ask
    how the cameras were fed in.
    """

    def __init__(
        self, *, time_upscale: int = TIME_UPSCALE, frames_to_trim: int = FRAMES_TO_TRIM
    ):
        self.time_upscale = time_upscale
        self.frames_to_trim = frames_to_trim
        self.calls: list[tuple[int, ...]] = []
        self._warm = False

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(z.shape))
        # [V, t, C, H, W] -> [V, t*4, 3, H*16, W*16], content carried through so
        # a test can tell which latent frame became which pixel frame.
        pixels = z[:, :, :3].repeat_interleave(self.time_upscale, dim=1)
        pixels = pixels.repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1)
        if not self._warm:
            self._warm = True
            pixels = pixels[:, self.frames_to_trim :]
        return pixels


def latent(frames: int, *, views: int = VIEWS, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(views, CHANNELS, frames, HEIGHT, WIDTH, generator=generator)


# --- the frame arithmetic ---------------------------------------------------


def test_the_clip_comes_back_the_length_it_went_in():
    # The point of the trim: the encoder turned 17 pixel frames into 5 latent
    # frames, and 5 latent frames have to decode back to 17, not 20.
    for latent_frames in (2, 5, 9, 21):
        assert decoded_pixel_frames(latent_frames) == pixel_frames_for_latents(
            latent_frames
        )


def test_the_bring_up_target_is_seventeen_frames():
    assert decoded_pixel_frames(5) == 17


def test_one_latent_frame_survives_the_warm_up_by_exactly_one_frame():
    # The shortest clip that decodes to anything, and the reason the first
    # rollout chunk being a single frame is not a problem.
    assert decoded_pixel_frames(1) == 1


def test_a_clip_too_short_to_survive_the_warm_up_is_an_error():
    with pytest.raises(ValueError, match="warm-up"):
        decoded_pixel_frames(1, frames_to_trim=7)


# --- the streaming wrapper -------------------------------------------------


def test_the_whole_clip_decodes_to_the_frames_the_arithmetic_promised():
    decode = multiview_chunk_decoder(FakeTaehv())
    pixels = decode(latent(5))
    assert pixels.shape == (VIEWS, 17, 3, HEIGHT * 16, WIDTH * 16)


def test_the_views_do_not_bleed_into_each_other():
    fake = FakeTaehv()
    decode = multiview_chunk_decoder(fake)
    decode(latent(5, views=11))

    # Every call carried all eleven cameras as rows of one batch. A per-camera
    # loop would show eleven calls of batch 1 -- and would hand each camera the
    # previous camera's temporal state.
    assert [shape[0] for shape in fake.calls] == [11] * len(fake.calls)


def test_the_decoder_is_fed_uniform_slices():
    fake = FakeTaehv()
    decode = multiview_chunk_decoder(fake, len_t=1)
    decode(latent(5))
    assert [shape[1] for shape in fake.calls] == [1, 1, 1, 1, 1]


def test_slices_are_handed_over_as_the_decoder_wants_them():
    # [V, C, T, H, W] into the wrapper, [V, T, C, H, W] into TAEHV.
    fake = FakeTaehv()
    decode = multiview_chunk_decoder(fake, len_t=1)
    decode(latent(5))
    assert fake.calls[0] == (VIEWS, 1, CHANNELS, HEIGHT, WIDTH)


def test_streaming_a_chunk_at_a_time_is_the_same_as_the_whole_clip():
    # What the rollout will do: decode chunk [0, 1) and then [1, 5), against one
    # decoder, and get what a single call over all five frames would have given.
    clip = latent(5)
    whole = multiview_chunk_decoder(FakeTaehv())(clip)

    streamed = multiview_chunk_decoder(FakeTaehv())
    pieces = [streamed(clip[:, :, :1]), streamed(clip[:, :, 1:])]

    assert torch.equal(torch.cat(pieces, dim=1), whole)


def test_only_the_first_chunk_pays_the_warm_up():
    decode = multiview_chunk_decoder(FakeTaehv())
    first = decode(latent(5)[:, :, :1])
    second = decode(latent(5)[:, :, 1:])
    assert first.shape[1] == TIME_UPSCALE - FRAMES_TO_TRIM
    assert second.shape[1] == 4 * TIME_UPSCALE


def test_a_longer_slice_is_still_one_call_per_slice():
    fake = FakeTaehv()
    decode = multiview_chunk_decoder(fake, len_t=2)
    pixels = decode(latent(4))
    assert [shape[1] for shape in fake.calls] == [2, 2]
    assert pixels.shape[1] == decoded_pixel_frames(4)


# --- the ways a caller can get it wrong ------------------------------------


def test_a_latent_missing_its_view_axis_is_an_error():
    decode = multiview_chunk_decoder(FakeTaehv())
    with pytest.raises(ValueError, match=r"\[V, C, T, H, W\]"):
        decode(torch.randn(CHANNELS, 5, HEIGHT, WIDTH))


def test_frames_that_do_not_fill_a_slice_are_an_error():
    decode = multiview_chunk_decoder(FakeTaehv(), len_t=2)
    with pytest.raises(ValueError, match="not a multiple"):
        decode(latent(5))


def test_the_view_count_cannot_change_mid_stream():
    decode = multiview_chunk_decoder(FakeTaehv())
    decode(latent(2, views=3))
    with pytest.raises(ValueError, match="cannot switch"):
        decode(latent(2, views=7))


def test_a_decoder_told_its_view_count_complains_on_the_first_call():
    decode = multiview_chunk_decoder(FakeTaehv(), num_views=11)
    with pytest.raises(ValueError, match="pinned to 11 views"):
        decode(latent(2, views=7))


def test_a_decoder_that_trims_differently_is_caught():
    # If TAEHV-super's warm-up were ever not 3 frames, every clip would come out
    # quietly the wrong length. It should fail instead.
    decode = multiview_chunk_decoder(FakeTaehv(frames_to_trim=1))
    with pytest.raises(ValueError, match="warm-up trim"):
        decode(latent(5))
