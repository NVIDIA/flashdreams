# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the input path: geometry, sample parsing and token order.

Everything here runs without a GPU, a checkpoint or a network. The one part that
needs the real VAE is the encode, so :func:`control_latents` takes the encode
function as an argument and these tests pass a stand-in -- what is worth pinning
is the stacking order and the shape agreement, not the VAE's arithmetic.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cosmos3_nano_sim_auto import inputs as inputs_module
from cosmos3_nano_sim_auto.impl.conditioning.captions import (
    AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    CAMERA_CAPTION_PREFIXES,
    caption_json,
)
from cosmos3_nano_sim_auto.impl.conditioning.controls import (
    generated_control_clips,
    synthetic_control_clip,
)
from cosmos3_nano_sim_auto.inputs import (
    MultiviewSample,
    condition_latents,
    condition_pixel_frames,
    control_latents,
    control_tokens,
    latent_frames_for_pixels,
    patch_grid,
    pixel_frames_for_latents,
    prompt_token_ids,
)

TEST_SAMPLE = Path(__file__).with_name("sample.jsonl")


pytestmark = pytest.mark.ci_cpu

PATCH = 2
"""The checkpoint's ``spec.vfm.latent_patch_size``."""

CHANNELS = 48
"""The checkpoint's ``spec.vfm.vae_latent_channels``."""


def _sample() -> MultiviewSample:
    return MultiviewSample.load(TEST_SAMPLE)


def _geometry(*, pixel_frames: int = 17, frames_per_chunk: int = 4, fps: float = 30.0):
    """The bring-up geometry: eleven views, 17 pixel frames, one four-frame chunk."""
    return _sample().geometry(
        pixel_frames=pixel_frames,
        latent_patch_size=PATCH,
        frames_per_chunk=frames_per_chunk,
        fps=fps,
    )


def _fake_encoder(
    *, channels: int = CHANNELS, latent_frames: int, height: int, width: int
):
    """An encode stand-in that stamps each view with a value only it has.

    The value lets a caller tell, from the tokens alone, which view a row came
    from, which is what makes the ordering assertions below possible without a VAE.
    """
    calls: list[int] = []

    def encode(clip: torch.Tensor) -> torch.Tensor:
        view = len(calls)
        calls.append(view)
        latent = torch.full((channels, latent_frames, height, width), float(view + 1))
        # Vary along time as well, so a frame transposition cannot pass either.
        latent += (
            torch.arange(latent_frames, dtype=latent.dtype).view(1, -1, 1, 1) * 0.01
        )
        return latent

    return encode


# --- geometry ---------------------------------------------------------------


def test_seventeen_pixel_frames_is_five_latent_frames() -> None:
    assert latent_frames_for_pixels(17) == 5
    assert pixel_frames_for_latents(5) == 17


@pytest.mark.parametrize("latent_frames", [1, 2, 5, 76])
def test_frame_conversions_round_trip(latent_frames: int) -> None:
    assert (
        latent_frames_for_pixels(pixel_frames_for_latents(latent_frames))
        == latent_frames
    )


@pytest.mark.parametrize("pixel_frames", [2, 3, 4, 16, 18])
def test_a_partial_trailing_group_is_refused(pixel_frames: int) -> None:
    # Rounding here would put the control track and the target a frame out of
    # step, which the mask cannot express and would not report.
    with pytest.raises(ValueError, match="must be 1 \\+ a multiple of 4"):
        latent_frames_for_pixels(pixel_frames)


def test_the_error_for_a_bad_frame_count_names_usable_ones() -> None:
    with pytest.raises(ValueError, match="nearest usable counts are 17 and 21"):
        latent_frames_for_pixels(19)


def test_480p_view_patches_to_the_grid_the_mask_was_sized_for() -> None:
    # 832x480 pixels -> 52x30 latent -> 26x15 patches.
    assert patch_grid(480, 832, latent_patch_size=PATCH) == (15, 26)


@pytest.mark.parametrize(("height", "width"), [(480, 830), (470, 832), (33, 33)])
def test_a_view_size_that_does_not_divide_is_refused(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="must divide by 16"):
        patch_grid(height, width, latent_patch_size=PATCH)


def test_geometry_of_the_bring_up_target() -> None:
    geometry = _geometry()

    assert geometry.num_views == 11
    assert geometry.frames_per_view == 5
    assert (geometry.patch_h, geometry.patch_w) == (15, 26)
    assert geometry.spatial_tokens == 390
    assert geometry.tokens_per_frame == 4290
    # text2video: the control track is the whole of the conditioning.
    assert geometry.condition_frames == 0


def test_seconds_per_frame_follows_fps() -> None:
    geometry = _geometry(fps=30.0)
    assert geometry.seconds_per_frame == pytest.approx(4.0 / 30.0)


# --- sample parsing ---------------------------------------------------------


def test_the_synthetic_sample_is_eleven_text2video_views() -> None:
    sample = _sample()

    assert sample.num_views == 11
    assert sample.model_mode == "text2video"
    assert sample.condition_frames == 0
    assert sample.view_size == (832, 480)
    assert sample.prompt.strip()


def test_camera_order_is_the_canonical_rig_order() -> None:
    # Nothing else in the codebase pins this, and a camera's view number is its
    # position in this tuple.
    assert _sample().cameras == (
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_right_70fov",
        "camera_rear_tele_30fov",
        "camera_rear_left_70fov",
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
        "camera_front_fisheye_200fov",
        "camera_left_fisheye_200fov",
        "camera_right_fisheye_200fov",
        "camera_rear_fisheye_200fov",
    )


def test_control_paths_stay_aligned_with_cameras() -> None:
    sample = _sample()
    assert len(sample.control_paths) == sample.num_views
    assert len(set(sample.control_paths)) == sample.num_views


def test_rig_order_is_not_physical_camera_order() -> None:
    # The trap in the whole input path. Pairing clips with views by physical
    # index would hand six of the eleven cameras another camera's road, and the
    # result would still decode and still look like driving.
    assert _sample().physical_camera_indexes == (0, 2, 4, 5, 3, 1, 6, 7, 8, 10, 9)


def test_a_control_path_without_a_camera_index_is_refused() -> None:
    sample = MultiviewSample.from_record(
        {
            "model_mode": "text2video",
            "prompt": "p",
            "resolution": "480",
            "multiview": {
                "views": [{"camera_key": "front", "control_path": "front.mp4"}]
            },
        }
    )
    with pytest.raises(ValueError, match="does not end in _cN.mp4"):
        _ = sample.physical_camera_indexes


def test_an_unknown_mode_or_resolution_raises_rather_than_guessing() -> None:
    views = {"views": [{"camera_key": "c", "control_path": "x.mp4"}]}
    unknown_mode = MultiviewSample.from_record(
        {
            "model_mode": "video2video",
            "prompt": "p",
            "resolution": "480",
            "multiview": views,
        }
    )
    with pytest.raises(ValueError, match="model_mode 'video2video'"):
        _ = unknown_mode.condition_frames

    unknown_resolution = MultiviewSample.from_record(
        {
            "model_mode": "text2video",
            "prompt": "p",
            "resolution": "1080",
            "multiview": views,
        }
    )
    with pytest.raises(ValueError, match="resolution '1080'"):
        _ = unknown_resolution.view_size


def test_a_record_without_views_is_refused(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps({"model_mode": "text2video", "prompt": "p", "multiview": {}})
    )
    with pytest.raises(ValueError, match="no multiview views"):
        MultiviewSample.load(path)


def test_asking_for_a_record_that_is_not_there_is_refused() -> None:
    with pytest.raises(ValueError, match="asked for index 9"):
        MultiviewSample.load(TEST_SAMPLE, record=9)


# --- tokens -----------------------------------------------------------------


def test_control_tokens_have_the_shape_vae2llm_expects() -> None:
    geometry = _geometry()
    latents = torch.zeros(
        geometry.num_views,
        CHANNELS,
        geometry.frames_per_view,
        geometry.patch_h * PATCH,
        geometry.patch_w * PATCH,
    )

    tokens = control_tokens(latents, latent_patch_size=PATCH)

    expected_rows = (
        geometry.num_views * geometry.frames_per_view * geometry.spatial_tokens
    )
    assert tokens.shape == (expected_rows, PATCH * PATCH * CHANNELS)
    assert expected_rows == 11 * 5 * 390


def test_tokens_run_view_outer_frame_inner() -> None:
    # The order packing.py stamps metadata in. Getting it wrong would pair every
    # token with another view's mRoPE position, which no shape check would catch.
    views, frames, patch_h, patch_w = 3, 4, 2, 2
    latents = torch.zeros(views, CHANNELS, frames, patch_h * PATCH, patch_w * PATCH)
    for view in range(views):
        for frame in range(frames):
            latents[view, :, frame] = view * 10 + frame

    tokens = control_tokens(latents, latent_patch_size=PATCH)

    per_frame = patch_h * patch_w
    for view in range(views):
        for frame in range(frames):
            start = (view * frames + frame) * per_frame
            block = tokens[start : start + per_frame]
            assert torch.equal(
                block, torch.full_like(block, float(view * 10 + frame))
            ), f"rows {start}..{start + per_frame} should be view {view}, frame {frame}"


def test_control_latents_stacks_views_in_rig_order() -> None:
    sample = _sample()
    geometry = _geometry()
    clips = generated_control_clips(sample, pixel_frames=17)
    encode = _fake_encoder(
        latent_frames=geometry.frames_per_view,
        height=geometry.patch_h * PATCH,
        width=geometry.patch_w * PATCH,
    )

    latents = control_latents(clips, encode, geometry=geometry)

    assert latents.shape[0] == geometry.num_views
    for view in range(geometry.num_views):
        assert latents[view, 0, 0, 0, 0] == pytest.approx(view + 1)


def test_a_clip_count_that_disagrees_with_the_geometry_is_refused() -> None:
    sample = _sample()
    geometry = _geometry()
    clips = generated_control_clips(sample, pixel_frames=17)[:4]
    encode = _fake_encoder(latent_frames=5, height=30, width=52)

    with pytest.raises(ValueError, match="expected 11 clips, got 4"):
        control_latents(clips, encode, geometry=geometry)


def test_an_encode_returning_the_wrong_frame_count_is_refused() -> None:
    sample = _sample()
    geometry = _geometry()
    clips = generated_control_clips(sample, pixel_frames=17)
    # Six latent frames where the geometry says five: the shape the rest of the
    # package trusts, so it has to fail here rather than downstream.
    encode = _fake_encoder(latent_frames=6, height=30, width=52)

    with pytest.raises(ValueError, match="encoded to 6 latent frames"):
        control_latents(clips, encode, geometry=geometry)


# --- conditioning frames (i2v) ----------------------------------------------


def _i2v_record(*, views: int = 2) -> dict:
    """An image2video record with one local video path per camera."""
    return {
        "name": "i2v",
        "model_mode": "image2video",
        "resolution": "480",
        "multiview": {
            "views": [
                {
                    "camera_key": camera,
                    "control_path": f"{camera}.mp4",
                    "image_path": f"{camera}.rgb.mp4",
                }
                for camera in _sample().cameras[:views]
            ]
        },
    }


def test_an_i2v_record_conditions_on_one_latent_frame() -> None:
    sample = MultiviewSample.from_record(_i2v_record())

    assert sample.image_paths == (
        "camera_front_wide_120fov.rgb.mp4",
        "camera_cross_right_120fov.rgb.mp4",
    )
    assert sample.condition_frames == 1
    geometry = sample.geometry(
        pixel_frames=17, latent_patch_size=PATCH, frames_per_chunk=4
    )
    assert geometry.condition_frames == 1


def test_image2video_without_footage_is_refused() -> None:
    """The geometry would reserve clean frames that nothing ever writes."""
    record = _i2v_record()
    for view in record["multiview"]["views"]:
        del view["image_path"]

    with pytest.raises(ValueError, match="carries no image_path"):
        MultiviewSample.from_record(record)


def test_conditioning_is_all_views_or_none() -> None:
    record = _i2v_record()
    del record["multiview"]["views"][1]["image_path"]

    with pytest.raises(ValueError, match="all or none"):
        MultiviewSample.from_record(record)


# --- the caption's frame range ----------------------------------------------


def test_the_caption_s_frame_range_is_read_rather_than_dropped() -> None:
    """It was written into sample.json and then ignored at load.

    So a ``--chunk 1`` fetch described frames 200-400 and the run conditioned on
    frames from 0 -- the prompt about one stretch of the drive, the pixels from
    another, at full confidence.
    """
    record = _i2v_record()
    record["frame_range"] = [200, 400]

    sample = MultiviewSample.from_record(record)

    assert sample.frame_range == (200, 400)
    assert sample.offset_for(pixel_frames=61) == 200


def test_a_sample_without_a_range_starts_where_it_always_did() -> None:
    """The vendored sample and hand-written records describe what they are given."""
    sample = MultiviewSample.from_record(_i2v_record())

    assert sample.frame_range is None
    assert sample.offset_for(pixel_frames=61) == 0


def test_asking_for_more_frames_than_the_caption_covers_is_refused() -> None:
    """Clamping would hide the request; the words stop before the pixels would."""
    record = _i2v_record()
    record["frame_range"] = [200, 260]
    sample = MultiviewSample.from_record(record)

    assert sample.offset_for(pixel_frames=60) == 200
    with pytest.raises(ValueError, match="describe source frames 200-260"):
        sample.offset_for(pixel_frames=61)


def test_a_run_can_be_told_to_outlive_its_caption() -> None:
    """A stream outlives every caption eventually; the default still refuses."""
    record = _i2v_record()
    record["frame_range"] = [200, 260]
    sample = MultiviewSample.from_record(record)

    assert sample.offset_for(pixel_frames=600, outlive_caption=True) == 200
    with pytest.raises(ValueError, match="--outlive-caption"):
        sample.offset_for(pixel_frames=600)


def test_outliving_a_caption_still_starts_where_the_caption_does() -> None:
    """The opt-out lifts the length check, not the offset the range supplies."""
    record = _i2v_record()
    record["frame_range"] = [200, 260]
    sample = MultiviewSample.from_record(record)

    assert sample.offset_for(
        pixel_frames=600, outlive_caption=True
    ) == sample.offset_for(pixel_frames=60)


def test_trimming_views_keeps_the_range() -> None:
    """The range is the clip's, not a view's, so trimming the rig cannot lose it."""
    record = _i2v_record()
    record["frame_range"] = [200, 400]

    trimmed = MultiviewSample.from_record(record).trimmed_to(1)

    assert trimmed.frame_range == (200, 400)


def test_a_rig_named_out_of_order_renumbers_the_views_to_match() -> None:
    """Reordering is the point: view N is whichever camera was named Nth."""
    record = _i2v_record(views=3)
    for index, view in enumerate(record["multiview"]["views"]):
        view["caption"] = {"summary": f"view {index}"}
    named = (
        "camera_rear_right_70fov",
        "camera_front_wide_120fov",
    )

    rigged = MultiviewSample.from_record(record).for_rig(named)

    assert rigged.cameras == named
    assert rigged.view_captions == ({"summary": "view 2"}, {"summary": "view 0"})
    assert rigged.image_paths == (
        "camera_rear_right_70fov.rgb.mp4",
        "camera_front_wide_120fov.rgb.mp4",
    )


def test_a_rig_of_a_sample_carrying_no_captions_stays_carrying_none() -> None:
    """An empty tuple is what says there are none, and indexing it would raise."""
    rigged = MultiviewSample.from_record(_i2v_record()).for_rig(
        ("camera_cross_right_120fov",)
    )

    assert rigged.view_captions == ()
    assert rigged.control_paths == ("camera_cross_right_120fov.mp4",)


def test_a_rig_naming_a_camera_the_sample_has_not_got_is_refused() -> None:
    with pytest.raises(ValueError, match="not cameras of"):
        MultiviewSample.from_record(_i2v_record()).for_rig(("camera_nonesuch",))


def test_a_rig_naming_one_camera_twice_is_refused() -> None:
    """Two positions reading one view leaves another with no conditioning at all."""
    with pytest.raises(ValueError, match="more than once"):
        MultiviewSample.from_record(_i2v_record()).for_rig(
            ("camera_front_wide_120fov", "camera_front_wide_120fov")
        )


def test_a_malformed_range_is_refused() -> None:
    record = _i2v_record()
    record["frame_range"] = [200]

    with pytest.raises(ValueError, match="start, end"):
        MultiviewSample.from_record(record)


def test_two_conditioning_frames_need_five_pixel_frames() -> None:
    """One latent frame is one pixel frame; two is 1 + 4, not 2."""
    sample = MultiviewSample.from_record(_i2v_record())
    geometry = sample.geometry(
        pixel_frames=17, latent_patch_size=PATCH, frames_per_chunk=4, condition_frames=2
    )

    assert geometry.condition_frames == 2
    assert condition_pixel_frames(geometry) == 5


def test_one_conditioning_frame_needs_one_pixel_frame() -> None:
    sample = MultiviewSample.from_record(_i2v_record())
    geometry = sample.geometry(
        pixel_frames=17, latent_patch_size=PATCH, frames_per_chunk=4
    )

    assert condition_pixel_frames(geometry) == 1


def test_conditioning_without_footage_is_refused_even_when_overridden() -> None:
    """A text2video sample cannot be talked into conditioning by a flag."""
    sample = _sample()

    with pytest.raises(ValueError, match="no image_path"):
        sample.geometry(
            pixel_frames=17,
            latent_patch_size=PATCH,
            frames_per_chunk=4,
            condition_frames=1,
        )


def test_condition_latents_encode_only_the_given_prefix() -> None:
    sample = MultiviewSample.from_record(_i2v_record())
    geometry = sample.geometry(
        pixel_frames=17, latent_patch_size=PATCH, frames_per_chunk=4, condition_frames=2
    )
    clips = [
        synthetic_control_clip(pixel_frames=5, pixel_height=480, pixel_width=832)
        for _ in range(geometry.num_views)
    ]
    encode = _fake_encoder(
        latent_frames=geometry.condition_frames,
        height=geometry.patch_h * PATCH,
        width=geometry.patch_w * PATCH,
    )

    latents = condition_latents(clips, encode, geometry=geometry)

    assert latents.shape[0] == geometry.num_views
    assert latents.shape[2] == geometry.condition_frames


def test_condition_latents_the_clip_length_disagrees_is_refused() -> None:
    """The prefix must be exactly the conditioning frames, not the whole clip."""
    sample = MultiviewSample.from_record(_i2v_record())
    geometry = sample.geometry(
        pixel_frames=17, latent_patch_size=PATCH, frames_per_chunk=4
    )
    clips = [
        synthetic_control_clip(pixel_frames=17, pixel_height=480, pixel_width=832)
        for _ in range(geometry.num_views)
    ]
    encode = _fake_encoder(latent_frames=5, height=30, width=52)

    with pytest.raises(ValueError, match="1 were expected"):
        condition_latents(clips, encode, geometry=geometry)


def test_condition_latents_on_a_text2video_geometry_is_refused() -> None:
    geometry = _geometry()
    clips = generated_control_clips(_sample(), pixel_frames=17)
    encode = _fake_encoder(latent_frames=5, height=30, width=52)

    with pytest.raises(ValueError, match="no conditioning frames"):
        condition_latents(clips, encode, geometry=geometry)


# --- generated controls -----------------------------------------------------


def test_a_generated_clip_has_the_layout_the_encoder_takes() -> None:
    clip = synthetic_control_clip(pixel_frames=17, pixel_height=480, pixel_width=832)

    assert clip.shape == (17, 3, 480, 832)
    assert clip.min() >= -1.0 and clip.max() <= 1.0


def test_generated_clips_differ_between_cameras() -> None:
    # Identical tracks across views would let a wrong view position or a
    # mis-stamped view id pass unnoticed.
    sample = _sample()
    clips = generated_control_clips(sample, pixel_frames=5)

    assert len(clips) == sample.num_views
    for view in range(1, len(clips)):
        assert not torch.equal(clips[0], clips[view]), f"view {view} matches view 0"


def test_a_generated_clip_moves_over_time() -> None:
    # Motion is the only thing in a stand-in that a break between chunks would
    # disturb, so a static clip could not show whether the rollout is coherent.
    clip = synthetic_control_clip(pixel_frames=8, pixel_height=240, pixel_width=416)
    for frame in range(1, clip.shape[0]):
        assert not torch.equal(clip[0], clip[frame]), f"frame {frame} matches frame 0"


def test_a_view_outside_the_rig_is_refused() -> None:
    with pytest.raises(ValueError, match="view must lie in \\[0, 4\\)"):
        synthetic_control_clip(
            pixel_frames=2, pixel_height=64, pixel_width=64, view=4, num_views=4
        )


def _stub_tokenizer(monkeypatch) -> list[str]:
    """Record what gets tokenized, and return a length that identifies it.

    The real tokenizer is a Qwen checkpoint on disk, which these tests do not
    have. What is worth pinning without it is the wiring: how many runs there
    are, what text each holds, and whether their lengths describe the stream
    they were concatenated into.
    """
    seen: list[str] = []

    def tokenize(text, **_kwargs):
        seen.append(text if isinstance(text, str) else caption_json(text))
        return torch.arange(len(seen) + 2)

    monkeypatch.setattr(inputs_module, "tokenize_prompt", tokenize)
    monkeypatch.setattr(
        inputs_module, "caption_token_ids", lambda payload, **_kwargs: tokenize(payload)
    )
    return seen


def test_a_per_camera_prompt_is_one_run_per_camera(monkeypatch) -> None:
    """The runs have to describe the ids, since nothing else says where they split."""
    seen = _stub_tokenizer(monkeypatch)
    sample = dataclasses.replace(
        _sample(), view_captions=tuple({"summary": f"view {i}"} for i in range(11))
    )

    prompt = prompt_token_ids(sample, pixel_frames=200)

    assert prompt.view_text_tokens == (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
    assert sum(prompt.view_text_tokens) == prompt.text_ids.shape[0]
    assert len(seen) == len(sample.cameras)
    assert '"camera_view"' in seen[0] and '"views"' not in seen[0]


def test_the_merged_layout_is_one_run_for_the_rig(monkeypatch) -> None:
    """``None`` is what tells everything downstream that no scoping applies."""
    seen = _stub_tokenizer(monkeypatch)
    sample = dataclasses.replace(
        _sample(), view_captions=tuple({"summary": f"view {i}"} for i in range(11))
    )

    prompt = prompt_token_ids(sample, pixel_frames=200, per_view_captions=False)

    assert prompt.view_text_tokens is None
    assert len(seen) == 1
    assert '"views"' in seen[0]


def test_a_hand_written_prompt_is_pointed_at_each_camera_in_turn(monkeypatch) -> None:
    """The vendored sample has one sentence and no per-camera captions.

    It is not what the checkpoint was fit to either way, but replicating it per
    camera at least gives the model the layout it expects, with each copy
    saying which way its camera points.
    """
    seen = _stub_tokenizer(monkeypatch)
    sample = _sample()

    prompt = prompt_token_ids(sample, pixel_frames=200)

    assert prompt.view_text_tokens is not None
    assert len(prompt.view_text_tokens) == len(sample.cameras)
    assert seen[0].startswith(CAMERA_CAPTION_PREFIXES[sample.cameras[0]])
    assert seen[0].endswith(sample.prompt)


@pytest.mark.parametrize("views", [1, 7])
@pytest.mark.parametrize("lidar", [False, True])
@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("per_view_captions", [False, True])
def test_system_message_reaches_chat_template_for_every_caption_path(
    monkeypatch: pytest.MonkeyPatch,
    views: int,
    lidar: bool,
    structured: bool,
    per_view_captions: bool,
) -> None:
    from cosmos_interactive import _tokenizer

    conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        messages: list[dict[str, str]], **kwargs: object
    ) -> list[int]:
        conversations.append(messages)
        return [10, 11]

    tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template)
    monkeypatch.setattr(_tokenizer, "load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(_tokenizer, "generation_special_token_ids", lambda _: [12])
    sample = dataclasses.replace(
        _sample().trimmed_to(views),
        prompt="A rainy residential street.",
        view_captions=tuple({"summary": f"scene {view}"} for view in range(views))
        if structured
        else (),
    )

    prompt = prompt_token_ids(
        sample,
        pixel_frames=200,
        per_view_captions=per_view_captions,
        lidar=lidar,
    )

    expected = (
        AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
        if lidar
        else AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT
    )
    assert len(conversations) == (views if per_view_captions else 1)
    for messages in conversations:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"] == expected
        if structured:
            assert "resolution" in json.loads(messages[1]["content"])
            assert "scene " in messages[1]["content"]
        else:
            assert sample.prompt in messages[1]["content"]
    assert prompt.text_ids.tolist() == [10, 11, 12] * len(conversations)
    assert prompt.view_text_tokens == ((3,) * views if per_view_captions else None)


def test_720p_padding_preserves_seven_camera_latents_and_token_order():
    from cosmos3_nano_sim_auto.impl.rollout import _to_latent
    from cosmos_interactive._patchify import unpatchify_latent

    sample = dataclasses.replace(_sample().trimmed_to(7), resolution="720")
    geometry = sample.geometry(pixel_frames=9, latent_patch_size=2, frames_per_chunk=3)
    assert sample.view_size == (1280, 720)
    assert (geometry.patch_h, geometry.patch_w) == (23, 40)
    assert geometry.latent_hw(2) == (45, 80)
    latent = torch.arange(7 * 2 * 3 * 45 * 80).reshape(7, 2, 3, 45, 80).float()
    tokens = control_tokens(latent, latent_patch_size=2)
    assert tokens.shape == (7 * 3 * 920, 8)
    reconstructed = _to_latent(
        tokens.reshape(7, 3, 920, 8), geometry, channels=2, patch=2
    )
    assert torch.equal(reconstructed, latent)
    first = unpatchify_latent(
        tokens[: 3 * 920], channels=2, grid_t=3, grid_h=23, grid_w=40, patch_spatial=2
    )
    assert torch.equal(first[..., :45, :], latent[0])
    assert torch.count_nonzero(first[..., 45:, :]) == 0


def test_selected_camera_keeps_its_physical_embedding_row() -> None:
    from cosmos3_nano_sim_auto.inputs import camera_view_ids

    assert camera_view_ids(("camera_front_tele_30fov", "camera_rear_left_70fov")) == (
        6,
        4,
    )
    sample = _sample().for_rig(("camera_front_tele_30fov",))
    geometry = sample.geometry(pixel_frames=13, latent_patch_size=2, frames_per_chunk=3)
    assert geometry.rig_view_ids == (6,)
    assert camera_view_ids(("unknown",)) is None
