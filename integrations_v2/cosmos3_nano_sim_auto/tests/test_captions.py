# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the prompt payload: shape, key order, and chunk selection.

These pin a transcription rather than an algorithm. The payload is only correct
insofar as it matches what the model's augmentor chain emitted, so what is worth
asserting is the shape the tokenizer sees and the order the fields serialize in
-- both of which are silent when wrong. The tokenizing wrapper is checked with
a stand-in so these tests do not need the Qwen tokenizer on disk.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from cosmos3_nano_sim_auto.impl.conditioning.captions import (
    AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    CAMERA_CAPTION_PREFIXES,
    caption_json,
    format_view_caption,
    multiview_caption,
    structured_caption,
    view_caption_payloads,
)

pytestmark = pytest.mark.ci_cpu

CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
)


def _captions(count: int) -> list[dict[str, str]]:
    return [{"summary": f"view {index}"} for index in range(count)]


def _payload(count: int = 3, **overrides):
    settings: dict[str, Any] = {
        "num_frames": 200,
        "fps": 30.0,
        "pixel_height": 480,
        "pixel_width": 832,
    }
    settings.update(overrides)
    return multiview_caption(_captions(count), CAMERAS[:count], **settings)


def test_view_caption_tags_a_dict_without_disturbing_it() -> None:
    tagged = format_view_caption({"summary": "a road"}, "camera_rear_tele_30fov")

    assert tagged["summary"] == "a road"
    assert tagged["camera_view"] == CAMERA_CAPTION_PREFIXES["camera_rear_tele_30fov"]


def test_view_caption_prefixes_a_string() -> None:
    """The string branch glues the sentence on rather than making a dict."""
    tagged = format_view_caption("a road.", "camera_front_wide_120fov")

    assert tagged.startswith(CAMERA_CAPTION_PREFIXES["camera_front_wide_120fov"])
    assert tagged.endswith("a road.")


def test_unknown_camera_raises_rather_than_defaulting() -> None:
    with pytest.raises(KeyError):
        format_view_caption({"summary": "a road"}, "camera_that_does_not_exist")


def test_several_views_become_a_dict_and_not_a_list() -> None:
    """The failure this guards is silent: a list tokenizes to an empty prompt."""
    payload = _payload(3)

    assert isinstance(payload, dict)
    assert isinstance(payload["views"], list)
    assert len(payload["views"]) == 3


def test_one_view_is_passed_through_unwrapped() -> None:
    """``format_multiview_caption``'s documented special case for a lone view."""
    payload = _payload(1)

    assert "views" not in payload
    assert payload["summary"] == "view 0"


def test_views_keep_rig_order() -> None:
    payload = _payload(3)

    facings = [view["camera_view"] for view in payload["views"]]
    assert facings == [CAMERA_CAPTION_PREFIXES[camera] for camera in CAMERAS]


def test_metadata_fields_match_the_augmentors() -> None:
    payload = _payload(3)

    assert payload["duration"] == "6s"
    assert payload["fps"] == 30.0
    assert payload["resolution"] == {"H": 480, "W": 832}
    assert payload["aspect_ratio"] == "16,9"


def test_duration_truncates_the_way_the_augmentor_does() -> None:
    """``int()``, not ``round()``. A short clip honestly reports ``0s``."""
    assert _payload(3, num_frames=17)["duration"] == "0s"
    assert _payload(3, num_frames=239)["duration"] == "7s"


def test_key_order_survives_serialization() -> None:
    """Insertion order is the token order, and ``json.dumps`` preserves it."""
    text = caption_json(_payload(3))

    assert list(json.loads(text)) == [
        "views",
        "duration",
        "fps",
        "resolution",
        "aspect_ratio",
    ]
    assert (
        text.index('"views"') < text.index('"duration"') < text.index('"aspect_ratio"')
    )


def test_caption_json_refuses_a_list() -> None:
    with pytest.raises(TypeError, match="mapping"):
        caption_json([{"summary": "a road"}])  # ty: ignore[invalid-argument-type]  # Exercise invalid-input validation.


def test_mismatched_counts_raise() -> None:
    with pytest.raises(ValueError, match="pair by position"):
        multiview_caption(
            _captions(2),
            CAMERAS,
            num_frames=200,
            fps=30.0,
            pixel_height=480,
            pixel_width=832,
        )


def _record(**chunks) -> str:
    return json.dumps({"caption_structured": json.dumps(chunks)})


def test_structured_caption_parses_a_nested_json_chunk() -> None:
    record = _record(
        chunk_0={
            "caption": json.dumps({"summary": "first"}),
            "start_frame": 0,
            "end_frame": 200,
        },
        chunk_1={
            "caption": json.dumps({"summary": "second"}),
            "start_frame": 200,
            "end_frame": 400,
        },
    )

    selected = structured_caption(record, chunk=1)

    assert selected["caption"] == {"summary": "second"}
    assert selected["chunk_key"] == "chunk_1"
    assert (selected["start_frame"], selected["end_frame"]) == (200, 400)


def test_structured_caption_selects_by_key_for_the_other_cameras() -> None:
    """Chunk maps need not line up by position, which is why the key travels."""
    record = _record(
        chunk_1={
            "caption": json.dumps({"summary": "second"}),
            "start_frame": 200,
            "end_frame": 400,
        },
        chunk_0={
            "caption": json.dumps({"summary": "first"}),
            "start_frame": 0,
            "end_frame": 200,
        },
    )

    assert structured_caption(record, chunk_key="chunk_0")["caption"] == {
        "summary": "first"
    }
    assert structured_caption(record, chunk=0)["caption"] == {"summary": "second"}


def test_structured_caption_accepts_bytes_and_plain_text_captions() -> None:
    record = _record(
        chunk_0={"caption": "just a sentence", "start_frame": 0, "end_frame": 200}
    )

    selected = structured_caption(record.encode())

    assert selected["caption"] == "just a sentence"


def test_chunks_without_a_frame_range_are_not_usable() -> None:
    record = _record(
        metadata={"caption": json.dumps({"summary": "no range"})},
        chunk_0={
            "caption": json.dumps({"summary": "usable"}),
            "start_frame": 0,
            "end_frame": 200,
        },
    )

    assert structured_caption(record, chunk=0)["chunk_key"] == "chunk_0"
    with pytest.raises(ValueError, match="asked for index 1"):
        structured_caption(record, chunk=1)


def test_a_missing_chunk_key_names_what_was_there() -> None:
    record = _record(
        chunk_0={
            "caption": json.dumps({"summary": "first"}),
            "start_frame": 0,
            "end_frame": 200,
        }
    )

    with pytest.raises(ValueError, match="chunk_0"):
        structured_caption(record, chunk_key="chunk_9")


def _per_view(count: int = 3, **overrides):
    settings: dict[str, Any] = {
        "num_frames": 200,
        "fps": 30.0,
        "pixel_height": 480,
        "pixel_width": 832,
    }
    settings.update(overrides)
    return view_caption_payloads(_captions(count), CAMERAS[:count], **settings)


def test_each_camera_gets_a_whole_payload_of_its_own() -> None:
    """Not one payload holding a list, which is the other layout.

    Training's augmentors walk an ordered caption list item by item, so the
    duration, fps, resolution and aspect ratio land on every caption rather
    than once on the rig.
    """
    payloads = _per_view()

    assert len(payloads) == 3
    for index, (payload, camera) in enumerate(zip(payloads, CAMERAS, strict=True)):
        assert payload["summary"] == f"view {index}"
        assert payload["camera_view"] == CAMERA_CAPTION_PREFIXES[camera]
        assert payload["duration"] == "6s"
        assert payload["fps"] == 30.0
        assert payload["resolution"] == {"H": 480, "W": 832}
        assert "views" not in payload


def test_a_camera_s_payload_is_what_the_merged_one_holds_for_it() -> None:
    """The two layouts differ in how captions are grouped, not in their content.

    Same fields in the same order, so a difference in output is the scoping and
    not a caption that says something else.
    """
    merged = _payload()
    solo = _per_view()[1]
    metadata = ["duration", "fps", "resolution", "aspect_ratio"]

    assert list(solo) == ["summary", "camera_view", *metadata]
    assert list(merged) == ["views", *metadata]
    assert merged["views"][1] == {
        "summary": solo["summary"],
        "camera_view": solo["camera_view"],
    }
    assert caption_json(solo).startswith('{"summary": "view 1", "camera_view": ')


@pytest.mark.parametrize(
    ("captions", "cameras", "match"),
    [
        ([{"a": "b"}], CAMERAS[:2], "they pair by position"),
        ([], (), "at least one camera"),
    ],
)
def test_per_view_payloads_refuse_a_pairing_they_cannot_make(
    captions, cameras, match
) -> None:
    with pytest.raises(ValueError, match=match):
        view_caption_payloads(
            captions,
            cameras,
            num_frames=200,
            fps=30.0,
            pixel_height=480,
            pixel_width=832,
        )


@pytest.mark.parametrize(
    "system_prompt",
    [AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT, AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT],
)
def test_caption_framing_includes_the_explicit_system_message(
    monkeypatch: pytest.MonkeyPatch, system_prompt: str
) -> None:
    from cosmos3_nano_sim_auto.impl.conditioning import captions

    captured = {}
    monkeypatch.setattr(
        captions, "tokenize_prompt", lambda *args, **kwargs: captured.update(kwargs)
    )
    captions.caption_token_ids(_payload(), system_prompt=system_prompt)
    assert captured["system_prompt"] == system_prompt
    assert captured["append_resolution"] is False
