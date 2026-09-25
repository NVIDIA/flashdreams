# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-state rules behind the Lingbot browser UI, without a pipeline."""

import json

import pytest
from lingbot.apps.cam2v.scene import (
    DEFAULT_TEXT_EVENTS,
    MAX_PROMPT_CHARS,
    MAX_TEXT_EVENT_LABEL_CHARS,
    MAX_TEXT_EVENT_PROMPT_CHARS,
    MAX_TEXT_EVENTS,
    USER_PROMPT_EVENT_ID,
    SceneState,
    TextEventSpec,
    normalize_event_state,
    normalize_prompt_text,
    normalize_text_events,
)

pytestmark = pytest.mark.ci_cpu


def make_scene(**overrides) -> SceneState:
    """Return a scene with the fields every test needs already set."""
    values = {
        "prompt": "a quiet street",
        "model": "lingbot-world-fast",
        "video_width": 832,
        "video_height": 464,
    }
    values.update(overrides)
    return SceneState(**values)


def test_normalize_prompt_text_collapses_whitespace() -> None:
    assert normalize_prompt_text("  a \n  b\tc  ") == "a b c"


def test_scene_payload_carries_what_the_page_renders() -> None:
    scene = make_scene(default_image_url="https://example.test/first.jpg")

    payload = scene.as_dict()

    # Every key the page reads; a rename here breaks the UI silently.
    assert payload["prompt"] == "a quiet street"
    assert payload["model"] == "lingbot-world-fast"
    assert payload["resolution"] == {"width": 832, "height": 464}
    assert payload["first_frame_url"] == "/api/session/first_frame"
    assert payload["image_url"] == "https://example.test/first.jpg"
    assert payload["default_image_url"] == "https://example.test/first.jpg"
    # A URL is loaded by the page directly; only uploaded bytes are
    # something /api/session/first_frame can answer with.
    assert payload["has_first_frame"] is False
    assert payload["active_event_id"] is None
    assert payload["input_source"] == "default"
    assert payload["capabilities"] == {"text_events": True}
    assert [event["event_id"] for event in payload["event_catalog"]] == [
        event.event_id for event in DEFAULT_TEXT_EVENTS
    ]


def test_no_first_frame_is_reported_as_absent() -> None:
    assert make_scene().as_dict()["has_first_frame"] is False


def test_uploaded_bytes_are_reported_as_a_servable_first_frame() -> None:
    scene = make_scene()

    scene.apply({"image": b"\xff\xd8jpeg", "image_content_type": "image/jpeg"})

    assert scene.as_dict()["has_first_frame"] is True


def test_triggering_a_catalog_event_returns_its_prompt() -> None:
    scene = make_scene()

    prompt = scene.apply({"event_id": "storm", "state": "trigger"})

    assert prompt == DEFAULT_TEXT_EVENTS[1].prompt
    assert scene.active_event_id == "storm"
    assert scene.as_dict()["active_event_id"] == "storm"


def test_clearing_an_event_restores_the_base_prompt() -> None:
    scene = make_scene()
    scene.apply({"event_id": "storm", "state": "trigger"})

    prompt = scene.apply({"event_id": "storm", "state": "clear"})

    assert prompt == "a quiet street"
    assert scene.active_event_id is None


def test_unknown_event_is_rejected_with_the_supported_ids() -> None:
    scene = make_scene()

    with pytest.raises(ValueError, match="Unknown event_id.*storm"):
        scene.apply({"event_id": "nope", "state": "trigger"})


def test_user_prompt_bypasses_the_catalog() -> None:
    scene = make_scene()

    prompt = scene.apply(
        {"event_id": USER_PROMPT_EVENT_ID, "prompt": "  a  sudden   eclipse "}
    )

    assert prompt == "a sudden eclipse"
    assert scene.active_event_id == USER_PROMPT_EVENT_ID


def test_user_prompt_without_text_is_rejected() -> None:
    scene = make_scene()

    with pytest.raises(ValueError, match="requires prompt text"):
        scene.apply({"event_id": USER_PROMPT_EVENT_ID, "prompt": "   "})


def test_setting_a_prompt_steers_the_running_rollout() -> None:
    scene = make_scene()

    # The runtime builds one session per process, so a prompt held back for
    # "the next session" would never be applied at all.
    assert scene.apply({"prompt": "a new scene"}) == "a new scene"
    assert scene.prompt == "a new scene"
    assert scene.as_dict()["input_source"] == "uploaded"


def test_an_unchanged_prompt_does_not_resteer() -> None:
    scene = make_scene()

    assert scene.apply({"prompt": "  a quiet   street "}) is None


def test_a_new_prompt_supersedes_the_active_event() -> None:
    scene = make_scene()
    scene.apply({"event_id": "storm", "state": "trigger"})

    assert scene.apply({"prompt": "a new scene"}) == "a new scene"
    assert scene.active_event_id is None


def test_an_image_only_change_does_not_steer_the_rollout() -> None:
    scene = make_scene()

    # A rollout cannot swap the frame it was initialized from.
    assert scene.apply({"image_url": "https://example.test/one.jpg"}) is None


def test_uploaded_image_supersedes_a_previously_chosen_url() -> None:
    scene = make_scene()
    scene.apply({"image_url": "https://example.test/one.jpg"})

    scene.apply({"image": b"\xff\xd8jpeg", "image_content_type": "image/jpeg"})

    assert scene.image_bytes == b"\xff\xd8jpeg"
    assert scene.image_url == ""


def test_non_image_upload_is_rejected() -> None:
    scene = make_scene()

    with pytest.raises(ValueError, match="must be an image"):
        scene.apply({"image": b"%PDF", "image_content_type": "application/pdf"})


def test_catalog_arrives_as_json_from_multipart() -> None:
    scene = make_scene()
    events = [{"label": "Meteor Shower", "prompt": "meteors streak overhead"}]

    scene.apply({"text_events": json.dumps(events)})

    assert [event.event_id for event in scene.text_events] == ["meteor-shower"]
    assert scene.apply({"event_id": "meteor-shower"}) == "meteors streak overhead"


def test_malformed_catalog_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        make_scene().apply({"text_events": "{not json"})


def test_empty_submission_is_rejected() -> None:
    with pytest.raises(ValueError, match="Submit a prompt"):
        make_scene().apply({})


def test_retired_active_event_does_not_dangle() -> None:
    scene = make_scene()
    scene.apply({"event_id": "storm", "state": "trigger"})

    # A catalog without "storm" leaves nothing for the page to clear.
    scene.apply({"text_events": [{"event_id": "calm", "prompt": "still air"}]})

    assert scene.active_event_id is None


def test_ids_are_slugified_from_labels_when_omitted() -> None:
    events = normalize_text_events(
        [{"label": "Rogue Wave!", "prompt": "a wall of water"}, {"prompt": "quiet"}]
    )

    assert [event.event_id for event in events] == ["rogue-wave", "event-2"]


def test_blank_catalog_rows_are_dropped_not_rejected() -> None:
    assert normalize_text_events([{"label": "", "prompt": "", "event_id": ""}]) == ()


def test_event_without_a_prompt_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt is required"):
        normalize_text_events([{"label": "Nameless"}])


def test_duplicate_event_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate text event id"):
        normalize_text_events(
            [
                {"event_id": "storm", "prompt": "one"},
                {"event_id": "storm", "prompt": "two"},
            ]
        )


def test_event_id_charset_is_enforced() -> None:
    with pytest.raises(ValueError, match="Text event ids must be"):
        normalize_text_events([{"event_id": "bad id!", "prompt": "x"}])


@pytest.mark.parametrize(
    ("field", "limit", "message"),
    [
        ("label", MAX_TEXT_EVENT_LABEL_CHARS, "labels must be"),
        ("prompt", MAX_TEXT_EVENT_PROMPT_CHARS, "prompts must be"),
    ],
)
def test_event_field_length_limits(field: str, limit: int, message: str) -> None:
    event = {"event_id": "long", "label": "L", "prompt": "p"}
    event[field] = "x" * (limit + 1)

    with pytest.raises(ValueError, match=message):
        normalize_text_events([event])


def test_catalog_size_is_capped() -> None:
    events = [
        {"event_id": f"e{index}", "prompt": "x"} for index in range(MAX_TEXT_EVENTS + 1)
    ]

    with pytest.raises(ValueError, match=f"At most {MAX_TEXT_EVENTS}"):
        normalize_text_events(events)


def test_catalog_at_the_cap_is_accepted() -> None:
    events = [
        {"event_id": f"e{index}", "prompt": "x"} for index in range(MAX_TEXT_EVENTS)
    ]

    assert len(normalize_text_events(events)) == MAX_TEXT_EVENTS


def test_prompt_length_is_capped() -> None:
    with pytest.raises(ValueError, match=f"<= {MAX_PROMPT_CHARS}"):
        make_scene().apply({"prompt": "x" * (MAX_PROMPT_CHARS + 1)})


def test_catalog_must_be_a_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        normalize_text_events({"event_id": "storm", "prompt": "x"})


def test_specs_pass_through_normalization() -> None:
    spec = TextEventSpec(event_id="storm", label="Storm", prompt="  wind   rises ")

    assert normalize_text_events([spec])[0].prompt == "wind rises"


@pytest.mark.parametrize("state", ["trigger", "hold", "on", "clear", "release", "off"])
def test_recognized_event_states(state: str) -> None:
    assert normalize_event_state(state) == state


def test_unrecognized_event_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="Event state must be"):
        normalize_event_state("sideways")


def test_a_session_first_frame_is_servable() -> None:
    scene = make_scene(first_frame_path="/srv/example/first_frame.png")

    payload = scene.as_dict()

    # The path is not handed to the page: it is served through the endpoint.
    assert payload["has_first_frame"] is True
    assert payload["image_url"] == ""


def test_an_upload_is_served_ahead_of_the_session_frame() -> None:
    scene = make_scene(first_frame_path="/srv/example/first_frame.png")

    scene.apply({"image": b"\xff\xd8jpeg", "image_content_type": "image/jpeg"})

    assert scene.image_bytes == b"\xff\xd8jpeg"
    assert scene.as_dict()["has_first_frame"] is True


def test_active_prompt_is_the_base_prompt_when_no_event_runs() -> None:
    assert make_scene().active_prompt() == "a quiet street"


def test_active_prompt_follows_a_triggered_event() -> None:
    scene = make_scene()
    scene.apply({"event_id": "storm", "state": "trigger"})

    # What a reset must restore: the event, not the scene's base prompt.
    assert scene.active_prompt() == DEFAULT_TEXT_EVENTS[1].prompt


def test_active_prompt_returns_to_base_after_a_clear() -> None:
    scene = make_scene()
    scene.apply({"event_id": "storm", "state": "trigger"})
    scene.apply({"event_id": "storm", "state": "clear"})

    assert scene.active_prompt() == "a quiet street"
