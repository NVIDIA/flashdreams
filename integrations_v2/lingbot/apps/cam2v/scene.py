# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene state behind the Lingbot browser UI: prompt, first frame, events.

The browser page picks a scene -- a prompt, a first frame, and a catalog of
text events it can trigger during the rollout -- and this module holds what it
picked. The serving layer stays generic: it copies :meth:`SceneState.as_dict`
into a JSON response and hands :meth:`SceneState.apply` the decoded request
body, without knowing what any of it means.

Text events are the model-facing half. Each carries a prompt that replaces the
rollout's text conditioning while the event is active, so triggering "Storm"
mid-rollout steers generation without restarting the session. The reserved
``user_prompt`` id carries free-form text supplied at request time instead of a
catalog entry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

MAX_TEXT_EVENTS = 20
"""Ceiling on a client-supplied catalog, matching the page's own limit."""

MAX_TEXT_EVENT_LABEL_CHARS = 64
"""Ceiling on one event's button label."""

MAX_TEXT_EVENT_PROMPT_CHARS = 1_000
"""Ceiling on one event's prompt, well above any written for the presets."""

MAX_PROMPT_CHARS = 2_000
"""Ceiling on the scene prompt itself."""

MAX_IMAGE_BYTES = 15 * 1024 * 1024
"""Ceiling on an uploaded first frame."""

USER_PROMPT_EVENT_ID = "user_prompt"
"""Reserved id for free-form text supplied at request time.

Not a catalog entry, so it deliberately bypasses the membership check in
:meth:`SceneState.resolve_event_prompt`. Must match the literal in
``web/adapter.js`` and ``lingbot/impl/input_mapping.py``.
"""

_TEXT_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

_CLEAR_STATES = frozenset({"clear", "release", "off", "none"})
_TRIGGER_STATES = frozenset({"trigger", "hold", "on"})


def normalize_prompt_text(prompt: str) -> str:
    """Collapse prompt whitespace into a single line."""
    return " ".join(prompt.split())


def _normalize_field(value: object) -> str:
    return normalize_prompt_text(str(value)) if value is not None else ""


def _slugify_event_id(label: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return (slug or f"event-{index + 1}")[:64]


@dataclass(frozen=True, slots=True)
class TextEventSpec:
    """One text event the page can trigger, addressed by stable id."""

    event_id: str
    """Stable identifier the page sends back to trigger this event."""

    label: str
    """Short client-facing button label."""

    prompt: str
    """Text conditioning activated while this event is active."""

    category: str = "environment"
    """Client-facing group used to organize the event controls."""

    def as_public_dict(self) -> dict[str, str]:
        """Return the client-facing event payload."""
        return {
            "event_id": self.event_id,
            "label": self.label,
            "prompt": self.prompt,
            "category": self.category,
        }


DEFAULT_TEXT_EVENTS: tuple[TextEventSpec, ...] = (
    TextEventSpec(
        event_id="portal",
        label="Portal",
        prompt=(
            "A luminous magical portal opens in the scene, casting colored light "
            "and swirling particles into the environment."
        ),
    ),
    TextEventSpec(
        event_id="storm",
        label="Storm",
        prompt=(
            "A dramatic storm rolls in with dark clouds, wind, rain, and flashes "
            "of lightning reshaping the atmosphere."
        ),
    ),
    TextEventSpec(
        event_id="fireworks",
        label="Fireworks",
        prompt=(
            "Bright fireworks burst overhead, filling the sky with colorful sparks "
            "and reflections across the scene."
        ),
    ),
)
"""Events advertised when the page supplies no catalog of its own."""


def normalize_text_events(raw_events: object) -> tuple[TextEventSpec, ...]:
    """Validate and normalize a client-supplied text-event catalog.

    Args:
        raw_events: Decoded ``text_events`` value from the page.

    Returns:
        The catalog, with ids filled in from labels where the page omitted
        them.

    Raises:
        ValueError: The catalog is malformed, over a size limit, or contains
            a duplicate id.
    """
    if not isinstance(raw_events, (list, tuple)):
        raise ValueError("Text events must be a list.")

    text_events: list[TextEventSpec] = []
    seen_ids: set[str] = set()
    for index, raw_event in enumerate(raw_events):
        if isinstance(raw_event, TextEventSpec):
            event_id = raw_event.event_id.strip()
            label = normalize_prompt_text(raw_event.label)
            prompt = normalize_prompt_text(raw_event.prompt)
            category = normalize_prompt_text(raw_event.category) or "custom"
        elif isinstance(raw_event, Mapping):
            label = _normalize_field(raw_event.get("label"))
            prompt = _normalize_field(raw_event.get("prompt"))
            # Left unslugified until after the blank-row check below, so a
            # row with nothing in it does not acquire an id and become an
            # event with a missing prompt.
            event_id = _normalize_field(
                raw_event.get("event_id", raw_event.get("id"))
            )
            category = _normalize_field(raw_event.get("category"))
        else:
            raise ValueError("Each text event must be an object.")

        # A wholly empty entry is a trailing blank row in the page's editor,
        # not an error.
        if not event_id and not label and not prompt:
            continue
        if not event_id:
            event_id = _slugify_event_id(label, index)
        if not prompt:
            raise ValueError("Text event prompt is required.")
        if not label:
            label = event_id
        if len(label) > MAX_TEXT_EVENT_LABEL_CHARS:
            raise ValueError(
                f"Text event labels must be <= {MAX_TEXT_EVENT_LABEL_CHARS} characters."
            )
        if len(prompt) > MAX_TEXT_EVENT_PROMPT_CHARS:
            raise ValueError(
                "Text event prompts must be "
                f"<= {MAX_TEXT_EVENT_PROMPT_CHARS} characters."
            )
        if not _TEXT_EVENT_ID_RE.fullmatch(event_id):
            raise ValueError(
                "Text event ids must be 1-64 characters using only letters, "
                "numbers, '_', '.', ':', or '-'."
            )
        if event_id in seen_ids:
            raise ValueError(f"Duplicate text event id={event_id!r}.")
        seen_ids.add(event_id)
        text_events.append(
            TextEventSpec(
                event_id=event_id,
                label=label,
                prompt=prompt,
                category=category or "custom",
            )
        )

    if len(text_events) > MAX_TEXT_EVENTS:
        raise ValueError(f"At most {MAX_TEXT_EVENTS} text events are supported.")
    return tuple(text_events)


def normalize_event_state(state: object) -> str:
    """Return a trigger or clear state, defaulting anything else to trigger.

    Raises:
        ValueError: The state is a word this does not recognize.
    """
    normalized = str(state or "trigger").strip().lower() or "trigger"
    if normalized in _CLEAR_STATES or normalized in _TRIGGER_STATES:
        return normalized
    raise ValueError(
        "Event state must be one of trigger, hold, on, clear, release, off, none."
    )


def is_clear_state(state: str) -> bool:
    """Return whether ``state`` asks to restore the scene's base prompt."""
    return state in _CLEAR_STATES


@dataclass
class SceneState:
    """What the page has chosen for the session, and what it may change.

    Holds no model objects: the application owns those and reads this for the
    prompt and events to apply. Kept separate so the catalog rules are
    testable without a pipeline.
    """

    prompt: str
    """Base prompt, restored whenever an active event is cleared."""

    model: str
    """Model variant name, shown by the page."""

    video_width: int
    """Generated frame width, shown by the page."""

    video_height: int
    """Generated frame height, shown by the page."""

    default_image_url: str = ""
    """First frame the application starts with, absent a page upload."""

    image_url: str = ""
    """First frame the page selected, by URL."""

    image_bytes: bytes | None = None
    """First frame the page uploaded, which wins over :attr:`image_url`."""

    image_content_type: str = "image/jpeg"
    """Content type reported for :attr:`image_bytes`."""

    first_frame_path: str = ""
    """File the session starts from, served through the first-frame endpoint.

    A resolved first frame is a path on the server, which a browser cannot
    load, so it is read and served rather than handed over as a URL.
    """

    text_events: tuple[TextEventSpec, ...] = DEFAULT_TEXT_EVENTS
    """Catalog the page may trigger from."""

    active_event_id: str | None = None
    """Event currently steering generation, or ``None`` for the base prompt."""

    input_source: str = "default"
    """``"default"`` until the page submits a change, then ``"uploaded"``."""

    _event_prompts: dict[str, str] = field(default_factory=dict, repr=False)
    """Prompt by event id, rebuilt whenever the catalog changes."""

    def __post_init__(self) -> None:
        self.prompt = normalize_prompt_text(self.prompt)
        self._reindex_events()

    def _reindex_events(self) -> None:
        self._event_prompts = {
            event.event_id: event.prompt for event in self.text_events
        }
        # A catalog change can retire the active event; fall back to the base
        # prompt rather than leaving a dangling id the page cannot clear.
        if self.active_event_id is not None:
            if self.active_event_id not in self._event_prompts:
                if self.active_event_id != USER_PROMPT_EVENT_ID:
                    self.active_event_id = None

    def as_dict(self) -> dict[str, Any]:
        """Return the scene payload the page renders itself from."""
        return {
            "first_frame_url": "/api/session/first_frame",
            "image_url": self.image_url or self.default_image_url,
            "default_image_url": self.default_image_url,
            # True only when /api/session/first_frame has something to
            # answer with: uploaded bytes, or a file the session starts from.
            # A URL is not fetched server-side -- the page loads it directly
            # from ``image_url`` -- so reporting one here would send the page
            # to an endpoint that can only 404, once per action.
            "has_first_frame": bool(self.image_bytes or self.first_frame_path),
            "prompt": self.prompt,
            "input_source": self.input_source,
            "model": self.model,
            "capabilities": {"text_events": bool(self.text_events)},
            "event_catalog": [event.as_public_dict() for event in self.text_events],
            "active_event_id": self.active_event_id,
            "resolution": {"width": self.video_width, "height": self.video_height},
        }

    def active_prompt(self) -> str:
        """Return the text the rollout should currently be conditioned on.

        The active event's prompt while one is active, otherwise the scene's
        base prompt. Used to restore conditioning after a reset re-seeds the
        rollout from the session's own prompt.
        """
        if self.active_event_id is None:
            return self.prompt
        return self._event_prompts.get(self.active_event_id, self.prompt)

    def resolve_event_prompt(
        self, *, event_id: str, state: str, prompt: str = ""
    ) -> str | None:
        """Return the prompt an event asks for, and record it as active.

        Returns:
            The text to condition on, or ``None`` when nothing changes.

        Raises:
            ValueError: The event is not in the catalog, or a free-form
                request carried no text.
        """
        state = normalize_event_state(state)
        event_id = event_id.strip()
        if is_clear_state(state) or not event_id:
            self.active_event_id = None
            return self.prompt

        if event_id == USER_PROMPT_EVENT_ID:
            free_form = normalize_prompt_text(prompt)
            if not free_form:
                raise ValueError("A free-form prompt requires prompt text.")
            if len(free_form) > MAX_PROMPT_CHARS:
                raise ValueError(f"Prompt must be <= {MAX_PROMPT_CHARS} characters.")
            self.active_event_id = event_id
            return free_form

        event_prompt = self._event_prompts.get(event_id)
        if event_prompt is None:
            supported = ", ".join(sorted(self._event_prompts)) or "none"
            raise ValueError(f"Unknown event_id={event_id!r}. Supported: {supported}")
        self.active_event_id = event_id
        return event_prompt

    def apply(self, payload: Mapping[str, Any]) -> str | None:
        """Apply one page submission and return any prompt to condition on.

        Args:
            payload: Decoded request body. ``prompt``, ``image``/``image_url``,
                and ``text_events`` set the scene; ``event_id``/``state``
                trigger one event from the catalog.

        Returns:
            Text to condition the rollout on now, or ``None`` when the change
            only takes effect for the next session.

        Raises:
            ValueError: The submission is malformed or names an unknown event.
        """
        changed = False

        raw_events = payload.get("text_events", payload.get("events"))
        if isinstance(raw_events, str) and raw_events.strip():
            try:
                raw_events = json.loads(raw_events)
            except json.JSONDecodeError as error:
                raise ValueError("Text events must be valid JSON.") from error
        if raw_events is not None and not isinstance(raw_events, str):
            self.text_events = normalize_text_events(raw_events)
            self._reindex_events()
            changed = True

        prompt_changed = False
        raw_prompt = payload.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt.strip():
            prompt = normalize_prompt_text(raw_prompt)
            if len(prompt) > MAX_PROMPT_CHARS:
                raise ValueError(f"Prompt must be <= {MAX_PROMPT_CHARS} characters.")
            prompt_changed = prompt != self.prompt
            self.prompt = prompt
            changed = True

        image = payload.get("image")
        if isinstance(image, bytes) and image:
            if len(image) > MAX_IMAGE_BYTES:
                raise ValueError(
                    f"First-frame image must be <= {MAX_IMAGE_BYTES} bytes."
                )
            content_type = str(payload.get("image_content_type", "image/jpeg"))
            if not content_type.startswith("image/"):
                raise ValueError("Uploaded first frame must be an image.")
            self.image_bytes = image
            self.image_content_type = content_type
            # An upload supersedes any previously chosen URL.
            self.image_url = ""
            changed = True
        else:
            raw_url = payload.get("image_url")
            if isinstance(raw_url, str) and raw_url.strip():
                self.image_url = raw_url.strip()
                self.image_bytes = None
                changed = True

        event_id = payload.get("event_id")
        if isinstance(event_id, str) and event_id.strip():
            return self.resolve_event_prompt(
                event_id=event_id,
                state=str(payload.get("state", "trigger")),
                prompt=str(payload.get("prompt", "")),
            )

        if not changed:
            raise ValueError(
                "Submit a prompt, an image, an image URL, text events, or an event."
            )
        self.input_source = "uploaded"
        if prompt_changed:
            # Steer the running rollout rather than waiting for a session that
            # may never come: the runtime builds one session per process, so a
            # prompt recorded for "next time" would never be seen. Switching
            # scenes mid-rollout is exactly what the text-context swap is for.
            # Any active event is superseded by the new base prompt.
            self.active_event_id = None
            return self.prompt
        return None


__all__ = [
    "DEFAULT_TEXT_EVENTS",
    "MAX_IMAGE_BYTES",
    "MAX_PROMPT_CHARS",
    "MAX_TEXT_EVENTS",
    "MAX_TEXT_EVENT_LABEL_CHARS",
    "MAX_TEXT_EVENT_PROMPT_CHARS",
    "USER_PROMPT_EVENT_ID",
    "SceneState",
    "TextEventSpec",
    "is_clear_state",
    "normalize_event_state",
    "normalize_prompt_text",
    "normalize_text_events",
]
