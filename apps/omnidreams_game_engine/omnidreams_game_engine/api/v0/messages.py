# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message types of the external-engine protocol, version 0.

Each message is a dataclass whose fields are the JSON header keys. Unknown
header keys are kept in ``extra`` and written back unchanged, so a receiver
never drops fields a newer sender adds. ``PROTOCOL.md`` beside this module is
the wire specification; ``schema.json`` is generated from these classes by
:func:`json_schema`. The module imports neither torch nor a renderer.
"""

from __future__ import annotations

import json
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Union

PROTOCOL = "omnidreams-engine/0"
"""Value of ``hello.api`` and ``config.api`` for this version."""

_RESERVED = ("type", "bytes")
"""Header keys owned by the framing; never message fields or ``extra``."""

REFUSAL_CODES = frozenset({"version", "state_format", "busy", "timing"})
"""``error`` codes that refuse an engine; the host then closes the connection."""


class ProtocolError(ValueError):
    """A message that violates the protocol; ``code`` goes into an ``error``."""

    def __init__(self, message: str, *, code: str = "protocol") -> None:
        super().__init__(message)
        self.code = code


@dataclass(kw_only=True, slots=True)
class Message:
    """Base of every protocol message."""

    TYPE: ClassVar[str] = ""
    SENDER: ClassVar[str] = "both"
    """Who sends it: ``engine``, ``host`` or ``both``."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Header keys this version does not define, kept verbatim."""

    def to_header(self) -> dict[str, Any]:
        """Return the JSON header (without ``bytes``; framing adds it)."""
        header: dict[str, Any] = {"type": self.TYPE}
        for name, value in self.extra.items():
            if name not in _RESERVED:
                header[name] = value
        for item in fields(self):
            if item.name == "extra":
                continue
            value = getattr(self, item.name)
            if value is None:
                continue
            header[item.name] = value
        return header


@dataclass(kw_only=True, slots=True)
class Unknown(Message):
    """A message type this version does not define; receivers ignore it."""

    type: str


## Engine to host


@dataclass(kw_only=True, slots=True)
class Hello(Message):
    """First message of every connection, sent by the engine."""

    TYPE: ClassVar[str] = "hello"
    SENDER: ClassVar[str] = "engine"

    session: str
    """Engine session name; a reconnect sends the same name."""
    api: str | None = None
    """Protocol the engine speaks, ``omnidreams-engine/0``; a hello without it is refused."""
    state_format: str | None = None
    """Vocabulary of ``frame.state``; the host's state decoder must accept it."""
    level: str | None = None
    """Name of the engine's map or level, for the host's scene provider and logs."""
    layout_crc32: int | None = None
    """Identifies the map; a host's scene provider may refuse one it cannot draw."""
    driver: str | None = None
    """``human`` or ``autopilot``."""
    sim_hz: int = 60
    """Fixed simulation rate of the engine."""
    tick: int = 0
    """Engine tick when the hello was sent."""
    front_mount: dict[str, Any] | None = None
    """Mount of the engine's front camera on the vehicle, free-form."""
    wide_seed: dict[str, Any] | None = None
    """Size and field of view of the engine's seed camera, free-form."""
    engine_build: str | None = None
    t_engine_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Start(Message):
    """The engine begins (or, after a reconnect, resumes) a driving session."""

    TYPE: ClassVar[str] = "start"
    SENDER: ClassVar[str] = "engine"

    tick: int
    reason: str = "start"
    """``start`` or ``reconnect``."""


@dataclass(kw_only=True, slots=True)
class Frame(Message):
    """The engine's state at one granted model frame; the seed carries an image."""

    TYPE: ClassVar[str] = "frame"
    SENDER: ClassVar[str] = "engine"

    epoch: int
    g: int
    """Model frame index within the epoch; 0 is the seed frame."""
    tick: int
    """Engine tick the state belongs to."""
    state: dict[str, Any]
    """Per-frame state in the vocabulary named by ``hello.state_format``."""
    seed: bool = False
    """True on the answer to a ``seed_req``; the payload is the seed image."""
    encoding: str | None = None
    """Payload encoding: ``jpeg``, ``png`` or ``raw`` (RGB24); absent without image."""
    width: int | None = None
    height: int | None = None
    t_engine_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Pong(Message):
    """Answer to ``ping``."""

    TYPE: ClassVar[str] = "pong"
    SENDER: ClassVar[str] = "engine"

    t_host_us: int
    t_engine_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Views(Message):
    """The host views the engine shows; the host sends their ``view_frame``s.

    An empty list stops them; a new connection starts with none. A host
    ignores a view it does not serve.
    """

    TYPE: ClassVar[str] = "views"
    SENDER: ClassVar[str] = "engine"

    views: list[str] = field(default_factory=list)
    t_engine_us: int | None = None


## Host to engine


@dataclass(kw_only=True, slots=True)
class Config(Message):
    """The host's answer to ``hello``: the session's timing and frame return."""

    TYPE: ClassVar[str] = "config"
    SENDER: ClassVar[str] = "host"

    session: str
    api: str
    model_fps: int
    frame_ticks: int
    """Engine ticks per model frame (``sim_hz / model_fps``)."""
    chunk_plan: list[int]
    """Frames per chunk; the last entry repeats (``[5, 8]``)."""
    lead_chunks: int
    """How many chunks beyond the one being generated the host grants."""
    display_delay_ticks: int
    """Default presentation delay of a generated frame after its tick."""
    return_encoding: str
    """Encoding of ``gen_frame`` payloads: ``jpeg`` or ``raw``."""
    return_width: int | None = None
    return_height: int | None = None
    return_quality: int | None = None
    render: str = "off"
    """Engine world rendering between seeds: ``off`` or ``always``."""
    preset: str | None = None
    host_build: str | None = None


@dataclass(kw_only=True, slots=True)
class SeedReq(Message):
    """Ask the engine for the seed image and state that start an epoch."""

    TYPE: ClassVar[str] = "seed_req"
    SENDER: ClassVar[str] = "host"

    epoch: int
    tick: int
    """Earliest tick; an engine already past it answers at its current tick."""
    reason: str = "start"
    width: int | None = None
    height: int | None = None
    fov_deg: float | None = None


@dataclass(kw_only=True, slots=True)
class FrameReq(Message):
    """Grant: the engine may simulate up to ``tick`` and answers with its state."""

    TYPE: ClassVar[str] = "frame_req"
    SENDER: ClassVar[str] = "host"

    epoch: int
    chunk: int
    index: int
    g: int
    tick: int


@dataclass(kw_only=True, slots=True)
class GenFrame(Message):
    """One generated frame; the payload is the encoded image."""

    TYPE: ClassVar[str] = "gen_frame"
    SENDER: ClassVar[str] = "host"

    epoch: int
    chunk: int
    g: int
    tick: int
    """Engine tick whose state produced the frame."""
    encoding: str
    width: int
    height: int
    t_host_us: int | None = None


@dataclass(kw_only=True, slots=True)
class ViewFrame(Message):
    """One frame of a host view the engine listed; the payload is the image.

    ``view`` names it (``map``: the conditioning frame the model generated
    the frame ``g`` from).
    """

    TYPE: ClassVar[str] = "view_frame"
    SENDER: ClassVar[str] = "host"

    view: str
    epoch: int
    chunk: int
    g: int
    tick: int
    """Engine tick whose state produced the frame."""
    encoding: str
    width: int
    height: int
    t_host_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Input(Message):
    """A key edge forwarded from a host-side view."""

    TYPE: ClassVar[str] = "input"
    SENDER: ClassVar[str] = "host"

    action: str
    phase: str
    """``down`` or ``up``."""
    t_host_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Cancel(Message):
    """Drop the outstanding grants (of ``epoch`` when given)."""

    TYPE: ClassVar[str] = "cancel"
    SENDER: ClassVar[str] = "host"

    epoch: int | None = None


@dataclass(kw_only=True, slots=True)
class Ping(Message):
    """Liveness check; the engine answers with ``pong``."""

    TYPE: ClassVar[str] = "ping"
    SENDER: ClassVar[str] = "host"

    t_host_us: int


## Both directions


@dataclass(kw_only=True, slots=True)
class Hold(Message):
    """Engine: the player paused. Host: the engine must stop advancing."""

    TYPE: ClassVar[str] = "hold"

    reason: str | None = None
    tick: int | None = None


@dataclass(kw_only=True, slots=True)
class Release(Message):
    """Ends a ``hold``; the same epoch continues."""

    TYPE: ClassVar[str] = "release"

    reason: str | None = None
    tick: int | None = None


@dataclass(kw_only=True, slots=True)
class Restart(Message):
    """A discontinuity: the host starts a new epoch with a new seed."""

    TYPE: ClassVar[str] = "restart"

    reason: str = "restart"
    tick: int | None = None
    epoch: int | None = None


@dataclass(kw_only=True, slots=True)
class End(Message):
    """The driving session ends; the connection and the host process stay."""

    TYPE: ClassVar[str] = "end"

    reason: str | None = None
    tick: int | None = None


@dataclass(kw_only=True, slots=True)
class Status(Message):
    """Periodic status; every field other than ``tick`` is free-form."""

    TYPE: ClassVar[str] = "status"

    tick: int | None = None


@dataclass(kw_only=True, slots=True)
class Error(Message):
    """A refusal or failure with a machine-readable ``code``."""

    TYPE: ClassVar[str] = "error"

    code: str
    message: str = ""


MESSAGE_TYPES: tuple[type[Message], ...] = (
    Hello,
    Start,
    Frame,
    Pong,
    Config,
    SeedReq,
    FrameReq,
    GenFrame,
    Input,
    Cancel,
    Ping,
    Hold,
    Release,
    Restart,
    End,
    Status,
    Error,
    Views,
    ViewFrame,
)
"""Every message type of version 0."""

_REGISTRY: dict[str, type[Message]] = {cls.TYPE: cls for cls in MESSAGE_TYPES}


## Parsing


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    name: str
    kind: type
    item: type | None
    """Element type of a list field."""
    required: bool
    default: Any


_SPECS: dict[type[Message], tuple[_FieldSpec, ...]] = {}


def _field_specs(cls: type[Message]) -> tuple[_FieldSpec, ...]:
    cached = _SPECS.get(cls)
    if cached is not None:
        return cached
    hints = typing.get_type_hints(cls)
    specs: list[_FieldSpec] = []
    for item in fields(cls):
        if item.name == "extra":
            continue
        annotation = hints[item.name]
        if typing.get_origin(annotation) is Union or isinstance(
            annotation, types.UnionType
        ):
            (annotation,) = [
                a for a in typing.get_args(annotation) if a is not type(None)
            ]
        kind = typing.get_origin(annotation) or annotation
        args = typing.get_args(annotation)
        if item.default is not MISSING:
            default = item.default
        elif item.default_factory is not MISSING:
            default = item.default_factory()
        else:
            default = MISSING
        specs.append(
            _FieldSpec(
                name=item.name,
                kind=kind,
                item=args[0] if kind is list else None,
                required=default is MISSING,
                default=default,
            )
        )
    result = tuple(specs)
    _SPECS[cls] = result
    return result


def _coerce(kind: type, value: Any) -> Any:
    """Return ``value`` as ``kind``; ints pass as floats, integral floats as ints."""
    if isinstance(value, kind) and not (kind is int and isinstance(value, bool)):
        return value
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if kind is int and isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"must be {kind.__name__}, got {type(value).__name__}")


def from_header(header: Mapping[str, Any]) -> Message:
    """Build the message for one JSON header; unknown types give :class:`Unknown`.

    ``null`` counts as absent: an optional field keeps its default.
    """
    message_type = header.get("type")
    if not isinstance(message_type, str):
        raise ProtocolError("A message needs a string 'type'")
    extra = {k: v for k, v in header.items() if k not in _RESERVED}
    cls = _REGISTRY.get(message_type)
    if cls is None:
        return Unknown(type=message_type, extra=extra)
    values: dict[str, Any] = {}
    for spec in _field_specs(cls):
        value = extra.pop(spec.name, None)
        if value is None:
            if spec.required:
                raise ProtocolError(f"{message_type} needs '{spec.name}'")
            continue
        try:
            value = _coerce(spec.kind, value)
            if spec.item is not None:
                value = [_coerce(spec.item, element) for element in value]
        except TypeError as error:
            raise ProtocolError(f"{message_type}.{spec.name} {error}") from error
        values[spec.name] = value
    return cls(extra=extra, **values)


def check_api(api: str | None) -> None:
    """Raise a ``version`` :class:`ProtocolError` unless ``api`` is :data:`PROTOCOL`."""
    if api != PROTOCOL:
        raise ProtocolError(
            f"Unsupported protocol {api!r}; this host speaks {PROTOCOL}",
            code="version",
        )


## JSON Schema (schema.json beside this module is generated from here)

_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}

SCHEMA_PATH = Path(__file__).with_name("schema.json")


def json_schema() -> dict[str, Any]:
    """Return a JSON Schema (2020-12) of every version-0 header."""
    definitions: dict[str, Any] = {}
    for cls in MESSAGE_TYPES:
        properties: dict[str, Any] = {
            "type": {"const": cls.TYPE},
            "bytes": {"type": "integer", "minimum": 0},
        }
        required = ["type"]
        for spec in _field_specs(cls):
            schema: dict[str, Any] = {"type": _JSON_TYPES[spec.kind]}
            if spec.item is not None:
                schema["items"] = {"type": _JSON_TYPES[spec.item]}
            if spec.required:
                required.append(spec.name)
            else:
                schema["type"] = [schema["type"], "null"]
                if spec.default is not None:
                    schema["default"] = spec.default
            properties[spec.name] = schema
        definitions[cls.TYPE] = {
            "type": "object",
            "description": f"Sent by: {cls.SENDER}.",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }
    unknown = {
        "type": "object",
        "description": "A type this version does not define; receivers ignore it.",
        "properties": {"type": {"type": "string", "not": {"enum": list(definitions)}}},
        "required": ["type"],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{PROTOCOL}/schema.json",
        "title": f"External-engine protocol {PROTOCOL}: message headers",
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in definitions] + [unknown],
        "$defs": definitions,
    }


def write_schema() -> None:
    """Regenerate ``schema.json`` after a message type changes."""
    SCHEMA_PATH.write_text(json.dumps(json_schema(), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "MESSAGE_TYPES",
    "PROTOCOL",
    "REFUSAL_CODES",
    "Cancel",
    "Config",
    "End",
    "Error",
    "Frame",
    "FrameReq",
    "GenFrame",
    "Hello",
    "Hold",
    "Input",
    "Message",
    "Ping",
    "Pong",
    "ProtocolError",
    "Release",
    "Restart",
    "SeedReq",
    "Start",
    "Status",
    "Unknown",
    "ViewFrame",
    "Views",
    "check_api",
    "from_header",
    "json_schema",
]
