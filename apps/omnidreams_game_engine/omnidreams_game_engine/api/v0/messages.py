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

PROTOCOL_NAME = "omnidreams-engine"
"""Protocol family named in ``hello.api`` and ``config.api``."""

PROTOCOL_MAJOR = 0
"""Major protocol version served by this module."""

PROTOCOL = f"{PROTOCOL_NAME}/{PROTOCOL_MAJOR}"
"""Value of ``hello.api`` and ``config.api`` for this version."""

ENGINE = "engine"
HOST = "host"
BOTH = "both"


class ProtocolError(ValueError):
    """A message that violates the protocol; ``code`` goes into an ``error``."""

    def __init__(self, message: str, *, code: str = "protocol") -> None:
        super().__init__(message)
        self.code = code


@dataclass(kw_only=True, slots=True)
class Message:
    """Base of every protocol message."""

    TYPE: ClassVar[str] = ""
    SENDER: ClassVar[str] = BOTH
    """Who sends it: ``engine``, ``host`` or ``both``."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Header keys this version does not define, kept verbatim."""

    def to_header(self) -> dict[str, Any]:
        """Return the JSON header (without ``bytes``; framing adds it)."""
        header: dict[str, Any] = {"type": self.TYPE}
        for name, value in self.extra.items():
            if name not in ("type", "bytes"):
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

    def to_header(self) -> dict[str, Any]:
        header = {"type": self.type}
        header.update(
            {k: v for k, v in self.extra.items() if k not in ("type", "bytes")}
        )
        return header


# --------------------------------------------------------------------------
# Engine to host
# --------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class Hello(Message):
    """First message of every connection, sent by the engine."""

    TYPE: ClassVar[str] = "hello"
    SENDER: ClassVar[str] = ENGINE

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
    SENDER: ClassVar[str] = ENGINE

    tick: int
    reason: str = "start"
    """``start`` or ``reconnect``."""


@dataclass(kw_only=True, slots=True)
class Frame(Message):
    """The engine's state at one granted model frame; the seed carries an image."""

    TYPE: ClassVar[str] = "frame"
    SENDER: ClassVar[str] = ENGINE

    epoch: int
    g: int
    """Model frame index within the epoch; 0 is the seed frame."""
    tick: int
    """Engine tick the state belongs to."""
    state: dict[str, Any] | None = None
    """Per-frame state in the vocabulary named by ``hello.state_format``."""
    chunk: int | None = None
    index: int | None = None
    req_tick: int | None = None
    seed: bool = False
    """True on the answer to a ``seed_req``; the payload is the seed image."""
    encoding: str | None = None
    """Payload encoding: ``jpeg``, ``png`` or ``raw`` (RGB24); absent without image."""
    width: int | None = None
    height: int | None = None
    t_engine_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Pong(Message):
    TYPE: ClassVar[str] = "pong"
    SENDER: ClassVar[str] = ENGINE

    t_host_us: int
    t_engine_us: int | None = None


# --------------------------------------------------------------------------
# Host to engine
# --------------------------------------------------------------------------


@dataclass(kw_only=True, slots=True)
class Config(Message):
    """The host's answer to ``hello``: the session's timing and frame return."""

    TYPE: ClassVar[str] = "config"
    SENDER: ClassVar[str] = HOST

    session: str
    api: str = PROTOCOL
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
    render: str | None = "off"
    """Engine world rendering between seeds: ``off`` or ``always``."""
    checkpoint: str | None = None
    profile: str | None = None
    preset: str | None = None
    host_build: str | None = None


@dataclass(kw_only=True, slots=True)
class SeedReq(Message):
    """Ask the engine for the seed image and state that start an epoch."""

    TYPE: ClassVar[str] = "seed_req"
    SENDER: ClassVar[str] = HOST

    epoch: int
    tick: int
    """Earliest tick; an engine already past it answers at its current tick."""
    reason: str = "start"
    width: int | None = None
    height: int | None = None
    fov_deg: float | None = None
    warmup_renders: int | None = None


@dataclass(kw_only=True, slots=True)
class FrameReq(Message):
    """Grant: the engine may simulate up to ``tick`` and answers with its state."""

    TYPE: ClassVar[str] = "frame_req"
    SENDER: ClassVar[str] = HOST

    epoch: int
    chunk: int
    index: int
    g: int
    tick: int
    condition: bool = False
    keys: list[Any] = field(default_factory=list)


@dataclass(kw_only=True, slots=True)
class GenFrame(Message):
    """One generated frame; the payload is the encoded image."""

    TYPE: ClassVar[str] = "gen_frame"
    SENDER: ClassVar[str] = HOST

    epoch: int
    chunk: int
    g: int
    tick: int
    """Engine tick whose state produced the frame."""
    encoding: str
    width: int
    height: int
    t_host_us: int | None = None
    camera: dict[str, Any] | None = None


@dataclass(kw_only=True, slots=True)
class Views(Message):
    """The host views the engine shows now; the host sends their frames (``view_frame``) while they are listed.

    Added in API 0.2. An empty list stops them; a new connection starts with
    none. A host that does not serve a view ignores it.
    """

    TYPE: ClassVar[str] = "views"
    SENDER: ClassVar[str] = ENGINE

    views: list[str] = field(default_factory=list)
    t_engine_us: int | None = None


@dataclass(kw_only=True, slots=True)
class ViewFrame(Message):
    """One frame of a host view the engine asked for with ``views``; the payload is the image.

    ``view`` names it (``map``: the conditioning frame the model generated
    the frame ``g`` from). Added in API 0.2; sent only to an engine that
    listed the view.
    """

    TYPE: ClassVar[str] = "view_frame"
    SENDER: ClassVar[str] = HOST

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
    SENDER: ClassVar[str] = HOST

    action: str
    phase: str
    """``down`` or ``up``."""
    t_host_us: int | None = None


@dataclass(kw_only=True, slots=True)
class Cancel(Message):
    """Drop the outstanding grants (of ``epoch`` when given)."""

    TYPE: ClassVar[str] = "cancel"
    SENDER: ClassVar[str] = HOST

    epoch: int | None = None


@dataclass(kw_only=True, slots=True)
class Ping(Message):
    TYPE: ClassVar[str] = "ping"
    SENDER: ClassVar[str] = HOST

    t_host_us: int


# --------------------------------------------------------------------------
# Both directions
# --------------------------------------------------------------------------


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

REGISTRY: dict[str, type[Message]] = {cls.TYPE: cls for cls in MESSAGE_TYPES}


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _FieldSpec:
    name: str
    kinds: tuple[type, ...]
    optional: bool
    required: bool


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
        args = typing.get_args(annotation)
        origin = typing.get_origin(annotation)
        optional = False
        if origin is Union or isinstance(annotation, types.UnionType):
            optional = type(None) in args
            members = [arg for arg in args if arg is not type(None)]
            if len(members) != 1:
                raise TypeError(f"{cls.__name__}.{item.name}: unsupported union")
            annotation = members[0]
            origin = typing.get_origin(annotation)
        base = origin if origin is not None else annotation
        if base is Any:
            kinds: tuple[type, ...] = (object,)
        else:
            kinds = (base,)
        required = item.default is MISSING and item.default_factory is MISSING
        specs.append(_FieldSpec(item.name, kinds, optional, required))
    result = tuple(specs)
    _SPECS[cls] = result
    return result


def _coerce(cls: type[Message], spec: _FieldSpec, value: Any) -> Any:
    if value is None:
        if spec.optional:
            return None
        raise ProtocolError(f"{cls.TYPE}.{spec.name} must not be null")
    kind = spec.kinds[0]
    if kind is object:
        return value
    if kind is bool:
        if isinstance(value, bool):
            return value
    elif kind is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif kind is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(value, kind):
        return value
    raise ProtocolError(
        f"{cls.TYPE}.{spec.name} must be {kind.__name__}, got {type(value).__name__}"
    )


def from_header(header: Mapping[str, Any]) -> Message:
    """Build the message for one JSON header; unknown types give :class:`Unknown`."""
    message_type = header.get("type")
    if not isinstance(message_type, str):
        raise ProtocolError("A message needs a string 'type'")
    cls = REGISTRY.get(message_type)
    if cls is None:
        return Unknown(
            type=message_type,
            extra={k: v for k, v in header.items() if k not in ("type", "bytes")},
        )
    specs = _field_specs(cls)
    known = {spec.name for spec in specs}
    values: dict[str, Any] = {}
    for spec in specs:
        if spec.name in header:
            values[spec.name] = _coerce(cls, spec, header[spec.name])
        elif spec.required:
            raise ProtocolError(f"{message_type} needs '{spec.name}'")
    extra = {
        k: v for k, v in header.items() if k not in known and k not in ("type", "bytes")
    }
    return cls(extra=extra, **values)


def parse_api(api: str | None) -> tuple[str, int]:
    """Return the protocol family and major version of an ``api`` value."""
    if not api or "/" not in api:
        raise ProtocolError(
            f"Unsupported protocol {api!r}; this host speaks {PROTOCOL}",
            code="version",
        )
    family, _, version = api.partition("/")
    major_text = version.split(".", 1)[0]
    if not major_text.isdigit():
        raise ProtocolError(f"Unreadable protocol version {api!r}", code="version")
    return family, int(major_text)


def check_api(api: str | None) -> None:
    """Raise a ``version`` :class:`ProtocolError` unless ``api`` is version 0."""
    family, major = parse_api(api)
    if family != PROTOCOL_NAME or major != PROTOCOL_MAJOR:
        raise ProtocolError(
            f"Unsupported protocol {api!r}; this host speaks {PROTOCOL}",
            code="version",
        )


# --------------------------------------------------------------------------
# JSON Schema (schema.json beside this module is generated from here)
# --------------------------------------------------------------------------

_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


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
            kind = spec.kinds[0]
            if kind is object:
                schema: dict[str, Any] = {}
            else:
                names = [_JSON_TYPES[kind]]
                if spec.optional:
                    names.append("null")
                schema = {"type": names[0] if len(names) == 1 else names}
            properties[spec.name] = schema
            if spec.required:
                required.append(spec.name)
        definitions[cls.TYPE] = {
            "type": "object",
            "description": f"Sent by: {cls.SENDER}.",
            "properties": properties,
            "required": required,
            "additionalProperties": True,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{PROTOCOL}/schema.json",
        "title": f"External-engine protocol {PROTOCOL}: message headers",
        "oneOf": [{"$ref": f"#/$defs/{name}"} for name in definitions],
        "$defs": definitions,
    }


SCHEMA_PATH = Path(__file__).with_name("schema.json")


def schema_text() -> str:
    """Return the text of ``schema.json`` as generated from this module."""
    return json.dumps(json_schema(), indent=2) + "\n"


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    """Regenerate ``schema.json`` after a message type changes."""
    path.write_text(schema_text(), encoding="utf-8")
    return path


__all__ = [
    "BOTH",
    "ENGINE",
    "HOST",
    "MESSAGE_TYPES",
    "PROTOCOL",
    "PROTOCOL_MAJOR",
    "PROTOCOL_NAME",
    "REGISTRY",
    "SCHEMA_PATH",
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
    "parse_api",
    "schema_text",
    "write_schema",
]
