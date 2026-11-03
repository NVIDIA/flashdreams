# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the version-0 message types."""

from __future__ import annotations

import json

import pytest
from omnidreams_game_engine.api import v0
from omnidreams_game_engine.api.v0 import messages as m

pytestmark = pytest.mark.ci_cpu


def _examples() -> list[m.Message]:
    return [
        m.Hello(
            session="engine-session-1",
            api=m.PROTOCOL,
            state_format="flu/0",
            level="demo_city",
            layout_crc32=305419896,
            driver="human",
            sim_hz=60,
            tick=7,
            front_mount={"x_cm": 0.0, "y_cm": 0.0, "z_cm": 140.0, "pitch_deg": 0.0},
            wide_seed={"width": 1920, "height": 1080, "fov_deg": 120.0},
            engine_build="abc123",
            t_engine_us=1,
        ),
        m.Start(tick=12),
        m.Frame(
            epoch=3,
            g=0,
            tick=12,
            state={"rig_to_world": [1.0] * 16},
            seed=True,
            encoding="jpeg",
            width=1920,
            height=1080,
        ),
        m.Frame(epoch=3, g=9, tick=30, state={"a": 1}),
        m.Pong(t_host_us=5, t_engine_us=9),
        m.Config(
            session="s",
            api=m.PROTOCOL,
            model_fps=30,
            frame_ticks=2,
            chunk_plan=[5, 8],
            lead_chunks=2,
            display_delay_ticks=28,
            return_encoding="jpeg",
            return_width=1024,
            return_height=560,
            return_quality=90,
            render="always",
        ),
        m.SeedReq(epoch=4, tick=100, reason="restart", width=1920, height=1080),
        m.FrameReq(epoch=4, chunk=1, index=0, g=5, tick=110),
        m.GenFrame(
            epoch=4, chunk=1, g=5, tick=110, encoding="jpeg", width=1024, height=560
        ),
        m.Input(action="throttle", phase="down", t_host_us=3),
        m.Views(views=["map"], t_engine_us=4),
        m.ViewFrame(
            view="map",
            epoch=1,
            chunk=0,
            g=3,
            tick=6,
            encoding="jpeg",
            width=1024,
            height=560,
            t_host_us=5,
        ),
        m.Cancel(epoch=4),
        m.Ping(t_host_us=11),
        m.Hold(reason="pause", tick=200),
        m.Release(reason="resume", tick=200),
        m.Restart(reason="reset", tick=210),
        m.End(reason="title"),
        m.Status(tick=3, extra={"ticks_per_s": 60.0}),
        m.Error(code="busy", message="another engine is connected"),
    ]


def test_every_message_type_has_an_example_and_round_trips() -> None:
    examples = _examples()
    assert {type(message) for message in examples} == set(m.MESSAGE_TYPES)
    for message in examples:
        header = json.loads(json.dumps(message.to_header()))
        assert header["type"] == message.TYPE
        assert m.from_header(header) == message


def test_unknown_fields_are_kept_and_written_back() -> None:
    header = {
        "type": "frame",
        "epoch": 1,
        "g": 2,
        "tick": 4,
        "state": {},
        "gate_ms": 0.5,
        "bytes": 0,
    }
    frame = m.from_header(header)
    assert isinstance(frame, m.Frame)
    assert frame.extra == {"gate_ms": 0.5}
    assert frame.to_header()["gate_ms"] == 0.5
    assert "bytes" not in frame.to_header()


def test_unknown_types_are_returned_as_unknown_messages() -> None:
    message = m.from_header({"type": "teleport", "x": 1})
    assert isinstance(message, m.Unknown)
    assert message.to_header() == {"type": "teleport", "x": 1}


def test_required_fields_are_enforced() -> None:
    with pytest.raises(m.ProtocolError, match="needs 'tick'"):
        m.from_header({"type": "frame", "epoch": 1, "g": 2})
    with pytest.raises(m.ProtocolError, match="needs 'session'"):
        m.from_header({"type": "hello"})


@pytest.mark.parametrize(
    ("header", "error"),
    [
        (
            {"type": "frame", "epoch": "1", "g": 2, "tick": 3, "state": {}},
            "must be int",
        ),
        (
            {"type": "frame", "epoch": True, "g": 2, "tick": 3, "state": {}},
            "must be int",
        ),
        ({"type": "frame", "epoch": None, "g": 2, "tick": 3, "state": {}}, "needs"),
        ({"type": "frame", "epoch": 1, "g": 2, "tick": 3, "state": "x"}, "dict"),
        (
            {
                "type": "frame",
                "epoch": 1,
                "g": 2,
                "tick": 3,
                "state": {},
                "seed": "yes",
            },
            "bool",
        ),
        ({"type": "hello", "session": 5}, "must be str"),
        ({"type": "views", "views": [["map"]]}, "must be str"),
        ({"type": "views", "views": [{}]}, "must be str"),
    ],
)
def test_field_and_element_types_are_checked(header: dict, error: str) -> None:
    with pytest.raises(m.ProtocolError, match=error):
        m.from_header(header)


def test_integral_floats_are_ints_and_null_is_absent() -> None:
    frame = m.from_header(
        {"type": "frame", "epoch": 1.0, "g": 2, "tick": 3, "state": {}}
    )
    assert isinstance(frame, m.Frame) and frame.epoch == 1
    seed = m.from_header({"type": "seed_req", "epoch": 1, "tick": 0, "fov_deg": 136})
    assert isinstance(seed, m.SeedReq) and seed.fov_deg == 136.0
    assert m.from_header({"type": "start", "tick": 0, "reason": None}) == m.Start(
        tick=0
    )


def test_version_check_accepts_v0_and_refuses_others() -> None:
    m.check_api("omnidreams-engine/0")
    for api in (None, "", "omnidreams-engine/1", "omnidreams-engine/0.3", "other/0"):
        with pytest.raises(m.ProtocolError) as error:
            m.check_api(api)
        assert error.value.code == "version"


def test_schema_json_matches_the_message_classes() -> None:
    schema = m.json_schema()
    assert json.loads(m.SCHEMA_PATH.read_text(encoding="utf-8")) == schema, (
        "Regenerate with: uv run python -c "
        '"from omnidreams_game_engine.api.v0.messages import write_schema; write_schema()"'
    )
    assert set(schema["$defs"]) == {cls.TYPE for cls in m.MESSAGE_TYPES}
    frame = schema["$defs"]["frame"]
    assert frame["required"] == ["type", "epoch", "g", "tick", "state"]
    assert frame["properties"]["seed"] == {
        "type": ["boolean", "null"],
        "default": False,
    }
    # A type this version does not define matches the last branch only.
    assert schema["oneOf"][-1]["properties"]["type"]["not"]["enum"] == list(
        schema["$defs"]
    )


def test_the_stable_names_of_v0() -> None:
    assert v0.PROTOCOL == "omnidreams-engine/0"
    # Removing or renaming a name here breaks version 0.
    assert sorted(v0.__all__) == [
        "ActorState",
        "CameraCalibration",
        "ChunkAssembler",
        "ChunkTimeline",
        "ChunkTimeoutError",
        "ConditionBatch",
        "ConditionRenderer",
        "DriverCommand",
        "DriverConfig",
        "DriverStep",
        "DynamicActorTrajectory",
        "EncodedFrame",
        "EngineFactory",
        "EngineSource",
        "EngineStep",
        "ExternalChunk",
        "ExternalEngineSession",
        "ExternalSessionDriver",
        "FLU_STATE_FORMAT",
        "FluStateCodec",
        "FrameEncoder",
        "FrameState",
        "GameEngine",
        "GameRules",
        "GameUpdate",
        "PROTOCOL",
        "PacketLogReader",
        "PacketLogSource",
        "PacketRecord",
        "PacketRecorder",
        "ProtocolError",
        "RawRgbEncoder",
        "RolloutEngine",
        "SceneDefinition",
        "SceneProvider",
        "SeedFrame",
        "SimulationWorld",
        "SourceEvent",
        "StateDecoder",
        "TcpEngineLink",
        "TrajectoryChunk",
        "VehicleState",
        "VideoPostprocessor",
        "WorldModelRollout",
        "WorldModelStep",
        "framing",
        "messages",
        "units",
    ]
