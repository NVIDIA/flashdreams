# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External-engine API, version 0.

Every name exported here is stable within version 0 (``api/STABILITY.md``).
It combines the existing in-process engine contract (the rollout, the engine
step and the scene, trajectory and command types, re-exported unchanged) with
the external-engine layer: the wire messages and framing (``PROTOCOL.md``),
the tick and chunk timeline, decoded engine state, chunk assembly, the
``RolloutEngine`` for an external engine, the host-side session driver, the
TCP link, and the packet recorder and replay. The stub engine is in
:mod:`omnidreams_game_engine.api.v0.stub_engine`.
"""

from __future__ import annotations

from omnidreams_game_engine.api.v0 import framing, messages, units
from omnidreams_game_engine.api.v0.assembly import ChunkAssembler, ExternalChunk
from omnidreams_game_engine.api.v0.driver import (
    DriverConfig,
    DriverStep,
    EncodedFrame,
    ExternalSessionDriver,
    FrameEncoder,
    RawRgbEncoder,
    SceneProvider,
)
from omnidreams_game_engine.api.v0.external import (
    ChunkTimeoutError,
    EngineSource,
    ExternalEngineSession,
    SeedFrame,
    SourceEvent,
)
from omnidreams_game_engine.api.v0.link_tcp import TcpEngineLink
from omnidreams_game_engine.api.v0.messages import PROTOCOL, ProtocolError
from omnidreams_game_engine.api.v0.recorder import (
    PacketLogReader,
    PacketRecord,
    PacketRecorder,
)
from omnidreams_game_engine.api.v0.replay import PacketLogSource
from omnidreams_game_engine.api.v0.source import MessageEngineSource
from omnidreams_game_engine.api.v0.state import (
    FLU_STATE_FORMAT,
    ActorState,
    FluStateCodec,
    FrameState,
    StateDecoder,
)
from omnidreams_game_engine.api.v0.timeline import ChunkTimeline
from omnidreams_game_engine.contracts import (
    ConditionRenderer,
    GameRules,
    GameUpdate,
    SimulationWorld,
)
from omnidreams_game_engine.engine import EngineStep, GameEngine
from omnidreams_game_engine.model import (
    EngineFactory,
    RolloutEngine,
    VideoPostprocessor,
    WorldModelRollout,
    WorldModelStep,
)
from omnidreams_game_engine.types import (
    CameraCalibration,
    ConditionBatch,
    DriverCommand,
    DynamicActorTrajectory,
    SceneDefinition,
    TrajectoryChunk,
    VehicleState,
)

API_VERSION = "0.2"
"""Version of this API: the major is the protocol major, the minor counts additions."""

__all__ = [
    "API_VERSION",
    "FLU_STATE_FORMAT",
    "PROTOCOL",
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
    "FluStateCodec",
    "FrameEncoder",
    "FrameState",
    "GameEngine",
    "GameRules",
    "GameUpdate",
    "MessageEngineSource",
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
