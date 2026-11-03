# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Game-engine API version 0: the in-process contract and the external-engine layer."""

from __future__ import annotations

from omnidreams_game_engine.api.v0 import framing, messages, units
from omnidreams_game_engine.api.v0.assembly import ChunkAssembler, ExternalChunk
from omnidreams_game_engine.api.v0.driver import (
    DriverConfig,
    DriverState,
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
    SourceEventKind,
)
from omnidreams_game_engine.api.v0.link_tcp import TcpEngineLink
from omnidreams_game_engine.api.v0.messages import PROTOCOL, ProtocolError
from omnidreams_game_engine.api.v0.recorder import (
    PacketLogReader,
    PacketRecord,
    PacketRecorder,
)
from omnidreams_game_engine.api.v0.replay import PacketLogSource
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

__all__ = [
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
    "DriverState",
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
    "SourceEventKind",
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
