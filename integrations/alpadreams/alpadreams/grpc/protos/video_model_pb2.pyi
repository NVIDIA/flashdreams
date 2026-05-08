# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from alpadreams.grpc.protos import common_pb2 as _common_pb2
from alpadreams.grpc.protos import camera_pb2 as _camera_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActorClassId(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INVALID: _ClassVar[ActorClassId]
    CAR: _ClassVar[ActorClassId]
    TRUCK: _ClassVar[ActorClassId]
    PEDESTRIAN: _ClassVar[ActorClassId]
    CYCLIST: _ClassVar[ActorClassId]
    OTHER: _ClassVar[ActorClassId]

class ImageFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNDEFINED: _ClassVar[ImageFormat]
    PNG: _ClassVar[ImageFormat]
    JPEG: _ClassVar[ImageFormat]
    JPEG2000: _ClassVar[ImageFormat]
    RGB_UINT8_PLANAR: _ClassVar[ImageFormat]
    AVC: _ClassVar[ImageFormat]
    AV1: _ClassVar[ImageFormat]
INVALID: ActorClassId
CAR: ActorClassId
TRUCK: ActorClassId
PEDESTRIAN: ActorClassId
CYCLIST: ActorClassId
OTHER: ActorClassId
UNDEFINED: ImageFormat
PNG: ImageFormat
JPEG: ImageFormat
JPEG2000: ImageFormat
RGB_UINT8_PLANAR: ImageFormat
AVC: ImageFormat
AV1: ImageFormat

class StaticWorldMap(_message.Message):
    __slots__ = ("hdmap_parquets",)
    HDMAP_PARQUETS_FIELD_NUMBER: _ClassVar[int]
    hdmap_parquets: bytes
    def __init__(self, hdmap_parquets: _Optional[bytes] = ...) -> None: ...

class SessionId(_message.Message):
    __slots__ = ("session_id", "initial_ego_pose")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INITIAL_EGO_POSE_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    initial_ego_pose: _common_pb2.Pose
    def __init__(self, session_id: _Optional[str] = ..., initial_ego_pose: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ...) -> None: ...

class DynamicActor(_message.Message):
    __slots__ = ("class_id", "bbox_dims", "trajectory")
    CLASS_ID_FIELD_NUMBER: _ClassVar[int]
    BBOX_DIMS_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_FIELD_NUMBER: _ClassVar[int]
    class_id: ActorClassId
    bbox_dims: _common_pb2.AABB
    trajectory: _common_pb2.Trajectory
    def __init__(self, class_id: _Optional[_Union[ActorClassId, str]] = ..., bbox_dims: _Optional[_Union[_common_pb2.AABB, _Mapping]] = ..., trajectory: _Optional[_Union[_common_pb2.Trajectory, _Mapping]] = ...) -> None: ...

class DynamicWorldState(_message.Message):
    __slots__ = ("actors",)
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    actors: _containers.RepeatedCompositeFieldContainer[DynamicActor]
    def __init__(self, actors: _Optional[_Iterable[_Union[DynamicActor, _Mapping]]] = ...) -> None: ...

class Image(_message.Message):
    __slots__ = ("data", "format")
    DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    format: ImageFormat
    def __init__(self, data: _Optional[bytes] = ..., format: _Optional[_Union[ImageFormat, str]] = ...) -> None: ...

class TextPrompt(_message.Message):
    __slots__ = ("positive", "negative")
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    positive: str
    negative: str
    def __init__(self, positive: _Optional[str] = ..., negative: _Optional[str] = ...) -> None: ...

class DebugOptions(_message.Message):
    __slots__ = ("return_hdmap_frames", "return_bev_map", "skip_video_generation", "bev_height_m", "bev_fov_deg")
    RETURN_HDMAP_FRAMES_FIELD_NUMBER: _ClassVar[int]
    RETURN_BEV_MAP_FIELD_NUMBER: _ClassVar[int]
    SKIP_VIDEO_GENERATION_FIELD_NUMBER: _ClassVar[int]
    BEV_HEIGHT_M_FIELD_NUMBER: _ClassVar[int]
    BEV_FOV_DEG_FIELD_NUMBER: _ClassVar[int]
    return_hdmap_frames: bool
    return_bev_map: bool
    skip_video_generation: bool
    bev_height_m: float
    bev_fov_deg: float
    def __init__(self, return_hdmap_frames: bool = ..., return_bev_map: bool = ..., skip_video_generation: bool = ..., bev_height_m: _Optional[float] = ..., bev_fov_deg: _Optional[float] = ...) -> None: ...

class SessionRequest(_message.Message):
    __slots__ = ("static_world_map", "text_prompt", "start_frame_offset", "debug_options", "camera_specs", "initial_frames", "random_seed")
    STATIC_WORLD_MAP_FIELD_NUMBER: _ClassVar[int]
    TEXT_PROMPT_FIELD_NUMBER: _ClassVar[int]
    START_FRAME_OFFSET_FIELD_NUMBER: _ClassVar[int]
    DEBUG_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_SPECS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_FRAMES_FIELD_NUMBER: _ClassVar[int]
    RANDOM_SEED_FIELD_NUMBER: _ClassVar[int]
    static_world_map: StaticWorldMap
    text_prompt: TextPrompt
    start_frame_offset: int
    debug_options: DebugOptions
    camera_specs: _containers.RepeatedCompositeFieldContainer[_camera_pb2.CameraSpec]
    initial_frames: _containers.RepeatedCompositeFieldContainer[Image]
    random_seed: int
    def __init__(self, static_world_map: _Optional[_Union[StaticWorldMap, _Mapping]] = ..., text_prompt: _Optional[_Union[TextPrompt, _Mapping]] = ..., start_frame_offset: _Optional[int] = ..., debug_options: _Optional[_Union[DebugOptions, _Mapping]] = ..., camera_specs: _Optional[_Iterable[_Union[_camera_pb2.CameraSpec, _Mapping]]] = ..., initial_frames: _Optional[_Iterable[_Union[Image, _Mapping]]] = ..., random_seed: _Optional[int] = ...) -> None: ...

class SessionCloseRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class VideoChunkRequest(_message.Message):
    __slots__ = ("session_id", "rig_trajectory", "dynamic_state")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    RIG_TRAJECTORY_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_STATE_FIELD_NUMBER: _ClassVar[int]
    session_id: SessionId
    rig_trajectory: _common_pb2.Trajectory
    dynamic_state: DynamicWorldState
    def __init__(self, session_id: _Optional[_Union[SessionId, _Mapping]] = ..., rig_trajectory: _Optional[_Union[_common_pb2.Trajectory, _Mapping]] = ..., dynamic_state: _Optional[_Union[DynamicWorldState, _Mapping]] = ...) -> None: ...

class RoadState(_message.Message):
    __slots__ = ("ground_z", "pitch", "roll", "on_road")
    GROUND_Z_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    ROLL_FIELD_NUMBER: _ClassVar[int]
    ON_ROAD_FIELD_NUMBER: _ClassVar[int]
    ground_z: float
    pitch: float
    roll: float
    on_road: bool
    def __init__(self, ground_z: _Optional[float] = ..., pitch: _Optional[float] = ..., roll: _Optional[float] = ..., on_road: bool = ...) -> None: ...

class CameraOutput(_message.Message):
    __slots__ = ("camera_logical_id", "rgb_frames", "hdmap_condition_frames")
    CAMERA_LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    RGB_FRAMES_FIELD_NUMBER: _ClassVar[int]
    HDMAP_CONDITION_FRAMES_FIELD_NUMBER: _ClassVar[int]
    camera_logical_id: str
    rgb_frames: _containers.RepeatedCompositeFieldContainer[Image]
    hdmap_condition_frames: _containers.RepeatedCompositeFieldContainer[Image]
    def __init__(self, camera_logical_id: _Optional[str] = ..., rgb_frames: _Optional[_Iterable[_Union[Image, _Mapping]]] = ..., hdmap_condition_frames: _Optional[_Iterable[_Union[Image, _Mapping]]] = ...) -> None: ...

class VideoChunkReturn(_message.Message):
    __slots__ = ("poses_and_timestamps_of_frames", "road_states", "camera_outputs", "bev_map_frames")
    POSES_AND_TIMESTAMPS_OF_FRAMES_FIELD_NUMBER: _ClassVar[int]
    ROAD_STATES_FIELD_NUMBER: _ClassVar[int]
    CAMERA_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    BEV_MAP_FRAMES_FIELD_NUMBER: _ClassVar[int]
    poses_and_timestamps_of_frames: _common_pb2.Trajectory
    road_states: _containers.RepeatedCompositeFieldContainer[RoadState]
    camera_outputs: _containers.RepeatedCompositeFieldContainer[CameraOutput]
    bev_map_frames: _containers.RepeatedCompositeFieldContainer[Image]
    def __init__(self, poses_and_timestamps_of_frames: _Optional[_Union[_common_pb2.Trajectory, _Mapping]] = ..., road_states: _Optional[_Iterable[_Union[RoadState, _Mapping]]] = ..., camera_outputs: _Optional[_Iterable[_Union[CameraOutput, _Mapping]]] = ..., bev_map_frames: _Optional[_Iterable[_Union[Image, _Mapping]]] = ...) -> None: ...

class StartSessionEntry(_message.Message):
    __slots__ = ("request", "response")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    request: SessionRequest
    response: SessionId
    def __init__(self, request: _Optional[_Union[SessionRequest, _Mapping]] = ..., response: _Optional[_Union[SessionId, _Mapping]] = ...) -> None: ...

class RenderVideoChunkEntry(_message.Message):
    __slots__ = ("request", "response")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    request: VideoChunkRequest
    response: VideoChunkReturn
    def __init__(self, request: _Optional[_Union[VideoChunkRequest, _Mapping]] = ..., response: _Optional[_Union[VideoChunkReturn, _Mapping]] = ...) -> None: ...

class CloseSessionEntry(_message.Message):
    __slots__ = ("request", "response")
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    request: SessionCloseRequest
    response: _common_pb2.Empty
    def __init__(self, request: _Optional[_Union[SessionCloseRequest, _Mapping]] = ..., response: _Optional[_Union[_common_pb2.Empty, _Mapping]] = ...) -> None: ...

class LogEntry(_message.Message):
    __slots__ = ("seq", "timestamp_ns", "duration_ns", "start_session", "render_video_chunk", "close_session")
    SEQ_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    DURATION_NS_FIELD_NUMBER: _ClassVar[int]
    START_SESSION_FIELD_NUMBER: _ClassVar[int]
    RENDER_VIDEO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    CLOSE_SESSION_FIELD_NUMBER: _ClassVar[int]
    seq: int
    timestamp_ns: int
    duration_ns: int
    start_session: StartSessionEntry
    render_video_chunk: RenderVideoChunkEntry
    close_session: CloseSessionEntry
    def __init__(self, seq: _Optional[int] = ..., timestamp_ns: _Optional[int] = ..., duration_ns: _Optional[int] = ..., start_session: _Optional[_Union[StartSessionEntry, _Mapping]] = ..., render_video_chunk: _Optional[_Union[RenderVideoChunkEntry, _Mapping]] = ..., close_session: _Optional[_Union[CloseSessionEntry, _Mapping]] = ...) -> None: ...
