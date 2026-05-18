from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Quat(_message.Message):
    __slots__ = ("w", "x", "y", "z")
    W_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    w: float
    x: float
    y: float
    z: float
    def __init__(self, w: _Optional[float] = ..., x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Vec3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Pose(_message.Message):
    __slots__ = ("vec", "quat")
    VEC_FIELD_NUMBER: _ClassVar[int]
    QUAT_FIELD_NUMBER: _ClassVar[int]
    vec: Vec3
    quat: Quat
    def __init__(self, vec: _Optional[_Union[Vec3, _Mapping]] = ..., quat: _Optional[_Union[Quat, _Mapping]] = ...) -> None: ...

class DynamicState(_message.Message):
    __slots__ = ("angular_velocity", "linear_velocity")
    ANGULAR_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    LINEAR_VELOCITY_FIELD_NUMBER: _ClassVar[int]
    angular_velocity: Vec3
    linear_velocity: Vec3
    def __init__(self, angular_velocity: _Optional[_Union[Vec3, _Mapping]] = ..., linear_velocity: _Optional[_Union[Vec3, _Mapping]] = ...) -> None: ...

class AABB(_message.Message):
    __slots__ = ("size_x", "size_y", "size_z")
    SIZE_X_FIELD_NUMBER: _ClassVar[int]
    SIZE_Y_FIELD_NUMBER: _ClassVar[int]
    SIZE_Z_FIELD_NUMBER: _ClassVar[int]
    size_x: float
    size_y: float
    size_z: float
    def __init__(self, size_x: _Optional[float] = ..., size_y: _Optional[float] = ..., size_z: _Optional[float] = ...) -> None: ...

class PoseAtTime(_message.Message):
    __slots__ = ("pose", "timestamp_us")
    POSE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    pose: Pose
    timestamp_us: int
    def __init__(self, pose: _Optional[_Union[Pose, _Mapping]] = ..., timestamp_us: _Optional[int] = ...) -> None: ...

class StateAtTime(_message.Message):
    __slots__ = ("timestamp_us", "pose", "state")
    TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    timestamp_us: int
    pose: Pose
    state: DynamicState
    def __init__(self, timestamp_us: _Optional[int] = ..., pose: _Optional[_Union[Pose, _Mapping]] = ..., state: _Optional[_Union[DynamicState, _Mapping]] = ...) -> None: ...

class Trajectory(_message.Message):
    __slots__ = ("poses",)
    POSES_FIELD_NUMBER: _ClassVar[int]
    poses: _containers.RepeatedCompositeFieldContainer[PoseAtTime]
    def __init__(self, poses: _Optional[_Iterable[_Union[PoseAtTime, _Mapping]]] = ...) -> None: ...

class VersionId(_message.Message):
    __slots__ = ("version_id", "git_hash", "grpc_api_version")
    class APIVersion(_message.Message):
        __slots__ = ("major", "minor", "patch")
        MAJOR_FIELD_NUMBER: _ClassVar[int]
        MINOR_FIELD_NUMBER: _ClassVar[int]
        PATCH_FIELD_NUMBER: _ClassVar[int]
        major: int
        minor: int
        patch: int
        def __init__(self, major: _Optional[int] = ..., minor: _Optional[int] = ..., patch: _Optional[int] = ...) -> None: ...
    VERSION_ID_FIELD_NUMBER: _ClassVar[int]
    GIT_HASH_FIELD_NUMBER: _ClassVar[int]
    GRPC_API_VERSION_FIELD_NUMBER: _ClassVar[int]
    version_id: str
    git_hash: str
    grpc_api_version: VersionId.APIVersion
    def __init__(self, version_id: _Optional[str] = ..., git_hash: _Optional[str] = ..., grpc_api_version: _Optional[_Union[VersionId.APIVersion, _Mapping]] = ...) -> None: ...

class SessionRequestStatus(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class AvailableScenesReturn(_message.Message):
    __slots__ = ("scene_ids",)
    SCENE_IDS_FIELD_NUMBER: _ClassVar[int]
    scene_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, scene_ids: _Optional[_Iterable[str]] = ...) -> None: ...
