# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FRAME_ENCODING_RAW_RGB: _ClassVar[FrameEncoding]
    FRAME_ENCODING_JPEG: _ClassVar[FrameEncoding]
FRAME_ENCODING_RAW_RGB: FrameEncoding
FRAME_ENCODING_JPEG: FrameEncoding

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("ready", "device", "model_name", "active_sessions")
    READY_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_SESSIONS_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    device: str
    model_name: str
    active_sessions: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, ready: bool = ..., device: _Optional[str] = ..., model_name: _Optional[str] = ..., active_sessions: _Optional[_Iterable[str]] = ...) -> None: ...

class StartSessionRequest(_message.Message):
    __slots__ = ("session_id", "input_height", "input_width", "scale", "sparse_ratio")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    INPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SPARSE_RATIO_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    input_height: int
    input_width: int
    scale: int
    sparse_ratio: float
    def __init__(self, session_id: _Optional[str] = ..., input_height: _Optional[int] = ..., input_width: _Optional[int] = ..., scale: _Optional[int] = ..., sparse_ratio: _Optional[float] = ...) -> None: ...

class StartSessionResponse(_message.Message):
    __slots__ = ("session_id", "success", "error")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    success: bool
    error: str
    def __init__(self, session_id: _Optional[str] = ..., success: bool = ..., error: _Optional[str] = ...) -> None: ...

class EndSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class EndSessionResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class UpscaleChunkRequest(_message.Message):
    __slots__ = ("session_id", "input_height", "input_width", "scale", "sparse_ratio", "frames_rgb", "num_frames", "height", "width", "chunk_index", "frame_encoding", "frames_jpeg", "display_only")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    INPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SPARSE_RATIO_FIELD_NUMBER: _ClassVar[int]
    FRAMES_RGB_FIELD_NUMBER: _ClassVar[int]
    NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    FRAME_ENCODING_FIELD_NUMBER: _ClassVar[int]
    FRAMES_JPEG_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_ONLY_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    input_height: int
    input_width: int
    scale: int
    sparse_ratio: float
    frames_rgb: bytes
    num_frames: int
    height: int
    width: int
    chunk_index: int
    frame_encoding: FrameEncoding
    frames_jpeg: _containers.RepeatedScalarFieldContainer[bytes]
    display_only: bool
    def __init__(self, session_id: _Optional[str] = ..., input_height: _Optional[int] = ..., input_width: _Optional[int] = ..., scale: _Optional[int] = ..., sparse_ratio: _Optional[float] = ..., frames_rgb: _Optional[bytes] = ..., num_frames: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., chunk_index: _Optional[int] = ..., frame_encoding: _Optional[_Union[FrameEncoding, str]] = ..., frames_jpeg: _Optional[_Iterable[bytes]] = ..., display_only: bool = ...) -> None: ...

class UpscaleChunkResponse(_message.Message):
    __slots__ = ("session_id", "frames_rgb", "num_frames", "height", "width", "chunk_index", "elapsed_ms", "error", "frames_omitted")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    FRAMES_RGB_FIELD_NUMBER: _ClassVar[int]
    NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    FRAMES_OMITTED_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    frames_rgb: bytes
    num_frames: int
    height: int
    width: int
    chunk_index: int
    elapsed_ms: float
    error: str
    frames_omitted: bool
    def __init__(self, session_id: _Optional[str] = ..., frames_rgb: _Optional[bytes] = ..., num_frames: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., chunk_index: _Optional[int] = ..., elapsed_ms: _Optional[float] = ..., error: _Optional[str] = ..., frames_omitted: bool = ...) -> None: ...
