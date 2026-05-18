from __future__ import annotations

import pytest
from alpadreams.grpc import server
from alpadreams.grpc.protos import common_pb2, video_model_pb2
from alpadreams.grpc.server import WorldModelService

pytestmark = pytest.mark.ci_gpu


class _DummyWrapper:
    frame_chunk_size = 4
    initial_frame_chunk_size = 8


class _DummyEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.sessions: dict[str, object] = {}
        self.conditioning_wrapper = _DummyWrapper()
        self.seed_for_every_rollout_default = 42
        self.n_cameras = 1

    def _cleanup_session(self, session_id: str) -> None:
        self.calls.append(("cleanup", session_id))

    def open_session_on_all_ranks(self, payload=None) -> None:
        self.calls.append(("open", payload))

    def render_video_chunk_all_ranks(self, payload=None):
        self.calls.append(("render", payload))
        return {"ok": True}

    def finalize_kv_cache_all_ranks(self, session_id=None) -> None:
        self.calls.append(("finalize", session_id))

    def close_session_all_ranks(self, session_id=None) -> None:
        self.calls.append(("close", session_id))


class _DummyRecorder:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_service_delegates_runtime_operations_to_engine() -> None:
    engine = _DummyEngine()
    service = WorldModelService(engine=engine)  # ty:ignore[invalid-argument-type]

    service.open_session_on_all_ranks("payload")  # ty:ignore[invalid-argument-type]
    render_result = service.render_video_chunk_all_ranks("payload")  # ty:ignore[invalid-argument-type]
    service.finalize_kv_cache_all_ranks("session")
    service.close_session_all_ranks("session")

    assert render_result == {"ok": True}
    assert engine.calls[:4] == [
        ("open", "payload"),
        ("render", "payload"),
        ("finalize", "session"),
        ("close", "session"),
    ]


def test_service_cleanup_closes_recorder_and_engine_session() -> None:
    engine = _DummyEngine()
    service = WorldModelService(engine=engine)  # ty:ignore[invalid-argument-type]
    recorder = _DummyRecorder()
    service.recorders["session-a"] = recorder  # ty:ignore[invalid-assignment]

    service._cleanup_session("session-a")

    assert recorder.closed is True
    assert "session-a" not in service.recorders
    assert ("cleanup", "session-a") in engine.calls


def test_rig_to_camera_must_match_camera_specs() -> None:
    request = video_model_pb2.SessionRequest(
        rig_to_camera=[
            common_pb2.Pose(
                vec=common_pb2.Vec3(x=0.0, y=0.0, z=0.0),
                quat=common_pb2.Quat(w=1.0, x=0.0, y=0.0, z=0.0),
            )
        ]
    )

    with pytest.raises(ValueError, match="one Pose per camera_spec"):
        server._parse_rig_to_camera_transforms(request, ["front", "rear"])


def test_rig_to_camera_rejects_default_pose() -> None:
    request = video_model_pb2.SessionRequest(rig_to_camera=[common_pb2.Pose()])

    with pytest.raises(ValueError, match="not a valid Pose"):
        server._parse_rig_to_camera_transforms(request, ["front"])
