# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isaac Teleop sidecar transport and input-source lifecycle."""

from __future__ import annotations

import asyncio
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from flashdreams_controller_input import (
    cloudxr_client_identity,
    frames,
    isaac_teleop_bridge,
    isaac_teleop_source,
)
from flashdreams_controller_input.isaac_teleop_bridge import (
    ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    SUPPORTED_ISAAC_TELEOP_VERSION,
    _build_controller_pipeline,
    _build_wrist_retargeters,
    _cloudxr_context,
    controller_result_to_message,
    require_supported_version,
)
from flashdreams_controller_input.isaac_teleop_bridge import (
    build_parser as build_bridge_parser,
)
from flashdreams_controller_input.isaac_teleop_source import (
    IsaacTeleopSource,
)
from flashdreams_controller_input.pose import (
    DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    OPENXR_LOCAL_GRIP_POSE_CONVENTION,
    OPENXR_STAGE_GRIP_POSE_CONVENTION,
    WRIST_PROXY_POSE_CONVENTION,
)
from flashdreams_controller_input.quest_source import FrameStore

pytestmark = pytest.mark.ci_cpu

_TIMEOUT_S = 5.0


def _option_value(command: list[str], option: str) -> str:
    """Return an option value from a sidecar command."""
    for index, argument in enumerate(command):
        normalized = argument.replace("_", "-")
        if normalized == option:
            return command[index + 1]
        if normalized.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    raise AssertionError(f"missing {option} in {command!r}")


def _has_option(command: list[str], option: str) -> bool:
    """Report whether a boolean option appears in a sidecar command."""
    return any(argument.replace("_", "-") == option for argument in command)


def _controller_frame(index: int = 0) -> dict[str, Any]:
    """Build one bridge frame in the shared Quest controller schema."""

    def controller(x: float, *, trigger: float) -> dict[str, Any]:
        return {
            "position": [x + 0.01 * index, 1.0, -0.5],
            # A non-unit quaternion verifies that the shared validator ran.
            "orientation": [0.0, 0.0, 0.0, 2.0],
            "trigger": trigger,
            "grip": 0.25,
            "tracked": True,
        }

    return {
        "type": "xr_frame",
        "t_client": 10.0 + index,
        "controllers": {
            "left": controller(-0.2, trigger=0.1),
            "right": controller(0.2, trigger=0.4),
        },
    }


def _hello(
    *,
    protocol: int = ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
    pose_convention: str | None = WRIST_PROXY_POSE_CONVENTION,
    wrist_offset_m: object = DEFAULT_GRIP_TO_WRIST_OFFSET_M,
) -> dict[str, Any]:
    """Build the direct CloudXR/WebXR bridge protocol handshake."""
    hello: dict[str, Any] = {
        "type": "hello",
        "protocol": protocol,
        "version": "test",
        "mode": "cloudxr-webxr",
        "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
        "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
        "wrist_offset_m": wrist_offset_m,
    }
    if pose_convention is not None:
        hello["pose_convention"] = pose_convention
    return hello


class _ControllerIndex(IntEnum):
    """Minimal stand-in for Isaac Teleop's generated controller enum."""

    GRIP_POSITION = 0
    GRIP_ORIENTATION = 1
    GRIP_IS_VALID = 2
    PRIMARY_CLICK = 3
    SECONDARY_CLICK = 4
    THUMBSTICK_X = 5
    THUMBSTICK_Y = 6
    SQUEEZE_VALUE = 7
    TRIGGER_VALUE = 8


class _ControllerGroup:
    """Small duck type for Isaac Teleop's optional tensor group."""

    def __init__(self, values: dict[_ControllerIndex, Any] | None = None) -> None:
        self._values = values or {}
        self.is_none = values is None

    def __getitem__(self, index: _ControllerIndex) -> Any:
        return self._values[index]


class _FakeProcess:
    """Socket-producing stand-in for the Isaac Teleop Python child."""

    def __init__(
        self,
        command: list[str],
        messages: list[dict[str, Any]] | None,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        self.args = command
        self.returncode = returncode
        self.pid = 4242
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO(stderr)
        self.terminate_calls = 0
        self.kill_calls = 0
        self._connection: socket.socket | None = None
        self._connection_lock = threading.Lock()
        self._connected = threading.Event()
        self._stopped = threading.Event()
        self._threads: list[threading.Thread] = []
        if returncode is not None:
            return
        if messages is not None:
            frame_thread = threading.Thread(
                target=self._connect,
                args=(messages,),
                name="fake-isaac-teleop-sidecar",
                daemon=True,
            )
            self._threads.append(frame_thread)
            frame_thread.start()

    def _connect(self, messages: list[dict[str, Any]]) -> None:
        host = _option_value(self.args, "--host")
        port = int(_option_value(self.args, "--port"))
        deadline = time.monotonic() + _TIMEOUT_S
        connection: socket.socket | None = None
        while not self._stopped.is_set() and time.monotonic() < deadline:
            try:
                connection = socket.create_connection((host, port), timeout=0.2)
                break
            except OSError:
                time.sleep(0.01)
        if connection is None:
            return
        with connection:
            with self._connection_lock:
                self._connection = connection
            self._connected.set()
            for message in messages:
                self.send(message)
            self._stopped.wait()
        with self._connection_lock:
            self._connection = None

    def send(self, message: dict[str, Any]) -> None:
        """Send one NDJSON bridge message over the live fake connection."""
        if not self._connected.wait(timeout=_TIMEOUT_S):
            raise AssertionError("the fake sidecar did not connect")
        payload = (json.dumps(message) + "\n").encode("utf-8")
        with self._connection_lock:
            if self._connection is None:
                raise AssertionError("the fake sidecar connection is closed")
            self._connection.sendall(payload)

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.returncode is None:
            self.returncode = 0
        self._stopped.set()

    def send_signal(self, _signal: int) -> None:
        self.terminate()

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self._stopped.set()

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None and not self._stopped.wait(timeout):
            assert timeout is not None
            raise subprocess.TimeoutExpired(self.args, timeout)
        for thread in self._threads:
            thread.join(timeout=timeout)
        return int(self.returncode or 0)

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.wait(timeout=timeout)
        return self.stdout.getvalue(), self.stderr.getvalue()


class _FakeLauncher:
    """Record one ``Popen`` call and optionally emulate its bridge client."""

    def __init__(
        self,
        messages: list[dict[str, Any]] | None,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        self.messages = messages
        self.returncode = returncode
        self.stderr = stderr
        self.command: list[str] | None = None
        self.kwargs: dict[str, Any] = {}
        self.process: _FakeProcess | None = None

    def __call__(self, command, *args, **kwargs) -> _FakeProcess:
        assert not args
        assert self.process is None, "the source launched more than one sidecar"
        self.command = [str(argument) for argument in command]
        self.kwargs = dict(kwargs)
        self.process = _FakeProcess(
            self.command,
            self.messages,
            returncode=self.returncode,
            stderr=self.stderr,
        )
        return self.process


def _source(
    *,
    rate_hz: float = 1000.0,
    tracking_rate_hz: float = 72.0,
    wrist_offset_m: float = DEFAULT_GRIP_TO_WRIST_OFFSET_M,
    startup_timeout_s: float | None = 1.0,
    install_dir: str = "~/.cloudxr",
    env_config: str | None = None,
    device_profile: str = "Quest3",
    accept_eula: bool = True,
    host_client: bool = False,
    offline_web_client: bool = False,
    oob_hub: bool = False,
    wait_for_controllers: bool = True,
    on_message: Callable[[str | bytes], None] | None = None,
    on_disconnect: Callable[[], None] | None = None,
    cloudxr_host: str | None = None,
    cloudxr_port: int = 48322,
    cloudxr_proxy_bind: str | None = None,
) -> IsaacTeleopSource:
    """Build a fast source for CPU transport tests."""
    return IsaacTeleopSource(
        python_executable=sys.executable,
        port=0,
        rate_hz=rate_hz,
        tracking_rate_hz=tracking_rate_hz,
        wrist_offset_m=wrist_offset_m,
        startup_timeout_s=startup_timeout_s,
        install_dir=install_dir,
        env_config=env_config,
        device_profile=device_profile,
        accept_eula=accept_eula,
        host_client=host_client,
        offline_web_client=offline_web_client,
        oob_hub=oob_hub,
        wait_for_controllers=wait_for_controllers,
        on_message=on_message,
        on_disconnect=on_disconnect,
        cloudxr_host=cloudxr_host,
        cloudxr_port=cloudxr_port,
        cloudxr_proxy_bind=cloudxr_proxy_bind,
    )


def _install_launcher(monkeypatch, launcher: _FakeLauncher) -> None:
    """Replace only the child-process boundary used by the source."""
    monkeypatch.setattr(
        "flashdreams_controller_input.isaac_teleop_source.subprocess.Popen",
        launcher,
    )


def test_bridge_preserves_retargeted_wrist_pose_and_controller_inputs() -> None:
    half_turn = 2.0**0.5
    left = _ControllerGroup(
        {
            _ControllerIndex.GRIP_POSITION: [-0.2, 1.1, -0.4],
            # Non-unit +90-degree Y rotation: local +Z points along world +X.
            _ControllerIndex.GRIP_ORIENTATION: [0.0, half_turn, 0.0, half_turn],
            _ControllerIndex.GRIP_IS_VALID: True,
            _ControllerIndex.PRIMARY_CLICK: 1.0,
            _ControllerIndex.SECONDARY_CLICK: 0.0,
            _ControllerIndex.THUMBSTICK_X: -2.0,
            _ControllerIndex.THUMBSTICK_Y: 0.25,
            _ControllerIndex.SQUEEZE_VALUE: 1.5,
            _ControllerIndex.TRIGGER_VALUE: float("nan"),
        }
    )
    message = controller_result_to_message(
        {
            "controller_left": left,
            "controller_right": _ControllerGroup(),
            # The serializer must trust Isaac's retargeted output rather than
            # recomputing a pose from the raw controller tensor.
            "wrist_left": [[-0.15, 1.1, -0.4, 0.0, 0.0, 0.0, 3.0]],
            "wrist_right": [[float("nan")] * 7],
        },
        _ControllerIndex,
        capture_time=12.5,
    )

    assert message["type"] == "xr_frame"
    assert message["t_client"] == 12.5
    assert message["controllers"]["right"] is None
    payload = message["controllers"]["left"]
    assert payload["position"] == pytest.approx([-0.15, 1.1, -0.4])
    assert payload["orientation"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert payload["tracked"] is True
    assert payload["trigger"] == 0.0
    assert payload["grip"] == 1.0
    assert payload["axes"] == pytest.approx([-1.0, 0.25])
    assert payload["primary_button"] is True
    assert payload["secondary_button"] is False


def test_bridge_named_pair_retains_controls_clocks_and_source_identity(
    monkeypatch,
) -> None:
    group = _ControllerGroup(
        {
            _ControllerIndex.GRIP_IS_VALID: True,
            _ControllerIndex.PRIMARY_CLICK: 1.0,
            _ControllerIndex.SECONDARY_CLICK: 1.0,
            _ControllerIndex.THUMBSTICK_X: -0.5,
            _ControllerIndex.THUMBSTICK_Y: 0.25,
            _ControllerIndex.SQUEEZE_VALUE: 0.5,
            _ControllerIndex.TRIGGER_VALUE: 0.75,
        }
    )
    result = {
        "controller_left": group,
        "controller_right": group,
        "wrist_left": [[-0.25, 1.0, -0.5, 0.0, 0.0, 0.0, 1.0]],
        "wrist_right": [[0.25, 1.0, -0.5, 0.0, 0.0, 0.0, 1.0]],
    }
    store = FrameStore()
    for index, capture_time in enumerate((12.5, 0.5)):
        message = controller_result_to_message(
            result, _ControllerIndex, capture_time=capture_time
        )
        host_capture_ns = 1_000_000_000 + index
        monkeypatch.setattr(
            frames.time,
            "perf_counter_ns",
            lambda capture_ns=host_capture_ns: capture_ns,
        )
        store.consume(json.dumps(message))
        frame = store.snapshot()
        assert frame is not None
        assert frame["source_frame_index"] == index
        assert frame["client_capture_ns"] == host_capture_ns
        assert frame["source_client_capture_ns"] == round(capture_time * 1_000_000_000)
        for side, controller in message["controllers"].items():
            assert frame[side] == controller
    assert store.received == 2
    assert store.rejected == 0


def test_bridge_treats_an_invalid_grip_pose_as_untracked() -> None:
    values = {
        _ControllerIndex.GRIP_POSITION: [0.0, 0.0, 0.0],
        _ControllerIndex.GRIP_ORIENTATION: [0.0, 0.0, 0.0, 1.0],
        _ControllerIndex.GRIP_IS_VALID: False,
    }
    message = controller_result_to_message(
        {
            "controller_left": _ControllerGroup(values),
            "controller_right": _ControllerGroup(),
            # Se3AbsRetargeter holds its last value during tracking loss. The
            # raw validity bit must be checked before either stale pose is read.
            "wrist_left": [[float("nan")] * 7],
            "wrist_right": [[float("nan")] * 7],
        },
        _ControllerIndex,
        capture_time=1.0,
    )

    assert message["controllers"] == {"left": None, "right": None}


def test_live_source_graph_applies_one_canonical_wrist_offset_per_hand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs: list[Any] = []
    nodes: list[Any] = []

    class FakeControllersSource:
        LEFT = "left"
        RIGHT = "right"

        def __init__(self, name: str) -> None:
            assert name == "controllers"

        def output(self, side: str) -> str:
            return f"{side}-output"

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            configs.append(self)

    class FakeRetargeter:
        def __init__(self, config: Any, name: str) -> None:
            self.config = config
            self.name = name
            self.inputs: dict[str, Any] = {}
            nodes.append(self)

        def connect(self, inputs: dict[str, Any]) -> Any:
            self.inputs = inputs
            return self

        def output(self, name: str) -> Any:
            assert name == "ee_pose"
            return self

    package = ModuleType("isaacteleop")
    retargeters = ModuleType("isaacteleop.retargeters")
    se3 = ModuleType("isaacteleop.retargeters.se3_retargeter")
    monkeypatch.setattr(se3, "Se3AbsRetargeter", FakeRetargeter, raising=False)
    monkeypatch.setattr(se3, "Se3RetargeterConfig", FakeConfig, raising=False)
    monkeypatch.setattr(package, "retargeters", retargeters, raising=False)
    monkeypatch.setattr(retargeters, "se3_retargeter", se3, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.retargeters", retargeters)
    monkeypatch.setitem(sys.modules, "isaacteleop.retargeters.se3_retargeter", se3)
    monkeypatch.setitem(
        sys.modules,
        "isaacteleop.retargeting_engine.deviceio_source_nodes",
        SimpleNamespace(ControllersSource=FakeControllersSource),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacteleop.retargeting_engine.interface",
        SimpleNamespace(OutputCombiner=dict),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaacteleop.retargeting_engine.tensor_types.indices",
        SimpleNamespace(ControllerInputIndex=_ControllerIndex),
    )

    pipeline, indices = _build_controller_pipeline(wrist_offset_m=0.075)

    assert indices is _ControllerIndex
    assert pipeline == {
        "controller_left": "left-output",
        "controller_right": "right-output",
        "wrist_left": nodes[0],
        "wrist_right": nodes[1],
    }
    assert [node.name for node in nodes] == [
        "webxr_left_grip_to_wrist",
        "webxr_right_grip_to_wrist",
    ]
    assert nodes[0].inputs == {"controller_left": "left-output"}
    assert nodes[1].inputs == {"controller_right": "right-output"}
    for side, config in zip(("left", "right"), configs, strict=True):
        assert config.input_device == f"controller_{side}"
        assert config.zero_out_xy_rotation is False
        assert config.use_wrist_position is True
        assert config.use_wrist_rotation is True
        assert config.target_offset_x == 0.0
        assert config.target_offset_y == 0.0
        assert config.target_offset_z == pytest.approx(0.075)
        assert config.target_offset_roll == 0.0
        assert config.target_offset_pitch == 0.0
        assert config.target_offset_yaw == 0.0


@pytest.mark.parametrize("offset", [-0.01, float("nan"), float("inf")])
def test_shared_wrist_retargeter_rejects_an_invalid_offset(offset: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        _build_wrist_retargeters(
            {"left": object(), "right": object()},
            wrist_offset_m=offset,
            node_prefix="invalid",
        )


def test_bridge_rejects_an_unvalidated_sdk_version(monkeypatch) -> None:
    monkeypatch.setattr(
        "flashdreams_controller_input.isaac_teleop_bridge.importlib.metadata.version",
        lambda _distribution: "1.5.0",
    )

    with pytest.raises(RuntimeError, match="isaacteleop 1.5.0 is unsupported"):
        require_supported_version()

    monkeypatch.setattr(
        "flashdreams_controller_input.isaac_teleop_bridge.importlib.metadata.version",
        lambda _distribution: f"{SUPPORTED_ISAAC_TELEOP_VERSION}+local",
    )
    assert require_supported_version() == f"{SUPPORTED_ISAAC_TELEOP_VERSION}+local"


def test_bridge_can_run_as_a_standalone_script(
    tmp_path: Path,
) -> None:
    bridge = Path(isaac_teleop_bridge.__file__)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
from pathlib import Path
import runpy
import sys
original_import = builtins.__import__
def isolated_import(name, *args, **kwargs):
    if name.split('.')[0] in {
        'handdreams', 'flashdreams', 'flashdreams_controller_input',
        'torch', 'isaacteleop',
    }:
        raise AssertionError(f'unexpected application or SDK import: {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = isolated_import
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
""",
            str(bridge),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Isaac Teleop controller bridge" in result.stdout


@pytest.mark.parametrize("value", ["-0.01", "nan", "inf"])
def test_bridge_cli_rejects_an_invalid_web_wrist_offset(value: str) -> None:
    with pytest.raises(SystemExit):
        isaac_teleop_bridge.main(["serve", "--port", "8091", "--wrist-offset-m", value])


def test_source_rejects_an_invalid_wrist_offset() -> None:
    with pytest.raises(
        ValueError, match="wrist offset must be finite and non-negative"
    ):
        _source(wrist_offset_m=float("nan"))


@pytest.mark.parametrize("timeout_s", [0.0, -1.0, float("nan"), float("inf")])
def test_source_rejects_an_invalid_startup_timeout(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="startup timeout"):
        _source(startup_timeout_s=timeout_s)


def test_source_accepts_an_indefinite_startup_timeout() -> None:
    source = _source(startup_timeout_s=None)

    assert source._startup_timeout_s is None


@pytest.mark.parametrize(
    "stage,ready",
    [
        ("", False),
        ("connected", False),
        ("runtime", False),
        ("session", True),
        ("streaming", True),
        ("failed", False),
    ],
)
def test_client_readiness_preserves_frames_and_requires_a_live_compatible_bridge(
    stage, ready
):
    source = _source()
    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    setattr(source, "_process", process)
    source._consume(json.dumps({"type": "status", "stage": stage}))
    assert not source.ready_for_clients
    source._consume(json.dumps(_hello()))
    assert source.ready_for_clients is ready
    source._consume(json.dumps(_controller_frame()))
    frame = source._store.snapshot()
    counters = source._next_ns, source._last_capture_ns
    assert source.ready_for_clients is ready
    assert source._store.snapshot() == frame
    assert (source._next_ns, source._last_capture_ns) == counters

    source._consume(json.dumps(_hello(protocol=999)))
    assert not source.ready_for_clients
    source._consume(json.dumps(_hello()))
    process.returncode = 1
    assert not source.ready_for_clients
    process.returncode = None
    source._stopped = True
    assert not source.ready_for_clients
    source._stopped = False
    setattr(source, "_process", None)
    assert not source.ready_for_clients


def test_source_cannot_restart_after_stop() -> None:
    source = _source(startup_timeout_s=None)
    source.stop()

    with pytest.raises(
        RuntimeError, match="Isaac Teleop source was stopped before starting"
    ):
        source.start()


@pytest.mark.parametrize("oob_hub", [False, True])
@pytest.mark.parametrize("inherited_hub_only", [None, "caller-owned"])
@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("proxy_bind", [None, "127.0.0.1", "10.0.0.2"])
def test_bridge_constructs_the_pinned_cloudxr_launcher_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    oob_hub: bool,
    inherited_hub_only: str | None,
    offline: bool,
    proxy_bind: str | None,
) -> None:
    calls: list[dict[str, Any]] = []
    observed_hub_only: list[str | None] = []
    loaded_env_files: list[str] = []
    monkeypatch.setenv("CONTROL_TOKEN", "inherited-test-token")
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        (tmp_path / name).write_text("cached asset")
    if inherited_hub_only is None:
        monkeypatch.delenv("TELEOP_OOB_HUB_ONLY", raising=False)
    else:
        monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", inherited_hub_only)

    class FakeLauncher:
        def __init__(self, **kwargs) -> None:
            assert "CONTROL_TOKEN" not in os.environ
            calls.append(kwargs)
            observed_hub_only.append(os.environ.get("TELEOP_OOB_HUB_ONLY"))

    class FakeEnvConfig:
        @staticmethod
        def _load_env_file(path: str) -> dict[str, str]:
            loaded_env_files.append(path)
            return {}

    package = ModuleType("isaacteleop")
    cloudxr = ModuleType("isaacteleop.cloudxr")
    env_config = ModuleType("isaacteleop.cloudxr.env_config")
    wss = ModuleType("isaacteleop.cloudxr.wss")
    bindings = []

    def serve(handler, **kwargs):
        bindings.append((handler, kwargs))
        return "listener"

    setattr(wss, "ws_serve", serve)
    setattr(cloudxr, "wss", wss)
    monkeypatch.setattr(env_config, "EnvConfig", FakeEnvConfig, raising=False)
    monkeypatch.setattr(cloudxr, "CloudXRLauncher", FakeLauncher, raising=False)
    monkeypatch.setattr(package, "cloudxr", cloudxr, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.env_config", env_config)
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "_prepare_cloudxr_64k_preload",
        lambda: None,
    )

    args = build_bridge_parser().parse_args(
        [
            "serve",
            "--port",
            "8091",
            "--cloudxr-install-dir",
            "/var/tmp/cloudxr",
            "--cloudxr-env-config",
            "/var/tmp/cloudxr.env",
            "--cloudxr-device-profile",
            "Quest3",
            "--accept-eula",
            "--host-client",
            *(["--offline-web-client"] if offline else []),
            *(["--oob-hub"] if oob_hub else []),
            *(["--cloudxr-proxy-bind", proxy_bind] if proxy_bind is not None else []),
        ]
    )
    launcher = _cloudxr_context(args)

    assert isinstance(launcher, FakeLauncher)
    assert calls == [
        {
            "install_dir": "/var/tmp/cloudxr",
            "env_config": "/var/tmp/cloudxr.env",
            "device_profile": "Quest3",
            "accept_eula": True,
            "setup_oob": oob_hub,
            "usb_local": False,
            "host_client": True,
        }
    ]
    expected_hub_only = "1" if oob_hub else inherited_hub_only
    assert observed_hub_only == [expected_hub_only]
    # The SDK proxy thread reads this after launcher construction returns.
    assert os.environ.get("TELEOP_OOB_HUB_ONLY") == expected_hub_only
    assert "CONTROL_TOKEN" not in os.environ
    assert loaded_env_files == ["/var/tmp/cloudxr.env"]
    # The proxy thread may reach its factory after construction returns.
    handler, tls = object(), object()
    options = {"host": "", "port": 48322, "ssl": tls, "ping_interval": None}
    assert wss.ws_serve(handler, **options) == "listener"
    expected_options = dict(options)
    if proxy_bind is not None or offline:
        host = proxy_bind or "127.0.0.1"
        expected_options.update(
            host=host if host == "127.0.0.1" else [host, "127.0.0.1"],
            family=socket.AF_INET,
        )
    else:
        assert wss.ws_serve is serve
    assert bindings == [(handler, expected_options)]


@pytest.mark.parametrize("address", ["127.0.0.1", "127.0.0.2"])
def test_cloudxr_proxy_keeps_usb_loopback_on_the_selected_interfaces(
    monkeypatch, address
):
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    cloudxr = ModuleType("isaacteleop.cloudxr")
    wss = ModuleType("isaacteleop.cloudxr.wss")
    setattr(wss, "ws_serve", serve)
    setattr(cloudxr, "wss", wss)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    isaac_teleop_bridge._bind_cloudxr_proxy(address)

    async def echo(ws):
        await ws.send(await ws.recv())

    async def check():
        async with wss.ws_serve(echo, host="", port=0) as server:
            assert {item.getsockname()[0] for item in server.sockets} == {
                address,
                "127.0.0.1",
            }
            for listener in server.sockets:
                assert listener.family == socket.AF_INET
                host, port = listener.getsockname()
                async with connect(f"ws://{host}:{port}", proxy=None) as client:
                    await client.send("usb-proxy")
                    assert (
                        await asyncio.wait_for(client.recv(), timeout=5) == "usb-proxy"
                    )

    asyncio.run(check())
    # Other websocket servers in this interpreter retain their normal factory.
    from websockets.asyncio import server

    assert server.serve is serve


def test_cloudxr_proxy_rejects_wildcard_before_sdk_binding(monkeypatch):
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", None)
    with pytest.raises(ValueError, match="concrete IPv4 bind address"):
        isaac_teleop_bridge._bind_cloudxr_proxy("0.0.0.0")


@pytest.mark.parametrize("address", ["", "localhost", "::1", "10.0.0.256"])
def test_proxy_binding_rejects_non_ipv4_addresses(address):
    with pytest.raises(ValueError):
        _source(cloudxr_proxy_bind=address)
    with pytest.raises(SystemExit):
        build_bridge_parser().parse_args(
            ["serve", "--port", "8091", "--cloudxr-proxy-bind", address]
        )


@pytest.mark.parametrize("oob_hub", [False, True])
@pytest.mark.parametrize(
    "key,inherited,override,conflict",
    [
        ("TELEOP_OOB_HUB_ONLY", "caller-owned", value, value != "1")
        for value in ("0", "false", "", "1")
    ]
    + [
        (key, inherited, override, conflict)
        for key in (
            "TELEOP_PROXY_HOST",
            "PROXY_PORT",
            "TELEOP_STREAM_SERVER_IP",
            "TELEOP_STREAM_PORT",
        )
        for inherited, override, conflict in (
            ("parent-value", "file-value", True),
            ("same-value", "same-value", False),
            (None, "file-value", False),
        )
    ]
    + [
        ("CONTROL_TOKEN", "parent-token", "file-token", True),
        ("CONTROL_TOKEN", "same-token", "same-token", True),
        ("CONTROL_TOKEN", None, "file-token", True),
        ("CONTROL_TOKEN", None, "", True),
        ("CONTROL_TOKEN", "parent-token", "", True),
    ],
)
def test_bridge_validates_cloudxr_overrides_before_launching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    oob_hub: bool,
    key: str,
    inherited: str | None,
    override: str,
    conflict: bool,
) -> None:
    config_path = tmp_path / "cloudxr.env"
    original_config = f"{key}={override}\n"
    config_path.write_text(original_config, encoding="utf-8")
    calls: list[dict[str, Any]] = []
    prepared: list[bool] = []
    loaded: list[str] = []

    class FakeEnvConfig:
        @staticmethod
        def _load_env_file(path: str) -> dict[str, str]:
            assert path == str(config_path)
            loaded.append(path)
            return {key: override}

    class FakeLauncher:
        def __init__(self, **kwargs) -> None:
            assert "CONTROL_TOKEN" not in os.environ
            assert os.environ["TELEOP_OOB_HUB_ONLY"] == (
                "1" if oob_hub else "caller-owned"
            )
            calls.append(kwargs)

    package = ModuleType("isaacteleop")
    cloudxr = ModuleType("isaacteleop.cloudxr")
    env_config = ModuleType("isaacteleop.cloudxr.env_config")
    monkeypatch.setattr(env_config, "EnvConfig", FakeEnvConfig, raising=False)
    monkeypatch.setattr(cloudxr, "CloudXRLauncher", FakeLauncher, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.env_config", env_config)
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "_prepare_cloudxr_64k_preload",
        lambda: prepared.append(True),
    )
    for name in (
        "TELEOP_PROXY_HOST",
        "PROXY_PORT",
        "TELEOP_STREAM_SERVER_IP",
        "TELEOP_STREAM_PORT",
        "CONTROL_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", "caller-owned")
    if inherited is not None:
        monkeypatch.setenv(key, inherited)
    args = build_bridge_parser().parse_args(
        [
            "serve",
            "--port",
            "8091",
            "--accept-eula",
            *(["--oob-hub"] if oob_hub else []),
            "--cloudxr-env-config",
            str(config_path),
        ]
    )

    if not conflict or (not oob_hub and key != "CONTROL_TOKEN"):
        assert isinstance(_cloudxr_context(args), FakeLauncher)
        assert len(calls) == 1
        assert calls[0]["setup_oob"] is oob_hub
        assert calls[0]["usb_local"] is False
        assert calls[0]["env_config"] == str(config_path)
        assert prepared == [True]
    else:
        expected_error = (
            "CONTROL_TOKEN authentication is unsupported"
            if key == "CONTROL_TOKEN"
            else f"requires {key}"
        )
        with pytest.raises(ValueError, match=expected_error) as error:
            _cloudxr_context(args)
        assert calls == []
        assert prepared == []
        assert os.environ["TELEOP_OOB_HUB_ONLY"] == "caller-owned"
        if key == "CONTROL_TOKEN":
            assert not override or override not in str(error.value)
            assert inherited is None or inherited not in str(error.value)

    assert "CONTROL_TOKEN" not in os.environ
    assert loaded == [str(config_path)]
    assert config_path.read_text(encoding="utf-8") == original_config


def test_bridge_does_not_enable_a_hub_without_launching_the_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TELEOP_OOB_HUB_ONLY", raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", None)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", None)
    args = build_bridge_parser().parse_args(
        ["serve", "--port", "8091", "--oob-hub", "--no-launch-cloudxr-runtime"]
    )

    with _cloudxr_context(args) as launcher:
        assert launcher is None

    assert "TELEOP_OOB_HUB_ONLY" not in os.environ


def test_bridge_scopes_the_64k_preload_to_launcher_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preload = tmp_path / "libNvStreamServer.flashdreams-64k.so"
    preload.touch()
    observed: list[str | None] = []

    class FakeLauncher:
        def __init__(self, **_kwargs) -> None:
            observed.append(os.environ.get("LD_PRELOAD"))

    package = ModuleType("isaacteleop")
    cloudxr = ModuleType("isaacteleop.cloudxr")
    monkeypatch.setattr(cloudxr, "CloudXRLauncher", FakeLauncher, raising=False)
    monkeypatch.setattr(package, "cloudxr", cloudxr, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "_prepare_cloudxr_64k_preload",
        lambda: preload,
    )
    monkeypatch.setenv("LD_PRELOAD", "/opt/existing-preload.so")
    args = build_bridge_parser().parse_args(
        ["serve", "--port", "8091", "--accept-eula"]
    )

    _cloudxr_context(args)

    assert observed == [f"{preload} /opt/existing-preload.so"]
    assert os.environ["LD_PRELOAD"] == "/opt/existing-preload.so"


@pytest.mark.parametrize("offline", [False, True])
def test_bridge_restores_the_environment_when_cloudxr_startup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, offline: bool
) -> None:
    preload = tmp_path / "libNvStreamServer.flashdreams-64k.so"

    class FailingLauncher:
        def __init__(self, **_kwargs) -> None:
            assert os.environ["LD_PRELOAD"] == str(preload)
            raise RuntimeError("runtime failed")

    package = ModuleType("isaacteleop")
    cloudxr = ModuleType("isaacteleop.cloudxr")
    wss = ModuleType("isaacteleop.cloudxr.wss")
    setattr(wss, "ws_serve", lambda **kwargs: kwargs)
    setattr(cloudxr, "wss", wss)
    monkeypatch.setattr(cloudxr, "CloudXRLauncher", FailingLauncher, raising=False)
    monkeypatch.setattr(package, "cloudxr", cloudxr, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "_prepare_cloudxr_64k_preload",
        lambda: preload,
    )
    monkeypatch.setattr(isaac_teleop_bridge, "cached_web_client_dir", lambda: tmp_path)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    args = build_bridge_parser().parse_args(
        [
            "serve",
            "--port",
            "8091",
            "--accept-eula",
            *(["--host-client", "--offline-web-client"] if offline else []),
        ]
    )

    with pytest.raises(RuntimeError, match="runtime failed"):
        _cloudxr_context(args)

    assert "LD_PRELOAD" not in os.environ
    # A delayed SDK thread must remain restricted even after startup fails.
    assert wss.ws_serve(host="") == (
        {"host": "127.0.0.1", "family": socket.AF_INET} if offline else {"host": ""}
    )


@pytest.mark.parametrize("oob_hub", [False, True])
@pytest.mark.parametrize("proxy_bind", [None, "10.0.0.2"])
def test_source_builds_the_selected_sidecar_command(
    monkeypatch, tmp_path: Path, oob_hub: bool, proxy_bind: str | None
) -> None:
    install_dir = tmp_path / "IsaacTeleop"
    install_dir.mkdir()
    env_config = tmp_path / "teleop.yaml"
    env_config.write_text("retargeting: []\n", encoding="utf-8")
    launcher = _FakeLauncher([_hello(wrist_offset_m=0.075), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(
        install_dir=str(install_dir),
        env_config=str(env_config),
        device_profile="quest3",
        tracking_rate_hz=90.0,
        wrist_offset_m=0.075,
        accept_eula=True,
        host_client=True,
        oob_hub=oob_hub,
        cloudxr_proxy_bind=proxy_bind,
    )

    try:
        source.start()
        command = launcher.command
        assert command is not None
        assert Path(command[0]).resolve() == Path(sys.executable).resolve()
        bridge = Path(command[1])
        assert bridge.is_absolute()
        assert bridge.name == "isaac_teleop_bridge.py"
        assert command[2] == "serve"
        assert _option_value(command, "--host") == "127.0.0.1"
        assert int(_option_value(command, "--port")) > 0
        assert float(_option_value(command, "--rate-hz")) == 90.0
        assert float(_option_value(command, "--wrist-offset-m")) == pytest.approx(0.075)
        assert not _has_option(command, "--calibration-fd")
        assert "pass_fds" not in launcher.kwargs
        assert Path(_option_value(command, "--cloudxr-install-dir")) == install_dir
        assert Path(_option_value(command, "--cloudxr-env-config")) == env_config
        assert _option_value(command, "--cloudxr-device-profile") == "quest3"
        assert _has_option(command, "--accept-eula")
        assert _has_option(command, "--host-client")
        assert _has_option(command, "--oob-hub") is oob_hub
        if proxy_bind is None:
            assert not _has_option(command, "--cloudxr-proxy-bind")
        else:
            assert _option_value(command, "--cloudxr-proxy-bind") == proxy_bind
    finally:
        source.stop()


@pytest.mark.parametrize("oob_hub", [False, True])
def test_source_removes_cuda_visibility_and_control_token_from_sidecar_environment(
    monkeypatch: pytest.MonkeyPatch,
    oob_hub: bool,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CONTROL_TOKEN", "cloudxr-test-token")
    monkeypatch.setenv("FLASHDREAMS_CONTROLLER_TOKEN", "relay-test-token")
    monkeypatch.setenv("ISAAC_TELEOP_TEST_ENV", "preserved")
    monkeypatch.setenv("TELEOP_OOB_HUB_ONLY", "caller-owned")
    parent_environment = os.environ.copy()
    expected_sidecar_environment = {
        name: value
        for name, value in parent_environment.items()
        if not name.startswith("PYTHON")
    }
    expected_sidecar_environment["PYTHONNOUSERSITE"] = "1"
    expected_sidecar_environment.pop("CUDA_VISIBLE_DEVICES")
    expected_sidecar_environment.pop("CONTROL_TOKEN")
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(oob_hub=oob_hub)

    try:
        source.start()

        sidecar_environment = launcher.kwargs["env"]
        assert sidecar_environment == expected_sidecar_environment
        assert sidecar_environment["ISAAC_TELEOP_TEST_ENV"] == "preserved"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    finally:
        source.stop()

    assert os.environ == parent_environment


@pytest.mark.parametrize("cloudxr_host", [None, "laptop.local"])
def test_source_scopes_cloudxr_advertising_to_child_environment(
    monkeypatch: pytest.MonkeyPatch, cloudxr_host: str | None
) -> None:
    inherited = {
        "TELEOP_PROXY_HOST": "dgx.local",
        "TELEOP_STREAM_SERVER_IP": "10.0.0.42",
        "TELEOP_STREAM_PORT": "49001",
        "PROXY_PORT": "49000",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    parent_environment = os.environ.copy()
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(cloudxr_host=cloudxr_host, cloudxr_port=48323)
    try:
        source.start()
        expected = (
            inherited
            if cloudxr_host is None
            else {
                "TELEOP_PROXY_HOST": cloudxr_host,
                "TELEOP_STREAM_SERVER_IP": cloudxr_host,
                "TELEOP_STREAM_PORT": "48323",
                "PROXY_PORT": "48323",
            }
        )
        assert {key: launcher.kwargs["env"][key] for key in inherited} == expected
    finally:
        source.stop()
    assert os.environ == parent_environment


@pytest.mark.parametrize("complete", [False, True])
def test_offline_source_requires_cached_assets_and_pins_child_path(
    monkeypatch, tmp_path, complete
):
    cache = tmp_path / "static-client"
    cache.mkdir()
    for name in ("index.html", "bundle.js", "bundle.emulator.js"):
        if complete or name != "bundle.emulator.js":
            (cache / name).write_text("cached asset")
    monkeypatch.setenv(
        "TELEOP_WEB_CLIENT_STATIC_DIR", str(cache / ".." / "static-client")
    )
    parent_environment = os.environ.copy()
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(host_client=True, offline_web_client=True)
    try:
        if complete:
            source.start()
            assert launcher.command is not None
            assert "--offline-web-client" in launcher.command
            assert launcher.kwargs["env"]["TELEOP_WEB_CLIENT_STATIC_DIR"] == str(cache)
        else:
            with pytest.raises(RuntimeError, match="bundle.emulator.js"):
                source.start()
            assert launcher.command is None
    finally:
        source.stop()
    assert os.environ == parent_environment


def test_offline_source_requires_local_client_hosting():
    with pytest.raises(ValueError, match="requires host_client"):
        _source(offline_web_client=True)


def test_offline_bridge_rejects_env_file_redirecting_checked_assets(monkeypatch):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", "/cached/client")
    monkeypatch.setattr(
        isaac_teleop_bridge, "cached_web_client_dir", lambda: Path("/cached/client")
    )
    config = ModuleType("isaacteleop.cloudxr.env_config")
    setattr(
        config,
        "EnvConfig",
        SimpleNamespace(
            _load_env_file=lambda _: {"TELEOP_WEB_CLIENT_STATIC_DIR": "/missing/client"}
        ),
    )
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr.env_config", config)
    args = build_bridge_parser().parse_args(
        [
            "serve",
            "--port",
            "8091",
            "--accept-eula",
            "--host-client",
            "--offline-web-client",
            "--cloudxr-env-config",
            "test.env",
        ]
    )
    with pytest.raises(
        ValueError, match="cannot override TELEOP_WEB_CLIENT_STATIC_DIR"
    ):
        _cloudxr_context(args)


@pytest.mark.parametrize("host_client", [False, True])
def test_offline_bridge_rejects_missing_assets_before_loading_sdk(
    monkeypatch, tmp_path, host_client
):
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(tmp_path))
    args = build_bridge_parser().parse_args(
        [
            "serve",
            "--port",
            "8091",
            "--accept-eula",
            "--offline-web-client",
            *(["--host-client"] if host_client else []),
        ]
    )
    with pytest.raises((ValueError, RuntimeError), match="host-client|cached WebXR"):
        _cloudxr_context(args)


@pytest.mark.parametrize("port", [0, -1, 65536, True, 48322.5, "48322"])
def test_source_rejects_invalid_cloudxr_advertising_port(port: Any) -> None:
    with pytest.raises(ValueError, match="CloudXR proxy port"):
        _source(cloudxr_host="laptop.local", cloudxr_port=port)


def test_source_taps_original_records_while_validating_locally(monkeypatch) -> None:
    from flashdreams_controller_input.controller_source_v2_window import (
        controller_frame_event,
    )

    status = {"type": "status", "stage": "runtime_ready", "detail": "OpenXR ready"}
    messages = [_hello(), status, _controller_frame()]
    tapped: list[str | bytes] = []
    malformed_received = threading.Event()
    disconnected = threading.Event()
    disconnects: list[None] = []

    def consume(raw: str | bytes) -> None:
        tapped.append(raw)
        if raw == b"  {not-json  ":
            malformed_received.set()

    def disconnect() -> None:
        disconnects.append(None)
        disconnected.set()

    launcher = _FakeLauncher(messages)
    _install_launcher(monkeypatch, launcher)
    source = _source(on_message=consume, on_disconnect=disconnect)
    try:
        source.start()
        process = launcher.process
        assert process is not None
        with process._connection_lock:
            assert process._connection is not None
            process._connection.sendall(b"  {not-json  \n")
        assert malformed_received.wait(timeout=_TIMEOUT_S)
        assert tapped == [json.dumps(message).encode() for message in messages] + [
            b"  {not-json  "
        ]
        assert source._store.rejected == 1
        observation = source.poll()
        assert observation is not None and observation.frame is not None
        assert observation.capture_ns == observation.frame["client_capture_ns"]
        assert frames.SOURCE_CLIENT_CAPTURE_NS_KEY not in observation.frame
        stored = source._store.snapshot()
        assert stored is not None
        assert stored[frames.SOURCE_CLIENT_CAPTURE_NS_KEY] == 10_000_000_000
        assert json.loads(tapped[2])["t_client"] == 10.0
        event = controller_frame_event(
            observation,
            session_origin_ns=0,
            frame_id=0,
            pose_convention=WRIST_PROXY_POSE_CONVENTION,
            source_name=source.name,
            client_capture_keys=(
                frames.SOURCE_CLIENT_CAPTURE_NS_KEY,
                "client_capture_ns",
            ),
        )
        assert event.client_capture_ns == observation.capture_ns
        assert observation.frame["left"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
        assert json.loads(tapped[2])["controllers"]["left"]["orientation"][-1] == 2.0
        process.terminate()
        assert disconnected.wait(timeout=_TIMEOUT_S)
    finally:
        source.stop()
    assert disconnects == [None]
    count = len(tapped)
    source._consume(json.dumps(_controller_frame(1)))
    assert len(tapped) == count


def test_concurrent_source_stops_join_the_in_flight_raw_callback(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    stopped: list[str] = []

    def consume(raw: str | bytes) -> None:
        if json.loads(raw).get("t_client") == 11.0:
            entered.set()
            assert release.wait(timeout=_TIMEOUT_S)

    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(on_message=consume)
    source.start()
    assert launcher.process is not None

    def stop(name: str) -> None:
        source.stop()
        stopped.append(name)

    stoppers = [threading.Thread(target=stop, args=(str(i),)) for i in range(2)]
    try:
        launcher.process.send(_controller_frame(1))
        assert entered.wait(timeout=_TIMEOUT_S)
        for stopper in stoppers:
            stopper.start()
        for stopper in stoppers:
            stopper.join(timeout=0.1)
            assert stopper.is_alive()
        assert not stopped
    finally:
        release.set()
        for stopper in stoppers:
            if stopper.ident is not None:
                stopper.join(timeout=_TIMEOUT_S)
        source.stop()
    assert sorted(stopped) == ["0", "1"]
    assert launcher.process.terminate_calls == 1


def test_receiver_handshake_yields_the_shared_controller_schema(monkeypatch) -> None:
    malformed = _controller_frame()
    malformed["controllers"]["left"].pop("orientation")
    launcher = _FakeLauncher([_hello(), malformed, _controller_frame(2)])
    _install_launcher(monkeypatch, launcher)
    source = _source()

    try:
        source.start()
        assert source.name == "isaac-teleop"
        observation = source.poll()
        assert observation is not None
        assert observation.source == "isaac-teleop"
        assert observation.capture_ns > 0
        assert observation.frame is not None
        validated = observation.frame
        assert validated["left"]["position"] == pytest.approx([-0.18, 1.0, -0.5])
        assert validated["right"]["position"] == pytest.approx([0.22, 1.0, -0.5])
        assert validated["left"]["orientation"] == pytest.approx([0.0, 0.0, 0.0, 1.0])
        assert validated["right"]["trigger"] == pytest.approx(0.4)
        assert validated["left"]["grip"] == pytest.approx(0.25)
    finally:
        source.stop()


def test_receiver_rejects_bridge_without_canonical_pose_semantics(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello(pose_convention=None), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=2.0)

    with pytest.raises(RuntimeError, match="pose convention"):
        source.start()
    source.stop()


@pytest.mark.parametrize(
    ("field", "advertised"),
    [
        pytest.param("mode", "native-apk", id="wrong-mode"),
        pytest.param(
            "input_pose_convention",
            OPENXR_LOCAL_GRIP_POSE_CONVENTION,
            id="local-instead-of-server-stage",
        ),
        pytest.param("output_pose_convention", "unexpected-output", id="wrong-output"),
        pytest.param("wrist_offset_m", 0.075, id="different-offset"),
        pytest.param("wrist_offset_m", True, id="boolean-offset"),
        pytest.param("wrist_offset_m", float("nan"), id="non-finite-offset"),
        pytest.param("wrist_offset_m", float("inf"), id="infinite-offset"),
    ],
)
def test_receiver_rejects_an_incompatible_bridge_contract(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    advertised: object,
) -> None:
    hello = _hello()
    hello[field] = advertised
    launcher = _FakeLauncher([hello, _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=2.0)

    with pytest.raises(RuntimeError, match=field):
        source.start()
    source.stop()


@pytest.mark.parametrize(
    "field",
    [
        "mode",
        "input_pose_convention",
        "output_pose_convention",
        "wrist_offset_m",
    ],
)
def test_receiver_rejects_a_bridge_with_a_missing_contract_field(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    hello = _hello()
    hello.pop(field)
    launcher = _FakeLauncher([hello, _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=2.0)

    with pytest.raises(RuntimeError, match=field):
        source.start()
    source.stop()


@pytest.mark.parametrize("wait_for_controllers", [False, True])
def test_receiver_rejects_an_incompatible_bridge_protocol(
    monkeypatch, wait_for_controllers
) -> None:
    launcher = _FakeLauncher([_hello(protocol=99), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=2.0, wait_for_controllers=wait_for_controllers)

    with pytest.raises(RuntimeError, match="protocol"):
        source.start()
    source.stop()


@pytest.mark.parametrize(
    "startup_timeout_s",
    [pytest.param(4.0, id="finite"), pytest.param(None, id="indefinite")],
)
def test_child_failure_is_reported_during_startup(
    monkeypatch, startup_timeout_s: float | None
) -> None:
    launcher = _FakeLauncher(None, returncode=23, stderr="bridge exploded")
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=startup_timeout_s)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="(disconnected|23|bridge exploded)"):
        source.start()

    assert time.monotonic() - started < 1.0
    source.stop()


@pytest.mark.parametrize("missing", [("left",), ("right",), ("left", "right")])
def test_source_can_start_on_partial_observations(monkeypatch, missing) -> None:
    frame = _controller_frame()
    for side in missing:
        frame["controllers"][side] = None
    launcher = _FakeLauncher([_hello(), frame])
    _install_launcher(monkeypatch, launcher)
    source = _source(wait_for_controllers=False)
    try:
        source.start()
        observation = source.poll()
        assert observation is not None
        assert observation.frame is not None
        for side in missing:
            assert observation.frame[side] is None
        source._store._store({**observation.frame, "client_capture_ns": 1})
        assert source.poll() is None
        source._store.consume(
            json.dumps({"type": "status", "stage": "failed", "detail": "lost OpenXR"})
        )
        with pytest.raises(RuntimeError, match="lost OpenXR"):
            source.poll()
    finally:
        source.stop()


@pytest.mark.parametrize("wait_for_controllers", [False, True])
def test_source_fails_fast_on_reported_startup_failure(
    monkeypatch, wait_for_controllers
) -> None:
    launcher = _FakeLauncher(
        [_hello(), {"type": "status", "stage": "failed", "detail": "failed OpenXR"}]
    )
    _install_launcher(monkeypatch, launcher)
    source = _source(wait_for_controllers=wait_for_controllers)
    with pytest.raises(RuntimeError, match="failed OpenXR"):
        source.start()


@pytest.mark.parametrize("exit_status", [None, 19])
def test_nonblocking_start_without_frames_retains_health_and_cleanup(
    monkeypatch, exit_status: int | None
) -> None:
    launcher = _FakeLauncher(None, returncode=exit_status)
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=0.01, wait_for_controllers=False)
    try:
        source.start(wait_for_frame=False)
        process = launcher.process
        assert process is not None
        assert source._process is process
        assert source._server is not None
        if exit_status is None:
            assert source.poll() is None
        else:
            with pytest.raises(RuntimeError, match=r"running \(status 19\)"):
                source.poll()
    finally:
        source.stop()
    assert source._process is None
    assert source._server is None
    assert process.terminate_calls == (1 if exit_status is None else 0)


def test_partial_startup_rejects_malformed_frames(monkeypatch) -> None:
    malformed = _controller_frame()
    malformed["controllers"]["left"].pop("orientation")
    launcher = _FakeLauncher([_hello(), malformed])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=0.15, wait_for_controllers=False)
    with pytest.raises(TimeoutError, match="fresh Isaac Teleop controller frame"):
        source.start()


def test_finite_startup_timeout_still_expires_without_controllers(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello()])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=0.15)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no tracked Isaac Teleop controllers"):
        source.start()

    assert time.monotonic() - started < 1.0
    source.stop()


@pytest.mark.parametrize("send_hello", [False, True])
@pytest.mark.parametrize("wait_for_controllers", [False, True])
def test_indefinite_startup_wait_is_interrupted_by_stop(
    monkeypatch, send_hello, wait_for_controllers
) -> None:
    launcher = _FakeLauncher([_hello()] if send_hello else [])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=None, wait_for_controllers=wait_for_controllers)
    failures: list[BaseException] = []

    def start() -> None:
        try:
            source.start()
        except BaseException as error:  # noqa: BLE001 - asserted below
            failures.append(error)

    thread = threading.Thread(target=start, name="indefinite-isaac-start")
    thread.start()
    deadline = time.monotonic() + 1.0
    while (
        launcher.process is None or (send_hello and source._store.protocol is None)
    ) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert launcher.process is not None
    assert source._store.protocol == (
        ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION if send_hello else None
    )
    assert thread.is_alive()
    source.stop()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "stopped while starting" in str(failures[0])


@pytest.mark.parametrize("blocked_boundary", ["receiver", "process"])
def test_stop_during_startup_resource_creation_leaves_no_late_resource(
    monkeypatch: pytest.MonkeyPatch, blocked_boundary: str
) -> None:
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    launcher = _FakeLauncher(None)
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=None)

    if blocked_boundary == "receiver":
        original_start = isaac_teleop_source.ControllerServer.start

        def delayed_start(server) -> None:
            entered.set()
            assert release.wait(timeout=_TIMEOUT_S)
            original_start(server)

        monkeypatch.setattr(
            isaac_teleop_source.ControllerServer, "start", delayed_start
        )
    else:

        def delayed_launch(command, *args, **kwargs):
            process = launcher(command, *args, **kwargs)
            entered.set()
            assert release.wait(timeout=_TIMEOUT_S)
            return process

        monkeypatch.setattr(isaac_teleop_source.subprocess, "Popen", delayed_launch)

    def start() -> None:
        try:
            source.start()
        except BaseException as error:  # noqa: BLE001 - asserted below
            failures.append(error)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=_TIMEOUT_S)
        server = source._server
        assert server is not None
        source.stop()
    finally:
        release.set()
        thread.join(timeout=_TIMEOUT_S)
        source.stop()

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert "stopped while starting" in str(failures[0])
    assert source._server is None
    assert source._process is None
    assert server._server is None
    if blocked_boundary == "process":
        assert launcher.process is not None
        assert launcher.process.terminate_calls == 1
    else:
        assert launcher.process is None


def test_indefinite_wait_still_rejects_incompatible_bridge(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello(protocol=99)])
    _install_launcher(monkeypatch, launcher)
    source = _source(startup_timeout_s=None)

    with pytest.raises(RuntimeError, match="protocol"):
        source.start()
    source.stop()


def test_poll_is_rate_limited_and_returns_only_the_latest_frame(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(rate_hz=20.0)

    try:
        source.start()
        first = source.poll()
        assert first is not None
        assert first.frame is not None
        assert first.frame["left"]["position"][0] == pytest.approx(-0.2)

        process = launcher.process
        assert process is not None
        process.send(_controller_frame(1))
        process.send(_controller_frame(2))
        assert source.poll() is None, "the source ignored its output-rate budget"

        time.sleep(0.08)
        latest = source.poll()
        assert latest is not None
        assert latest.frame is not None
        assert latest.frame["left"]["position"][0] == pytest.approx(-0.18)

        time.sleep(0.06)
        assert source.poll() is None, "one bridge frame was emitted more than once"
    finally:
        source.stop()


def test_active_client_identity_changes_begin_a_new_generation(monkeypatch) -> None:
    source = _source(rate_hz=20.0)
    now = 1_000_000_000
    monkeypatch.setattr(isaac_teleop_source.time, "perf_counter_ns", lambda: now)
    source._consume(json.dumps(_hello()))
    for index, (metadata, generation) in enumerate(
        [
            ({}, 0),
            ({}, 0),
            ({"client_connection_id": "first", "is_emulated": True}, 1),
            ({"client_connection_id": "first", "is_emulated": True}, 1),
            ({"client_connection_id": "first", "is_emulated": False}, 2),
            ({"client_connection_id": "second", "is_emulated": False}, 3),
            ({}, 4),
            ({}, 4),
        ]
    ):
        now += 100_000_000
        source._consume(json.dumps({**_controller_frame(index), **metadata}))
        observation = source.poll()
        assert observation is not None
        assert observation.generation == generation
        assert observation.capture_ns == now
        assert observation.frame is not None
        assert observation.frame["is_emulated"] is metadata.get("is_emulated", False)
        assert source.poll() is None


@pytest.mark.parametrize("emulated", [False, True])
def test_identity_poll_failure_does_not_change_controller_generation(
    monkeypatch, emulated
) -> None:
    source = _source(rate_hz=20.0)
    now = 1_000_000_000
    monkeypatch.setattr(isaac_teleop_source.time, "perf_counter_ns", lambda: now)
    source._consume(json.dumps(_hello()))

    async def run():
        nonlocal now
        snapshots = asyncio.Queue()
        observed = asyncio.Queue()

        async def get_snapshot():
            snapshot = await snapshots.get()
            observed.put_nowait(None)
            if isinstance(snapshot, Exception):
                raise snapshot
            return snapshot

        snapshot = {
            "headsets": [
                {
                    "clientId": "first",
                    "streamingSince": 123,
                    "connected": True,
                    "streaming": True,
                    "metricsByCadence": {
                        cloudxr_client_identity._METRICS_CADENCE: {
                            "metrics": {"iwer": int(emulated)}
                        }
                    },
                }
            ]
        }
        tracker = cloudxr_client_identity.CloudXRClientIdentity(enabled=True)
        task = asyncio.create_task(
            tracker._watch(SimpleNamespace(get_snapshot=get_snapshot))
        )
        try:
            for index, item in enumerate(
                (snapshot, RuntimeError("temporary hub failure"), snapshot)
            ):
                snapshots.put_nowait(item)
                await asyncio.wait_for(observed.get(), 1.0)
                assert not task.done()
                connection_id, is_emulated = tracker.sample()
                now += 100_000_000
                source._consume(
                    json.dumps(
                        {
                            **_controller_frame(index),
                            "client_connection_id": connection_id,
                            "is_emulated": is_emulated,
                        }
                    )
                )
                observation = source.poll()
                assert observation is not None
                assert observation.generation == 0
                assert observation.frame is not None
                assert observation.frame["client_connection_id"] == "first:123"
                assert observation.frame["is_emulated"] is emulated
        finally:
            tracker.close()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())


def test_health_checks_leave_the_latest_frame_and_rate_budget_untouched(
    monkeypatch,
) -> None:
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source(rate_hz=20.0)
    try:
        source.start()
        next_ns = source._next_ns
        for _ in range(3):
            source.check_alive()
        assert source._next_ns == next_ns
        assert source._last_capture_ns == 0
        assert source.poll() is not None
        assert source.poll() is None
        assert launcher.process is not None
        launcher.process.returncode = 19
        with pytest.raises(RuntimeError, match=r"running \(status 19\)"):
            source.check_alive()
    finally:
        source.stop()


@pytest.mark.parametrize(
    "message, match",
    [
        (_hello(protocol=99), "protocol"),
        ({"type": "status", "stage": "failed", "detail": "lost OpenXR"}, "lost OpenXR"),
    ],
)
def test_health_checks_report_bridge_failures_without_polling(message, match) -> None:
    source = _source()
    source._store.consume(json.dumps(_hello()))
    source._store.consume(json.dumps(message))
    with pytest.raises(RuntimeError, match=match):
        source.check_alive()


def test_stop_is_idempotent_and_terminates_the_sidecar_once(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source()
    source.start()
    process = launcher.process
    assert process is not None

    source.stop()
    source.stop()

    assert source._process is None
    assert source._server is None
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_runtime_child_exit_is_reported_after_source_became_ready(monkeypatch) -> None:
    launcher = _FakeLauncher([_hello(), _controller_frame()])
    _install_launcher(monkeypatch, launcher)
    source = _source()

    try:
        source.start()
        process = launcher.process
        assert process is not None
        process.returncode = 19

        with pytest.raises(RuntimeError, match=r"running \(status 19\)"):
            source.poll()
    finally:
        source.stop()


@pytest.mark.parametrize("failure", ["capture-stream", "runtime-health"])
@pytest.mark.parametrize("identity", [(None, False), ("iwer-session:10", True)])
def test_desktop_capture_keeps_hello_health_and_cleanup_without_calibration_rpc(
    monkeypatch: pytest.MonkeyPatch, failure: str, identity: tuple[str | None, bool]
) -> None:
    messages: list[dict[str, Any]] = []
    events: list[str] = []
    monkeypatch.setattr(
        isaac_teleop_bridge, "require_supported_version", lambda: "test-sdk"
    )
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "_build_controller_pipeline",
        lambda _offset: (object(), object()),
    )
    monkeypatch.setattr(isaac_teleop_bridge.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        isaac_teleop_bridge.CloudXRClientIdentity, "sample", lambda _: identity
    )

    def health_check() -> None:
        events.append("health")
        if failure == "runtime-health":
            raise RuntimeError("runtime stopped")

    @contextmanager
    def runtime(_args):
        try:
            yield SimpleNamespace(health_check=health_check)
        finally:
            events.append("runtime-exit")

    @contextmanager
    def session(_config):
        try:
            yield SimpleNamespace(step=lambda: object())
        finally:
            events.append("session-exit")

    sdk = ModuleType("isaacteleop.teleop_session_manager")
    monkeypatch.setattr(sdk, "TeleopSession", session, raising=False)
    monkeypatch.setattr(
        sdk, "TeleopSessionConfig", lambda **kwargs: kwargs, raising=False
    )
    monkeypatch.setitem(sys.modules, "isaacteleop", ModuleType("isaacteleop"))
    monkeypatch.setitem(sys.modules, "isaacteleop.teleop_session_manager", sdk)
    monkeypatch.setattr(isaac_teleop_bridge, "_cloudxr_context", runtime)
    connection = SimpleNamespace(
        setsockopt=lambda *_args: None,
        makefile=lambda *_args, **_kwargs: nullcontext(object()),
    )
    monkeypatch.setattr(
        isaac_teleop_bridge.socket,
        "create_connection",
        lambda *_args, **_kwargs: nullcontext(connection),
    )
    monkeypatch.setattr(
        isaac_teleop_bridge,
        "controller_result_to_message",
        lambda *_args, **_kwargs: {"type": "xr_frame"},
    )

    def send(_stream, payload) -> None:
        messages.append(payload)
        if payload["type"] == "xr_frame":
            raise BrokenPipeError("capture receiver closed")

    monkeypatch.setattr(isaac_teleop_bridge, "send_message", send)
    args = isaac_teleop_bridge.build_parser().parse_args(["serve", "--port", "8091"])
    error = BrokenPipeError if failure == "capture-stream" else RuntimeError
    with pytest.raises(error):
        isaac_teleop_bridge._serve(args)

    assert messages[0] == {
        "type": "hello",
        "protocol": isaac_teleop_bridge.ISAAC_TELEOP_BRIDGE_PROTOCOL_VERSION,
        "version": "test-sdk",
        "mode": "cloudxr-webxr",
        "input_pose_convention": OPENXR_STAGE_GRIP_POSE_CONVENTION,
        "output_pose_convention": WRIST_PROXY_POSE_CONVENTION,
        "pose_convention": WRIST_PROXY_POSE_CONVENTION,
        "wrist_offset_m": 0.05,
    }
    assert [
        message["stage"] for message in messages if message["type"] == "status"
    ] == ["connected", "runtime", "session"]
    assert events == ["health", "session-exit", "runtime-exit"]
    if failure == "capture-stream":
        assert messages[-1]["client_connection_id"] == identity[0]
        assert messages[-1]["is_emulated"] is identity[1]


def test_cloudxr_acceptance_blocks_source_before_process_or_receiver_start(monkeypatch):
    launcher = _FakeLauncher(messages=None)
    _install_launcher(monkeypatch, launcher)
    source = _source(accept_eula=False)

    with pytest.raises(ValueError, match="explicit acceptance") as error:
        source.start()

    assert "isaacteleop[cloudxr,retargeters-lite]==1.4.142" in str(error.value)
    assert isaac_teleop_bridge.CLOUDXR_EULA_HELP in str(error.value)
    assert source._server is None
    assert source._process is None
    assert launcher.process is None


def test_bridge_rejects_missing_acceptance_before_sdk_import(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "isaacteleop", None)
    with pytest.raises(SystemExit) as error:
        isaac_teleop_bridge.main(["serve", "--port", "8091"])
    assert error.value.code == 2
    assert isaac_teleop_bridge.CLOUDXR_EULA_HELP in capsys.readouterr().err

    args = build_bridge_parser().parse_args(["serve", "--port", "8091"])
    with pytest.raises(ValueError, match="requires --accept-eula"):
        _cloudxr_context(args)


def test_doctor_requires_no_eula_and_does_not_serve(monkeypatch):
    monkeypatch.setattr(isaac_teleop_bridge, "_doctor", lambda: 0)
    monkeypatch.setitem(sys.modules, "isaacteleop", None)
    assert isaac_teleop_bridge.main(["doctor"]) == 0


def test_missing_sidecar_reports_setup_and_selected_doctor(tmp_path):
    python = tmp_path / "missing sidecar" / "bin" / "python"
    with pytest.raises(RuntimeError) as error:
        isaac_teleop_source.resolve_python(str(python))
    assert "flashdreams-controller-input setup isaac-teleop" in str(error.value)
    assert str(python) in str(error.value)
    assert "isaac_teleop_bridge.py doctor" in str(error.value)


def test_failed_child_reports_setup_and_doctor(monkeypatch):
    launcher = _FakeLauncher(messages=None, returncode=23)
    _install_launcher(monkeypatch, launcher)
    with pytest.raises(RuntimeError) as error:
        _source().start()
    assert "status 23" in str(error.value)
    assert "flashdreams-controller-input setup isaac-teleop" in str(error.value)
    assert "isaac_teleop_bridge.py doctor" in str(error.value)


@pytest.mark.parametrize("variable", ["PYTHONHOME", "PYTHONPATH"])
def test_source_ignores_parent_python_environment_in_real_child(
    variable, tmp_path, monkeypatch
):
    shadow = tmp_path / "main-environment"
    shadow.mkdir()
    (shadow / "numpy.py").write_text("raise RuntimeError('main NumPy leaked')\n")
    monkeypatch.setenv(variable, str(shadow))
    parent_environment = os.environ.copy()
    source = IsaacTeleopSource(python_executable=sys.executable, launch_runtime=False)
    monkeypatch.setattr(
        source,
        "_bridge_command",
        lambda *, python, port: [python, str(isaac_teleop_bridge.__file__), "--help"],
    )
    try:
        source.start(wait_for_frame=False)
        assert source._process is not None
        assert source._process.wait(timeout=_TIMEOUT_S) == 0
    finally:
        source.stop()
    assert os.environ == parent_environment


def test_sidecar_disables_user_site_for_a_custom_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "user-base"))
    monkeypatch.setenv("PYTHONNOUSERSITE", "")
    environment = isaac_teleop_source.isolated_python_environment()
    assert "PYTHONUSERBASE" not in environment
    result = subprocess.run(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-c",
            "import site; assert site.ENABLE_USER_SITE is False",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr
