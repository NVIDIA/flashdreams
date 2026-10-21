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

"""CPU checks for command composition and private SSH/TLS bootstrap."""

import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.ci_cpu

_SPEC = importlib.util.spec_from_file_location(
    "handdreams_launcher",
    Path(__file__).resolve().parents[2] / "apps/handdreams/launch.py",
)
assert _SPEC is not None
launch = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(launch)
transport_credentials = launch._credentials

_WEB_ASSETS = {
    name: f"Isaac Teleop 1.4: {name}".encode()
    for name in ("index.html", "bundle.js", "bundle.emulator.js")
}


def test_launcher_bootstraps_from_public_checkout_without_installed_packages(tmp_path):
    repository = Path(__file__).resolve().parents[4]
    relative = Path(
        "integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py"
    )
    copied = tmp_path / relative
    copied.parent.mkdir(parents=True)
    shutil.copy2(repository / relative, copied)
    controllers = Path("apps/handdreams/controller_input")
    assert (repository / controllers / "uv.lock").resolve() == repository / "uv.lock"
    shutil.copytree(
        repository / controllers / "flashdreams_controller_input",
        tmp_path / controllers / "flashdreams_controller_input",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            """
from pathlib import Path
import runpy
import sys

root = Path(sys.argv[1])
launcher = runpy.run_path(sys.argv[2])
assert launcher['REPOSITORY'] == root
assert not (root / 'internal').exists()
assert Path(launcher['_credentials'].__file__).is_relative_to(root / 'apps')
assert Path(launcher['_setup_module']().__file__).is_relative_to(root / 'apps')
parser = launcher['_parser']()
server = parser.parse_args(['server', '--address', '127.0.0.1', '--accept-eula'])
command = launcher['_command'](server, [], root / 'credentials')
assert command[:6] == ['uv', 'run', '--package', 'flashdreams-cosmos3-nano-sim-bimanual',
                      'flashdreams-run-v2', 'handdreams']
remote = parser.parse_args(['remote-vr', '--server', 'demo@127.0.0.1', '--accept-eula'])
command = launcher['_command'](remote, [], root / 'credentials')
assert command[command.index('--with-editable') + 1] == str(root / 'apps/handdreams/controller_input') + '[webxr]'
""",
            str(tmp_path),
            str(copied),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def scene_arguments(tmp_path):
    scene = tmp_path / "scenarios" / "synthetic"
    scene.mkdir(parents=True)
    for filename in ("scene.json", "first_frame.jpg"):
        (scene / filename).touch()
    return ["--sample-dir", str(scene.parent)]


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize("source", ["--sample-dir", "--scene-path"])
def test_scene_source_is_absolute_and_forwarded_once(
    mode, passthrough, source, scene_arguments, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    relative = "scenarios" if source == "--sample-dir" else "scenarios/synthetic"
    options = [source, relative]
    args = launch._parser().parse_args(
        ["server", "--accept-eula", "--mode", mode, *([] if passthrough else options)]
    )
    extra = launch._validate_scene_source(
        args, [*(options if passthrough else []), "--sample", "synthetic"]
    )
    command = launch._command(args, extra, tmp_path / "credentials")
    application = command[command.index("--") + 1 :]
    assert application.count(source) == 1
    assert application[application.index(source) + 1] == str(tmp_path / relative)
    assert application[-2:] == ["--sample", "synthetic"]


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
def test_launcher_leaves_default_scene_preparation_to_application(mode, tmp_path):
    args = launch._parser().parse_args(["server", "--accept-eula", "--mode", mode])
    extra = launch._validate_scene_source(args, ["--auto-start"])
    assert extra == ["--auto-start"]
    command = launch._command(args, extra, tmp_path / "credentials")
    assert "--sample-dir" not in command
    assert "--scene-path" not in command


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
@pytest.mark.parametrize("address", ["127.0.0.1", "10.111.69.99"])
@pytest.mark.parametrize("prewarm_passthrough", [False, True])
def test_dataset_replay_skips_all_controller_setup(
    mode, address, prewarm_passthrough, monkeypatch, capsys
):
    def unexpected(*args, **kwargs):
        pytest.fail("Recorded replay must not prepare live controller resources")

    for name in ("server_credentials", "client_credentials", "read_token"):
        monkeypatch.setattr(transport_credentials, name, unexpected)
    for name in ("_setup_module", "_client_web_assets", "_validate_scene_source"):
        monkeypatch.setattr(launch, name, unexpected)
    checked = []

    def which(command):
        checked.append(command)
        return "/usr/bin/uv" if command == "uv" else None

    monkeypatch.setattr(launch.shutil, "which", which)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.delenv("FLASHDREAMS_CONTROLLER_TOKEN", raising=False)
    monkeypatch.delenv("TELEOP_WEB_CLIENT_STATIC_DIR", raising=False)
    monkeypatch.delenv("TELEOP_PROXY_HOST", raising=False)

    def execute(executable, command, environment):
        assert executable == "uv"
        assert command[:6] == [
            "uv",
            "run",
            "--package",
            launch.MODEL_PACKAGE,
            "flashdreams-run-v2",
            "handdreams-replay",
        ]
        assert command[command.index("--mode") + 1] == mode
        assert command[command.index("--") + 1 :] == [
            "--checkpoint",
            "/models/checkpoint",
            "--seed",
            "42",
            "--decoder",
            "wan",
        ]
        assert "--fps" not in command
        assert not any(
            "isaac" in option or "controller" in option for option in command
        )
        assert "FLASHDREAMS_CONTROLLER_TOKEN" not in environment
        assert "TELEOP_WEB_CLIENT_STATIC_DIR" not in environment
        assert "TELEOP_PROXY_HOST" not in environment
        assert environment["CUDA_VISIBLE_DEVICES"] == "1"
        if mode == "webrtc":
            expected_host = address if address == "127.0.0.1" else "0.0.0.0"
            assert command[command.index("--host") + 1] == expected_host
            assert command[command.index("--port") + 1] == "8082"
        else:
            assert "--host" not in command
            assert "--port" not in command
        raise SystemExit(0)

    monkeypatch.setattr(launch.os, "execvpe", execute)
    with pytest.raises(SystemExit) as result:
        launch.main(
            [
                "server",
                "--replay",
                *([] if prewarm_passthrough else ["--prewarm-all-scenes"]),
                "--address",
                address,
                "--port",
                "8082",
                "--mode",
                mode,
                "--gpu",
                "1",
                "--checkpoint",
                "/models/checkpoint",
                "--",
                *(["--prewarm-all-scenes"] if prewarm_passthrough else []),
                "--seed",
                "42",
                "--decoder",
                "wan",
            ]
        )
    assert result.value.code == 0
    assert checked == ["uv"]
    output = capsys.readouterr().out
    assert "Operator viewer:" in output
    assert "Isaac Teleop" not in output
    assert "CloudXR" not in output


@pytest.mark.parametrize("passthrough", [False, True])
@pytest.mark.parametrize(
    "options",
    [["--sample-dir", "/scenes"], ["--scene-path=/scene"]],
)
def test_dataset_replay_rejects_live_scene_options(options, passthrough, capsys):
    with pytest.raises(SystemExit) as result:
        launch.main(
            [
                "server",
                "--replay",
                *(["--"] if passthrough else []),
                *options,
            ]
        )
    assert result.value.code == 2
    assert "do not provide matching recorded actions" in capsys.readouterr().err


@pytest.mark.parametrize("fault", ["absent", "empty", "incomplete"])
def test_invalid_scene_source_fails_before_credential_bootstrap(
    fault, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(
        transport_credentials,
        "server_credentials",
        lambda _: pytest.fail("No bootstrap before valid scenes"),
    )
    path = tmp_path / "scenarios"
    if fault != "absent":
        path.mkdir()
    if fault == "incomplete":
        (path / "scene.json").touch()
    options = ["--sample-dir", str(path)]
    with pytest.raises(SystemExit) as result:
        launch.main(["server", "--accept-eula", *options])
    assert result.value.code == 2
    assert "scene" in capsys.readouterr().err.lower()


def test_launcher_rejects_scene_source_conflicts_across_separator(tmp_path):
    args = launch._parser().parse_args(["server", "--sample-dir", str(tmp_path)])
    with pytest.raises(SystemExit) as result:
        launch._validate_scene_source(args, ["--scene-path", str(tmp_path)])
    assert result.value.code == 2


@pytest.fixture
def web_client_cache(tmp_path, monkeypatch):
    directory = tmp_path / "client"
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(directory))
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("Unexpected external process"),
    )
    return directory


@pytest.mark.parametrize(
    "location", ["default", "empty", "blank", "custom", "relative"]
)
def test_web_assets_reuses_complete_cache_without_scp(
    location, web_client_cache, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    original_expanduser = launch.os.path.expanduser
    monkeypatch.setattr(
        launch.os.path,
        "expanduser",
        lambda path: str(tmp_path) if path == "~" else original_expanduser(path),
    )
    if location in ("default", "empty", "blank"):
        directory = tmp_path / ".cloudxr/static-client"
        if location == "default":
            monkeypatch.delenv("TELEOP_WEB_CLIENT_STATIC_DIR")
        else:
            monkeypatch.setenv(
                "TELEOP_WEB_CLIENT_STATIC_DIR", "" if location == "empty" else "   "
            )
    else:
        directory = web_client_cache
        if location == "relative":
            monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", "client")
    directory.mkdir(parents=True)
    for name, value in _WEB_ASSETS.items():
        (directory / name).write_bytes(value)
    monkeypatch.setattr(
        launch.shutil, "which", lambda _: pytest.fail("A complete cache needs no scp")
    )

    assert launch._client_web_assets("nvidia@dgx.example") == directory
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == _WEB_ASSETS


def test_web_assets_reuses_complete_cache_with_readable_symlink(
    web_client_cache, tmp_path, monkeypatch
):
    web_client_cache.mkdir()
    for name, value in _WEB_ASSETS.items():
        (web_client_cache / name).write_bytes(value)
    index = web_client_cache / "index.html"
    index.rename(tmp_path / "existing-index.html")
    index.symlink_to(tmp_path / "existing-index.html")
    monkeypatch.setattr(
        launch.shutil, "which", lambda _: pytest.fail("A complete cache needs no scp")
    )

    assert launch._client_web_assets("nvidia@dgx.example") == web_client_cache
    assert index.is_symlink()


@pytest.mark.parametrize(
    "state", ["absent", "empty-directory", "empty-files", "partial"]
)
def test_web_assets_stages_complete_bundle_over_verified_ssh(
    state, web_client_cache, monkeypatch
):
    if state != "absent":
        web_client_cache.mkdir()
    if state == "empty-files":
        for name in _WEB_ASSETS:
            (web_client_cache / name).touch()
    existing = web_client_cache / "index.html"
    if state == "partial":
        existing.write_bytes(_WEB_ASSETS["index.html"])
        original_inode = existing.stat().st_ino
    calls = []

    def copy(command, **kwargs):
        calls.append((command, kwargs))
        staging = Path(command[-1])
        assert staging.parent == web_client_cache
        for name, value in _WEB_ASSETS.items():
            (staging / name).write_bytes(value)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", copy)
    assert launch._client_web_assets("nvidia@dgx.example") == web_client_cache
    assert {
        path.name: path.read_bytes() for path in web_client_cache.iterdir()
    } == _WEB_ASSETS
    assert len(calls) == 1
    command, options = calls[0]
    assert command[0] == "scp"
    assert command[command.index("--") + 1 : -1] == [
        f"nvidia@dgx.example:.cloudxr/static-client/{name}" for name in _WEB_ASSETS
    ]
    assert {
        "StrictHostKeyChecking=yes",
        "UpdateHostKeys=no",
        "BatchMode=yes",
        "ConnectTimeout=10",
    } <= set(command)
    assert options == dict(check=True, capture_output=True, text=True, timeout=300)
    assert not Path(command[-1]).exists()
    if state == "partial":
        assert existing.stat().st_ino == original_inode


def test_web_assets_reports_missing_scp_without_creating_cache(
    web_client_cache, monkeypatch
):
    monkeypatch.setattr(launch.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="scp"):
        launch._client_web_assets("nvidia@dgx.example")
    assert not web_client_cache.exists()


@pytest.mark.parametrize("failure", ["failed", "timeout", "empty", "missing"])
def test_web_assets_failed_transfer_preserves_cache_and_cleans_staging(
    failure, web_client_cache, monkeypatch, capsys
):
    web_client_cache.mkdir()
    before = {"index.html": _WEB_ASSETS["index.html"], "bundle.js": b""}
    for name, value in before.items():
        (web_client_cache / name).write_bytes(value)
    staging_paths = []

    def copy(command, **kwargs):
        staging = Path(command[-1])
        staging_paths.append(staging)
        for name, value in _WEB_ASSETS.items():
            (staging / name).write_bytes(value)
        if failure == "failed":
            raise subprocess.CalledProcessError(255, command, output="untrusted output")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                command, kwargs["timeout"], output="untrusted output"
            )
        if failure == "empty":
            (staging / "bundle.emulator.js").write_bytes(b"")
        else:
            (staging / "bundle.emulator.js").unlink()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", copy)
    with pytest.raises(ValueError, match="preserved") as error:
        launch._client_web_assets("nvidia@dgx.example")
    assert {
        path.name: path.read_bytes() for path in web_client_cache.iterdir()
    } == before
    assert staging_paths and all(not path.exists() for path in staging_paths)
    output = capsys.readouterr()
    assert "untrusted output" not in str(error.value) + output.out + output.err


def test_web_assets_rejects_mismatched_partial_bundle_without_changes(
    web_client_cache, monkeypatch
):
    web_client_cache.mkdir()
    before = {"index.html": b"different client build", "bundle.js": b""}
    for name, value in before.items():
        (web_client_cache / name).write_bytes(value)

    def copy(command, **kwargs):
        for name, value in _WEB_ASSETS.items():
            (Path(command[-1]) / name).write_bytes(value)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", copy)
    with pytest.raises(ValueError, match="does not match"):
        launch._client_web_assets("nvidia@dgx.example")
    assert {
        path.name: path.read_bytes() for path in web_client_cache.iterdir()
    } == before


@pytest.mark.parametrize("kind", ["symlink", "broken-symlink", "directory", "fifo"])
def test_web_assets_rejects_nonregular_partial_cache_without_changes(
    kind, web_client_cache, tmp_path, monkeypatch
):
    web_client_cache.mkdir()
    destination = web_client_cache / "index.html"
    outside = tmp_path / "existing-client.html"
    outside.write_bytes(_WEB_ASSETS["index.html"])
    if kind == "symlink":
        destination.symlink_to(outside)
    elif kind == "broken-symlink":
        destination.symlink_to(tmp_path / "missing.html")
    elif kind == "directory":
        destination.mkdir()
    else:
        launch.os.mkfifo(destination)
    before = destination.lstat()

    def copy(command, **kwargs):
        for name, value in _WEB_ASSETS.items():
            (Path(command[-1]) / name).write_bytes(value)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", copy)
    with pytest.raises(ValueError, match="regular file"):
        launch._client_web_assets("nvidia@dgx.example")
    assert list(web_client_cache.iterdir()) == [destination]
    after = destination.lstat()
    assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    assert outside.read_bytes() == _WEB_ASSETS["index.html"]


@pytest.fixture(scope="module")
def server_credentials(tmp_path_factory):
    if launch.shutil.which("openssl") is None:
        pytest.skip("OpenSSL is required for local TLS checks")
    root = tmp_path_factory.mktemp("server") / "config"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with patch.object(transport_credentials, "_state_root", return_value=root):
        directory = transport_credentials.server_credentials("10.111.69.99")
    assert root.stat().st_mode & 0o777 == 0o755
    return directory


def test_server_credentials_are_private_valid_and_reused(server_credentials):
    before = {path.name: path.read_bytes() for path in server_credentials.iterdir()}
    assert server_credentials.stat().st_mode & 0o777 == 0o700
    assert all(
        path.stat().st_mode & 0o777 == 0o600 for path in server_credentials.iterdir()
    )
    # A certificate is public; an existing readable certificate is supported.
    (server_credentials / "server.crt").chmod(0o644)
    with patch.object(
        transport_credentials, "_state_root", return_value=server_credentials.parent
    ):
        assert (
            transport_credentials.server_credentials("10.111.69.99")
            == server_credentials
        )
    assert before == {
        path.name: path.read_bytes() for path in server_credentials.iterdir()
    }
    with pytest.raises(ValueError, match="does not cover"):
        transport_credentials.check_certificate(
            server_credentials / "server.crt", "10.111.69.100"
        )


def test_loopback_credentials_preserve_existing_lan_trust(server_credentials):
    before = {path.name: path.read_bytes() for path in server_credentials.iterdir()}
    with patch.object(
        transport_credentials, "_state_root", return_value=server_credentials.parent
    ):
        local = transport_credentials.server_credentials("127.0.0.1")
        assert local == server_credentials.parent / "quest-relay-127.0.0.1"
        local_before = {path.name: path.read_bytes() for path in local.iterdir()}
        assert transport_credentials.server_credentials("127.0.0.1") == local
        assert (
            transport_credentials.server_credentials("10.111.69.99")
            == server_credentials
        )
    assert {path.name: path.read_bytes() for path in local.iterdir()} == local_before
    assert {
        path.name: path.read_bytes() for path in server_credentials.iterdir()
    } == before
    assert local.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in local.iterdir())
    assert transport_credentials.read_token(local) != transport_credentials.read_token(
        server_credentials
    )
    transport_credentials.check_certificate(local / "server.crt", "127.0.0.1")
    with pytest.raises(ValueError, match="does not cover"):
        transport_credentials.check_certificate(local / "server.crt", "10.111.69.99")


def test_partial_and_symlink_credentials_are_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(transport_credentials, "_state_root", lambda: tmp_path)
    directory = tmp_path / "quest-relay"
    directory.mkdir(mode=0o700)
    (directory / "token").write_text("existing private state")
    monkeypatch.setattr(
        transport_credentials,
        "_run",
        lambda _: pytest.fail("Must not generate replacement credentials"),
    )
    with pytest.raises(ValueError, match="Incomplete credentials"):
        transport_credentials.server_credentials("10.111.69.99")
    assert (directory / "token").read_text() == "existing private state"
    with pytest.raises(ValueError, match="Unsafe ownership"):
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        transport_credentials._existing(alias, ("token",))


def test_exclusive_publication_preserves_existing_trust(tmp_path):
    directory = tmp_path / "client"
    directory.mkdir(mode=0o700)
    transport_credentials._write_private(directory / "token", "existing")
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    transport_credentials._write_private(staging / "token", "replacement")
    with pytest.raises(FileExistsError):
        transport_credentials._publish(staging, directory, ("token",))
    assert (directory / "token").read_text() == "existing"


def test_client_bootstraps_only_over_verified_ssh_and_reuses_trust(
    server_credentials, tmp_path, monkeypatch
):
    bundle = {
        name: (server_credentials / name).read_text()
        for name in ("server.crt", "token")
    }
    monkeypatch.setattr(transport_credentials, "_state_root", lambda: tmp_path)
    original_run = transport_credentials._run
    calls = []

    def run(command):
        if command[0] == "ssh":
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(bundle))
        return original_run(command)

    monkeypatch.setattr(transport_credentials, "_run", run)
    directory = transport_credentials.client_credentials(
        "nvidia@10.111.69.99", "10.111.69.99"
    )
    assert set(path.name for path in directory.iterdir()) == {"server.crt", "token"}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in directory.iterdir())
    assert len(calls) == 1
    assert {"StrictHostKeyChecking=yes", "UpdateHostKeys=no", "BatchMode=yes"} <= set(
        calls[0]
    )
    assert "server.key" not in calls[0][-1]
    assert bundle["token"] not in " ".join(calls[0])
    assert (
        transport_credentials.client_credentials("nvidia@10.111.69.99", "10.111.69.99")
        == directory
    )
    assert len(calls) == 1


def test_unverified_ssh_failure_does_not_create_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(transport_credentials, "_state_root", lambda: tmp_path)

    def failed(command):
        raise subprocess.CalledProcessError(255, command, output="untrusted output")

    monkeypatch.setattr(transport_credentials, "_run", failed)
    with pytest.raises(ValueError, match="verified SSH") as failure:
        transport_credentials.client_credentials("nvidia@10.111.69.99", "10.111.69.99")
    assert "untrusted output" not in str(failure.value)
    assert not (tmp_path / "quest-relay-clients/nvidia@10.111.69.99").exists()


@pytest.mark.parametrize(
    "value",
    [
        "-oProxyCommand=bad@host",
        "nvidia@host;command",
        "nvidia@../host",
        "nvidia@[::1]",
        "host",
    ],
)
def test_ssh_target_rejects_shell_or_path_syntax(value):
    with pytest.raises(launch.argparse.ArgumentTypeError):
        launch._ssh_target(value)


@pytest.mark.parametrize("value", ["::1", "0.0.0.0", "224.0.0.1", "dgx.example"])
def test_address_requires_unicast_ipv4(value):
    with pytest.raises(launch.argparse.ArgumentTypeError):
        launch._address(value)


@pytest.mark.parametrize("value", ["127.0.0.1", "127.0.0.2", "10.111.69.99"])
def test_server_accepts_loopback_and_lan_addresses(value):
    assert launch._address(value) == value


@pytest.mark.parametrize("host", ["dgx.example", "dgx-local"])
@pytest.mark.parametrize("answer_count", [1, 2])
def test_controller_hostname_resolves_one_unique_ipv4(host, answer_count):
    answer = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.111.69.99", 0))
    with patch.object(
        socket, "getaddrinfo", return_value=[answer] * answer_count
    ) as resolve:
        assert launch._controller_address(host) == "10.111.69.99"
    resolve.assert_called_once_with(
        host, None, family=socket.AF_INET, type=socket.SOCK_STREAM
    )


@pytest.mark.parametrize(
    "host", ["10.111.69.99", "::1", "0.0.0.0", "127.0.0.1", "224.0.0.1"]
)
def test_controller_literal_addresses_do_not_use_dns(host):
    with patch.object(
        socket, "getaddrinfo", side_effect=AssertionError("Unexpected DNS")
    ):
        if host == "10.111.69.99":
            assert launch._controller_address(host) == host
        else:
            with pytest.raises(ValueError, match="IPv4"):
                launch._controller_address(host)


@pytest.mark.parametrize(
    "addresses",
    [
        [],
        ["10.111.69.99", "10.111.69.100"],
        ["0.0.0.0"],
        ["127.0.0.1"],
        ["224.0.0.1"],
        ["::1"],
    ],
)
def test_controller_rejects_missing_ambiguous_or_unreachable_addresses(addresses):
    answers = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
        for address in addresses
    ]
    with patch.object(socket, "getaddrinfo", return_value=answers):
        with pytest.raises(ValueError, match="IPv4"):
            launch._controller_address("dgx.example")


def test_controller_resolution_failure_explains_ssh_alias_limit():
    with patch.object(socket, "getaddrinfo", side_effect=socket.gaierror("not found")):
        with pytest.raises(ValueError, match="IPv4") as failure:
            launch._controller_address("ssh-alias")
    assert "SSH" in str(failure.value) and "alias" in str(failure.value)


@pytest.mark.parametrize("address", ["10.111.69.99", "10.111.69.100"])
def test_remote_vr_resolves_hostname_for_certificate_and_relay(
    address, server_credentials, tmp_path, monkeypatch, capsys
):
    class Executed(Exception):
        pass

    bundle = {
        name: (server_credentials / name).read_text()
        for name in ("server.crt", "token")
    }
    original_run = transport_credentials._run
    calls, executions = [], []
    web_client_cache = tmp_path / "client"
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(web_client_cache))
    web_client_cache.mkdir()
    for name, value in _WEB_ASSETS.items():
        (web_client_cache / name).write_bytes(value)

    def run(command):
        calls.append(command)
        if command[0] == "ssh":
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(bundle))
        return original_run(command)

    def execute(executable, command, environment):
        executions.append((command, environment))
        raise Executed

    monkeypatch.setattr(transport_credentials, "_state_root", lambda: tmp_path)
    monkeypatch.setattr(transport_credentials, "_run", run)
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(launch.os, "execvpe", execute)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    answer = (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))
    arguments = ["remote-vr", "--server", "nvidia@dgx.example", "--accept-eula"]
    directory = tmp_path / "quest-relay-clients/nvidia@dgx.example"
    with patch.object(socket, "getaddrinfo", return_value=[answer]) as resolve:
        if address == "10.111.69.99":
            for _ in range(2):
                with pytest.raises(Executed):
                    launch.main(arguments)
            assert resolve.call_count == 2
            assert {path.name for path in directory.iterdir()} == set(bundle)
            assert all(
                (directory / name).read_text() == value
                for name, value in bundle.items()
            )
        else:
            with pytest.raises(SystemExit) as result:
                launch.main(arguments)
            assert result.value.code == 2
            assert not directory.exists() and not executions
    ssh_calls = [command for command in calls if command[0] == "ssh"]
    assert len(ssh_calls) == 1
    assert ssh_calls[0][ssh_calls[0].index("--") + 1] == "nvidia@dgx.example"
    assert {"StrictHostKeyChecking=yes", "UpdateHostKeys=no", "BatchMode=yes"} <= set(
        ssh_calls[0]
    )
    assert "server.key" not in ssh_calls[0][-1]
    checks = [command for command in calls if command[0] == "openssl"]
    assert checks and all(command[1] == "verify" for command in checks)
    assert all(
        command[command.index("-verify_ip") + 1] == address for command in checks
    )
    assert all("-verify_hostname" not in command for command in checks)
    for command, environment in executions:
        assert command[command.index("--server") + 1] == address
        assert command[command.index("--ca-cert") + 1] == str(directory / "server.crt")
        assert environment["FLASHDREAMS_CONTROLLER_TOKEN"] == bundle["token"].strip()
        assert bundle["token"].strip() not in " ".join(command)
    output = capsys.readouterr()
    assert bundle["token"].strip() not in output.out + output.err + repr(calls)
    if address != "10.111.69.99":
        assert f"does not cover {address}" in output.err


def test_remote_vr_resolution_failure_precedes_credential_bootstrap(
    monkeypatch, capsys
):
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(
        transport_credentials,
        "client_credentials",
        lambda *_: pytest.fail("No bootstrap before successful resolution"),
    )
    with patch.object(socket, "getaddrinfo", side_effect=socket.gaierror("not found")):
        with pytest.raises(SystemExit) as result:
            launch.main(["remote-vr", "--server", "nvidia@ssh-alias", "--accept-eula"])
    assert result.value.code == 2
    assert "IPv4" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["remote-vr", "server"])
@pytest.mark.parametrize("accept_eula", [False, True])
def test_eula_consent_is_forwarded_not_invented(mode, accept_eula):
    arguments = [mode]
    arguments += (
        ["--server", "nvidia@dgx.example"]
        if mode == "remote-vr"
        else ["--address", "192.168.1.20"]
    )
    if accept_eula:
        arguments.append("--accept-eula")
    args = launch._parser().parse_args(arguments)
    if not accept_eula:
        with pytest.raises(ValueError, match="--accept-eula"):
            launch._command(args, [], Path("/private/credentials"))
        return
    command = launch._command(args, [], Path("/private/credentials"))
    assert command.count("--isaac-teleop-accept-eula") == 1


def test_remote_usb_prepares_original_ssh_target_and_normalizes_only_child_environment(
    web_client_cache, tmp_path, monkeypatch
):
    class Executed(Exception):
        pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", "client")
    monkeypatch.setattr(launch, "_controller_address", lambda _: "10.111.69.99")
    monkeypatch.setattr(
        transport_credentials, "client_credentials", lambda *_: tmp_path
    )
    monkeypatch.setattr(transport_credentials, "read_token", lambda _: "test-token")
    monkeypatch.setattr(
        launch._setup_module(),
        "prepare_web_client",
        lambda **_: pytest.fail("Only server startup prepares assets through the SDK"),
    )
    events = []

    def copy(command, **kwargs):
        events.append("copy")
        assert Path.cwd() == tmp_path
        assert command[command.index("--") + 1 : -1] == [
            f"nvidia@dgx.example:.cloudxr/static-client/{name}" for name in _WEB_ASSETS
        ]
        for name, value in _WEB_ASSETS.items():
            (Path(command[-1]) / name).write_bytes(value)
        return subprocess.CompletedProcess(command, 0)

    def change_directory(directory):
        events.append("chdir")
        assert directory == launch.REPOSITORY
        assert {
            path.name: path.read_bytes() for path in web_client_cache.iterdir()
        } == _WEB_ASSETS

    def execute(executable, command, environment):
        events.append("exec")
        assert environment["TELEOP_WEB_CLIENT_STATIC_DIR"] == str(web_client_cache)
        assert launch.os.environ["TELEOP_WEB_CLIENT_STATIC_DIR"] == "client"
        assert command[command.index("--server") + 1] == "10.111.69.99"
        assert "--usb-local" not in command and "--cloudxr-host" not in command
        assert "--isaac-teleop-accept-eula" in command
        raise Executed

    monkeypatch.setattr(launch.subprocess, "run", copy)
    monkeypatch.setattr(launch.os, "chdir", change_directory)
    monkeypatch.setattr(launch.os, "execvpe", execute)
    with pytest.raises(Executed):
        launch.main(
            [
                "remote-vr",
                "--server",
                "nvidia@dgx.example",
                "--accept-eula",
            ]
        )
    assert events == ["copy", "chdir", "exec"]
    assert launch.os.environ["TELEOP_WEB_CLIENT_STATIC_DIR"] == "client"


@pytest.mark.parametrize("quest_device", [None, "quest-serial"])
@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
@pytest.mark.parametrize(
    "prewarm_options", [[], ["--prewarm-all-scenes"], ["--", "--prewarm-all-scenes"]]
)
def test_server_prepares_local_assets_and_forwards_scene_prewarm(
    quest_device, mode, prewarm_options, tmp_path, monkeypatch, scene_arguments
):
    class Executed(Exception):
        pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", "existing-relative-cache")
    cache = tmp_path / "existing-relative-cache"
    prepare = Mock(return_value=cache)
    monkeypatch.setattr(launch._setup_module(), "prepare_web_client", prepare)
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(
        transport_credentials, "client_credentials", lambda *_: tmp_path
    )
    monkeypatch.setattr(transport_credentials, "server_credentials", lambda _: tmp_path)
    monkeypatch.setattr(transport_credentials, "read_token", lambda _: "test-token")
    monkeypatch.setattr(
        launch,
        "_client_web_assets",
        lambda _: pytest.fail("Only remote-vr copies assets from the server"),
    )
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)

    def execute(executable, command, environment):
        prepare.assert_called_once_with(python="/custom/sidecar/bin/python")
        runtime = command[: command.index("--")]
        application = command[command.index("--") + 1 :]
        assert "--prewarm-all-scenes" not in runtime
        assert application.count("--prewarm-all-scenes") == bool(prewarm_options)
        assert environment["TELEOP_WEB_CLIENT_STATIC_DIR"] == str(cache)
        assert (
            launch.os.environ["TELEOP_WEB_CLIENT_STATIC_DIR"]
            == "existing-relative-cache"
        )
        raise Executed

    monkeypatch.setattr(launch.os, "execvpe", execute)
    arguments = [
        "server",
        "--address",
        "10.111.69.99",
        "--accept-eula",
        "--isaac-teleop-python",
        "/custom/sidecar/bin/python",
        "--mode",
        mode,
        *scene_arguments,
    ]
    if quest_device:
        arguments.extend(["--quest-device", quest_device])
    with pytest.raises(Executed):
        launch.main(arguments + prewarm_options)


def test_server_stops_before_launch_when_web_client_preparation_fails(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(transport_credentials, "server_credentials", lambda _: tmp_path)
    monkeypatch.setattr(transport_credentials, "read_token", lambda _: "test-token")
    monkeypatch.setattr(
        launch._setup_module(),
        "prepare_web_client",
        Mock(side_effect=ValueError("WebXR client download failed; retry online")),
    )
    monkeypatch.setattr(
        launch.os,
        "execvpe",
        lambda *_: pytest.fail("No launch without a prepared WebXR client"),
    )
    with pytest.raises(SystemExit) as result:
        launch.main(["server", "--accept-eula"])
    assert result.value.code == 2
    output = capsys.readouterr()
    assert "retry online" in output.err
    assert "Isaac Teleop / IWER:" not in output.out
    assert "nvidia.github.io" not in output.out + output.err


@pytest.mark.parametrize(
    "options", [["--prewarm-all-scenes"], ["--mode", "native-window"]]
)
def test_remote_vr_rejects_server_options(options):
    with pytest.raises(SystemExit) as result:
        launch._parser().parse_args(
            ["remote-vr", "--server", "nvidia@10.111.69.99", *options]
        )
    assert result.value.code == 2


@pytest.mark.parametrize(
    "options",
    [
        ["--display", "native-window"],
        ["--mode", "quest-remote-native-window"],
        ["--mode", "invalid"],
    ],
)
def test_server_rejects_nonstandard_output_options_before_setup(options, monkeypatch):
    monkeypatch.setattr(
        transport_credentials,
        "server_credentials",
        lambda *_: pytest.fail("No bootstrap before argument validation"),
    )
    with pytest.raises(SystemExit) as result:
        launch.main(["server", "--address", "10.111.69.99", *options])
    assert result.value.code == 2


def test_desktop_is_not_a_separate_server_mode():
    with pytest.raises(SystemExit) as result:
        launch._parser().parse_args(
            ["desktop", "--address", "192.168.1.20", "--accept-eula"]
        )
    assert result.value.code == 2


@pytest.mark.parametrize("mode", [None, "webrtc", "native-window"])
def test_command_shapes_keep_remote_vr_isolated_and_runtime_options_separate(mode):
    parser = launch._parser()
    directory = Path("/private/credentials")
    remote_vr = parser.parse_args(
        [
            "remote-vr",
            "--server",
            "nvidia@10.111.69.99",
            "--accept-eula",
        ]
    )
    assert remote_vr.command == "remote-vr"
    assert not hasattr(remote_vr, "mode")
    command = launch._command(remote_vr, [], directory)
    assert "--no-project" in command and "--with-editable" in command
    assert not any("[isaac" in argument for argument in command)
    assert "--package" not in command and "--require-physical-wake" in command
    assert "--cloudxr-host" not in command
    assert "--usb-local" not in command
    assert "--accept-server-certificate" not in command
    assert command[command.index("--server") + 1] == "10.111.69.99"
    server = parser.parse_args(
        [
            "server",
            "--address",
            "10.111.69.99",
            "--accept-eula",
            "--checkpoint",
            "/cache/existing",
            *(["--mode", mode] if mode else []),
        ]
    )
    assert server.command == "server"
    assert server.mode == (mode or "webrtc")
    command = launch._command(server, ["--sample", "scene"], directory)
    runtime, application = (
        command[: command.index("--")],
        command[command.index("--") + 1 :],
    )
    assert runtime[runtime.index("--mode") + 1] == (mode or "webrtc")
    assert runtime[runtime.index("--fps") + 1] == "36"
    if mode == "native-window":
        assert "--host" not in runtime and "--port" not in runtime
    else:
        assert runtime[runtime.index("--host") + 1] == "0.0.0.0"
        assert runtime[runtime.index("--port") + 1] == "8081"
    assert "--isaac-teleop-accept-eula" in runtime
    assert "--isaac-teleop-accept-eula" not in application
    assert runtime[runtime.index("--controller-host") + 1] == "10.111.69.99"
    assert runtime[runtime.index("--controller-port") + 1] == "8445"
    assert "--checkpoint" not in runtime and "--checkpoint" in application
    assert not {
        "--compile-denoise",
        "--control-fps",
        "--position-gain",
        "--auto-start",
    } & set(application)
    assert application[-2:] == ["--sample", "scene"]
    assert "--quest-device" not in runtime
    assert runtime[runtime.index("--quest-cloudxr-port") + 1] == "48322"
    assert "--quest-require-physical-wake" in runtime


@pytest.mark.parametrize("keep_open", [False, True])
@pytest.mark.parametrize("route", ["remote-vr", "webrtc", "native-window"])
def test_keep_browser_open_option_stays_in_the_controller_command(route, keep_open):
    arguments = (
        ["remote-vr", "--server", "nvidia@10.111.69.99"]
        if route == "remote-vr"
        else ["server", "--mode", route]
    )
    arguments.append("--accept-eula")
    if keep_open:
        arguments.append("--keep-browser-open-on-error")
    args = launch._parser().parse_args(arguments)
    command = launch._command(args, [], Path("/private/credentials"))
    if route == "remote-vr":
        assert ("--keep-browser-open-on-error" in command) is keep_open
        assert "--quest-keep-browser-open-on-error" not in command
    else:
        runtime, application = (
            command[: command.index("--")],
            command[command.index("--") + 1 :],
        )
        assert ("--quest-keep-browser-open-on-error" in runtime) is keep_open
        assert "--keep-browser-open-on-error" not in command
        assert "--quest-keep-browser-open-on-error" not in application


@pytest.mark.parametrize(
    "wake_options,wake_flag",
    [
        ([], "--quest-require-physical-wake"),
        (["--require-physical-wake"], "--quest-require-physical-wake"),
        (["--no-require-physical-wake"], "--no-quest-require-physical-wake"),
    ],
)
@pytest.mark.parametrize("accept_certificate", [False, True])
@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
@pytest.mark.parametrize("quest_device", [None, "quest-serial"])
def test_server_forwards_local_quest_options_with_optional_device(
    wake_options, wake_flag, accept_certificate, mode, quest_device
):
    arguments = [
        "server",
        "--address",
        "10.111.69.99",
        "--mode",
        mode,
        "--cloudxr-port",
        "48323",
        "--accept-eula",
        *wake_options,
    ]
    if quest_device:
        arguments.extend(["--quest-device", quest_device])
    if accept_certificate:
        arguments.append("--accept-server-certificate")
    args = launch._parser().parse_args(arguments)
    command = launch._command(args, ["--sample", "scene"], Path("/private/credentials"))
    runtime = command[: command.index("--")]
    application = command[command.index("--") + 1 :]
    assert runtime[runtime.index("--mode") + 1] == mode
    assert runtime[runtime.index("--quest-cloudxr-port") + 1] == "48323"
    if quest_device:
        assert runtime[runtime.index("--quest-device") + 1] == quest_device
    else:
        assert "--quest-device" not in runtime
    assert wake_flag in runtime
    assert ("--quest-accept-server-certificate" in runtime) == accept_certificate
    assert "--quest-usb-local" not in runtime
    assert "--quest-cloudxr-host" not in runtime
    assert "flashdreams-quest-relay" not in command
    assert "--server" not in command and "--ca-cert" not in command
    assert "--no-project" not in command and "--with-editable" not in command
    assert not any("quest" in option for option in application)
    assert application[-2:] == ["--sample", "scene"]


@pytest.mark.parametrize(
    "option",
    [
        "--require-physical-wake",
        "--no-require-physical-wake",
        "--accept-server-certificate",
    ],
)
def test_server_accepts_quest_automation_options_without_device(
    option, monkeypatch, capsys, scene_arguments
):
    class BootstrapReached(Exception):
        pass

    def bootstrap(*_):
        raise BootstrapReached

    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(transport_credentials, "server_credentials", bootstrap)
    with pytest.raises(BootstrapReached):
        launch.main(
            [
                "server",
                "--address",
                "10.111.69.99",
                "--accept-eula",
                option,
                *scene_arguments,
            ]
        )
    assert not capsys.readouterr().err


@pytest.mark.parametrize(
    "wake_options,wake_flag",
    [
        ([], "--require-physical-wake"),
        (["--require-physical-wake"], "--require-physical-wake"),
        (["--no-require-physical-wake"], "--no-require-physical-wake"),
    ],
)
def test_remote_vr_preserves_automation_and_port_options(wake_options, wake_flag):
    arguments = [
        "remote-vr",
        "--server",
        "nvidia@10.111.69.99",
        "--controller-port",
        "9445",
        "--cloudxr-port",
        "48323",
        "--quest-device",
        "quest-serial",
        "--accept-eula",
        "--accept-server-certificate",
        *wake_options,
    ]
    args = launch._parser().parse_args(arguments)
    command = launch._command(
        args, ["--startup-timeout", "180"], Path("/private/credentials")
    )
    assert "--cloudxr-host" not in command
    for flag, value in (
        ("--server", "10.111.69.99"),
        ("--port", "9445"),
        ("--ca-cert", "/private/credentials/server.crt"),
        ("--cloudxr-port", "48323"),
        ("--quest-device", "quest-serial"),
    ):
        assert command[command.index(flag) + 1] == value
    assert "--isaac-teleop-accept-eula" in command
    assert "--accept-server-certificate" in command
    assert wake_flag in command
    assert (
        sum(
            flag in command
            for flag in ("--require-physical-wake", "--no-require-physical-wake")
        )
        == 1
    )
    assert command[-2:] == ["--startup-timeout", "180"]


@pytest.mark.parametrize("cloudxr_port", [None, 48323])
@pytest.mark.parametrize("quest_device", [None, "quest-serial"])
@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
def test_server_advertises_available_clients_without_mutating_parent_environment(
    cloudxr_port,
    quest_device,
    mode,
    server_credentials,
    tmp_path,
    monkeypatch,
    capsys,
    scene_arguments,
):
    class Executed(Exception):
        pass

    observed = {}

    def execute(executable, command, environment):
        observed.update(command=command, environment=environment)
        raise Executed

    monkeypatch.setenv("COSMOS_CHECKPOINT_CREDENTIAL", "/existing/credential.secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("TELEOP_PROXY_HOST", "192.168.1.99")
    monkeypatch.setenv("CONTROL_TOKEN", "private-cloudxr-oob-token")
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(launch.os, "execvpe", execute)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(
        launch._setup_module(), "prepare_web_client", lambda **_: tmp_path / "client"
    )
    monkeypatch.setattr(
        transport_credentials, "server_credentials", lambda _: server_credentials
    )
    monkeypatch.setattr(
        transport_credentials,
        "client_credentials",
        lambda *_: pytest.fail("Local Quest must not bootstrap a relay client"),
    )
    arguments = [
        "server",
        "--address",
        "192.168.1.20",
        "--accept-eula",
        "--gpu",
        "1",
        "--mode",
        mode,
        *scene_arguments,
    ]
    if cloudxr_port is not None:
        arguments.extend(["--cloudxr-port", str(cloudxr_port)])
    if quest_device is not None:
        arguments.extend(
            ["--quest-device", quest_device, "--accept-server-certificate"]
        )
    with pytest.raises(Executed):
        launch.main(arguments)
    environment = observed["environment"]
    port = cloudxr_port or 48322
    assert (
        environment["TELEOP_PROXY_HOST"]
        == environment["TELEOP_STREAM_SERVER_IP"]
        == "192.168.1.20"
    )
    assert environment["PROXY_PORT"] == str(port)
    assert environment["TELEOP_STREAM_PORT"] == str(port)
    assert environment["COSMOS_CHECKPOINT_CREDENTIAL"] == "/existing/credential.secret"
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert launch.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert launch.os.environ["TELEOP_PROXY_HOST"] == "192.168.1.99"
    assert "--auto-start" not in observed["command"]
    assert "--isaac-teleop-accept-eula" in observed["command"]
    assert mode in observed["command"]
    token = transport_credentials.read_token(server_credentials)
    assert environment["FLASHDREAMS_CONTROLLER_TOKEN"] == token
    output = capsys.readouterr().out
    if mode == "webrtc":
        assert "Operator viewer: http://192.168.1.20:8081/" in output
    else:
        assert "Operator viewer: native window on the model host" in output
        assert "http://192.168.1.20:8081/" not in output
    assert f"CloudXR certificate: https://192.168.1.20:{port}/" in output
    assert (
        f"https://192.168.1.20:{port}/client/"
        f"?serverIP=192.168.1.20&port={port}&oobEnable=1"
    ) in output
    assert "nvidia.github.io" not in output
    command = observed["command"]
    assert command[command.index("--quest-cloudxr-port") + 1] == str(port)
    assert token not in output and token not in " ".join(observed["command"])
    assert "private-cloudxr-oob-token" not in output


@pytest.mark.parametrize("mode", ["webrtc", "native-window"])
@pytest.mark.parametrize("address_options", [[], ["--address", "127.0.0.1"]])
def test_local_server_does_not_require_a_network_interface(
    mode, address_options, tmp_path, monkeypatch, capsys, scene_arguments
):
    class Executed(Exception):
        pass

    def no_network(*_args, **_kwargs):
        pytest.fail("Local startup must not discover a LAN address or use SSH")

    def credentials(address):
        assert address == "127.0.0.1"
        return tmp_path

    def execute(executable, command, environment):
        runtime = command[: command.index("--")]
        assert runtime[runtime.index("--controller-host") + 1] == "127.0.0.1"
        assert runtime[runtime.index("--mode") + 1] == mode
        if mode == "webrtc":
            assert runtime[runtime.index("--host") + 1] == "127.0.0.1"
        else:
            assert "--host" not in runtime and "--port" not in runtime
        assert environment["TELEOP_PROXY_HOST"] == "127.0.0.1"
        assert environment["TELEOP_STREAM_SERVER_IP"] == "127.0.0.1"
        assert environment["UV_OFFLINE"] == "1"
        assert environment["UV_NO_SYNC"] == "1"
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert "--quest-require-physical-wake" in runtime
        assert "--quest-accept-server-certificate" in runtime
        raise Executed

    monkeypatch.setenv("TELEOP_PROXY_HOST", "10.111.69.99")
    monkeypatch.setenv("TELEOP_STREAM_SERVER_IP", "10.111.69.99")
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_NO_SYNC", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cache = tmp_path / "client"
    cache.mkdir()
    for name, value in _WEB_ASSETS.items():
        (cache / name).write_bytes(value)
    monkeypatch.setenv("TELEOP_WEB_CLIENT_STATIC_DIR", str(cache))
    monkeypatch.setattr(launch._setup_module(), "_run", no_network)
    monkeypatch.setattr(launch.socket, "getaddrinfo", no_network)
    monkeypatch.setattr(launch.socket, "socket", no_network)
    monkeypatch.setattr(transport_credentials, "_run", no_network)
    monkeypatch.setattr(transport_credentials, "server_credentials", credentials)
    monkeypatch.setattr(transport_credentials, "client_credentials", no_network)
    monkeypatch.setattr(transport_credentials, "read_token", lambda _: "test-token")
    monkeypatch.setattr(launch.shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(launch.os, "execvpe", execute)
    with pytest.raises(Executed):
        launch.main(
            [
                "server",
                "--mode",
                mode,
                "--accept-eula",
                "--accept-server-certificate",
                *scene_arguments,
                *address_options,
            ]
        )
    assert "https://127.0.0.1:48322/" in capsys.readouterr().out
    assert launch.os.environ["TELEOP_PROXY_HOST"] == "10.111.69.99"


def test_setup_alias_dispatches_without_bootstrapping_transport(monkeypatch):
    from unittest.mock import Mock

    installer = Mock()
    monkeypatch.setattr(launch, "_setup_module", lambda: installer)
    monkeypatch.setattr(
        transport_credentials, "_run", lambda _: pytest.fail("No transport bootstrap")
    )
    launch.main(["setup-isaac-teleop", "--accept-eula", "--python", "python3.12"])
    installer.main.assert_called_once_with(
        [
            "--accept-eula",
            "--python",
            "python3.12",
        ]
    )


def test_command_builder_never_forwards_unaccepted_terms():
    args = launch._parser().parse_args(["server", "--address", "10.111.69.99"])
    with pytest.raises(ValueError, match="--accept-eula"):
        launch._command(args, [], Path("/credentials"))


def test_eula_failure_identifies_exact_terms_without_preparing_assets(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        launch._setup_module(),
        "prepare_web_client",
        lambda **_: pytest.fail("No asset preparation without explicit consent"),
    )
    with pytest.raises(SystemExit):
        launch.main(["server", "--address", "10.111.69.99"])
    output = capsys.readouterr().err
    assert "1.4.142" in output
    assert (
        "https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE"
        in output
    )


@pytest.mark.parametrize("mode", ["server", "remote-vr"])
def test_custom_sidecar_interpreter_reaches_each_isolated_process(mode):
    arguments = [
        mode,
        "--accept-eula",
        "--isaac-teleop-python",
        "/custom/sidecar/bin/python",
    ]
    arguments += (
        ["--address", "10.111.69.99"]
        if mode == "server"
        else ["--server", "user@10.111.69.99"]
    )
    args = launch._parser().parse_args(arguments)
    command = launch._command(args, [], Path("/credentials"))
    selected = [
        command[i + 1]
        for i, argument in enumerate(command)
        if argument == "--isaac-teleop-python"
    ]
    assert selected == ["/custom/sidecar/bin/python"] * (2 if mode == "server" else 1)


@pytest.mark.parametrize("mode", ["server", "remote-vr"])
def test_eula_is_required_before_any_bootstrap(mode, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail(
            "Unaccepted launch must not bootstrap credentials, assets, or subprocesses"
        )

    for name in ("server_credentials", "client_credentials", "_run"):
        monkeypatch.setattr(transport_credentials, name, unexpected)
    for name in ("_client_web_assets", "_controller_address"):
        monkeypatch.setattr(launch, name, unexpected)
    monkeypatch.setattr(launch.os, "execvpe", unexpected)
    arguments = (
        ["server", "--address", "10.111.69.99"]
        if mode == "server"
        else ["remote-vr", "--server", "nvidia@dgx.example"]
    )
    with pytest.raises(SystemExit) as result:
        launch.main(arguments)
    assert result.value.code == 2


@pytest.mark.parametrize("mode", ["server", "remote-vr"])
def test_default_sidecar_is_shared_with_setup_outside_checkout(
    mode, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    arguments = [mode, "--accept-eula"]
    arguments += (
        ["--address", "10.111.69.99"]
        if mode == "server"
        else ["--server", "user@10.111.69.99"]
    )
    args = launch._parser().parse_args(arguments)
    command = launch._command(args, [], Path("/credentials"))
    selected = [
        command[i + 1]
        for i, argument in enumerate(command)
        if argument == "--isaac-teleop-python"
    ]
    python = launch._setup_module().DEFAULT_ISAAC_TELEOP_DIRECTORY / "bin/python"
    assert python.is_absolute()
    assert selected == [str(python)] * (2 if mode == "server" else 1)
