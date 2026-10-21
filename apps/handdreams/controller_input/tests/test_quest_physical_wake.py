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

"""CPU subprocess checks for the Quest physical-wake launch gate."""

from __future__ import annotations

import os
import select
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


def _wait(predicate, timeout: float = 8.0) -> None:
    """Wait for a subprocess condition within a bounded test deadline."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Quest launcher condition timed out"
        time.sleep(0.01)


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _wake(power_file: Path) -> None:
    """Publish a wake transition without exposing a truncated power-state file."""
    next_power = power_file.with_suffix(".next")
    next_power.write_text("Awake")
    next_power.replace(power_file)


@contextmanager
def _launcher(
    tmp_path: Path,
    initial_power: str,
    *,
    ignore_sleep: bool = False,
    fail_sleep: bool = False,
    meta_pin: str | None = "",
    properties: str = "disable_autosleep=false, set_proximity_close=false",
    set_success: bool = True,
    require_physical_wake: bool = True,
    block_hold: bool = False,
):
    """Run the real headset hold against a controllable fake ADB."""
    power_file = tmp_path / "power"
    power_file.write_text(initial_power)
    adb_log = tmp_path / "adb.log"
    adb_log.touch()
    fake_adb = tmp_path / "adb"
    fake_adb.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$QUEST_TEST_ADB_LOG"
case " $* " in
    *" get-state "*) printf 'device\\n' ;;
    *" --method GET_PROPERTY "*) printf 'Result: Bundle[{%s}]\\n' "$QUEST_TEST_PROPERTIES" ;;
    *" --method SET_PROPERTY "*) printf 'Result: Bundle[{Success=%s}]\\n' "$QUEST_TEST_SET_SUCCESS" ;;
    *" com.oculus.vrpowermanager.prox_close "*)
        if [[ -n "$QUEST_TEST_HOLD_RELEASE" ]]; then
            while [[ ! -e "$QUEST_TEST_HOLD_RELEASE" ]]; do sleep 0.01; done
        fi
        ;;
    *" shell dumpsys power "*)
        power="$(< "$QUEST_TEST_POWER_FILE")"
        printf 'POWER:%s\\n' "$power" >> "$QUEST_TEST_ADB_LOG"
        case "$power" in
            adb-error) printf 'device offline\\n' >&2; exit 1 ;;
            malformed) printf 'Power manager unavailable\\n' ;;
            *) printf '  mWakefulness=%s\\n' "$power" ;;
        esac
        ;;
    *" shell input keyevent KEYCODE_SLEEP "*)
        [[ "$QUEST_TEST_FAIL_SLEEP" == 1 ]] && exit 1
        if [[ "$QUEST_TEST_IGNORE_SLEEP" == 0 ]]; then
            printf 'Asleep' > "$QUEST_TEST_POWER_FILE"
        fi
        ;;
esac
exit 0
"""
    )
    fake_adb.chmod(0o700)
    owner = subprocess.Popen(["sleep", "60"])
    launcher = None
    ready_read, ready_write = os.pipe()
    try:
        os.set_blocking(ready_read, False)
        environment = os.environ.copy()
        environment.update(
            {
                "QUEST_ADB": str(fake_adb),
                "QUEST_LAUNCHER_LOCK_FILE": str(tmp_path / "launcher.lock"),
                "QUEST_OWNER_PID": str(owner.pid),
                "QUEST_REQUIRE_PHYSICAL_WAKE": str(int(require_physical_wake)),
                "QUEST_SERIAL": "quest-test",
                "QUEST_WAKE_READY_FD": str(ready_write),
                "QUEST_TEST_ADB_LOG": str(adb_log),
                "QUEST_TEST_IGNORE_SLEEP": str(int(ignore_sleep)),
                "QUEST_TEST_FAIL_SLEEP": str(int(fail_sleep)),
                "QUEST_TEST_POWER_FILE": str(power_file),
                "QUEST_TEST_PROPERTIES": properties,
                "QUEST_TEST_SET_SUCCESS": str(set_success).lower(),
                "QUEST_TEST_HOLD_RELEASE": str(tmp_path / "release-hold")
                if block_hold
                else "",
            }
        )
        environment.pop("QUEST_META_PIN", None)
        if meta_pin is not None:
            environment["QUEST_META_PIN"] = meta_pin
        pin_file = tmp_path / "saved-pin"
        pin_file.write_text("4567\n")
        environment["QUEST_META_PIN_FILE"] = str(pin_file)
        script = (
            Path(__file__).resolve().parents[1]
            / "flashdreams_controller_input"
            / "quest-offhead-teleop.sh"
        )
        launcher = subprocess.Popen(
            ["bash", str(script)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            pass_fds=(ready_write,),
        )
        os.close(ready_write)
        ready_write = None
        yield launcher, owner, power_file, adb_log, ready_read
    finally:
        if launcher is not None:
            _stop(launcher)
            assert launcher.stdout is not None
            launcher.stdout.close()
        _stop(owner)
        os.close(ready_read)
        if ready_write is not None:
            os.close(ready_write)


def _assert_no_startup(calls: str) -> None:
    for command in (
        "--method SET_PROPERTY",
        "prox_close",
        " reverse ",
        "shell am start",
        "KEYCODE_HOME",
        "KEYCODE_WAKEUP",
        "KEYCODE_POWER",
    ):
        assert command not in calls


@pytest.mark.parametrize("initial_power", ["Awake", "Asleep"])
def test_physical_wake_precedes_browser_hold(tmp_path, initial_power) -> None:
    with _launcher(tmp_path, initial_power) as (
        launcher,
        _,
        power_file,
        adb_log,
        ready_fd,
    ):
        _wait(lambda: adb_log.read_text().count("POWER:Asleep") >= 4)
        waiting_calls = adb_log.read_text()
        assert launcher.poll() is None
        assert os.read(ready_fd, 1) == b"W"
        with pytest.raises(BlockingIOError):
            os.read(ready_fd, 1)
        _assert_no_startup(waiting_calls)
        assert waiting_calls.count("KEYCODE_SLEEP") == int(initial_power == "Awake")
        if initial_power == "Awake":
            assert (
                waiting_calls.index("POWER:Awake")
                < waiting_calls.index("KEYCODE_SLEEP")
                < waiting_calls.index("POWER:Asleep")
            )

        _wake(power_file)
        _wait(lambda: bool(select.select([ready_fd], [], [], 0)[0]))
        assert os.read(ready_fd, 2) == b"1"
        assert os.read(ready_fd, 1) == b""
        calls = adb_log.read_text()
        assert calls.rindex("POWER:Awake") < calls.index("prox_close")
        _assert_no_native_operations(calls)
        assert "KEYCODE_WAKEUP" not in calls
        assert "KEYCODE_POWER" not in calls


@pytest.mark.parametrize("cancel", ["term", "owner_exit"])
def test_cancelling_physical_wake_wait_cannot_launch_later(tmp_path, cancel) -> None:
    with _launcher(tmp_path, "Asleep") as (
        launcher,
        owner,
        power_file,
        adb_log,
        ready_fd,
    ):
        _wait(lambda: adb_log.read_text().count("POWER:Asleep") >= 4)
        assert os.read(ready_fd, 1) == b"W"
        if cancel == "term":
            launcher.terminate()
        else:
            _stop(owner)
        _wake(power_file)
        launcher.wait(timeout=8)
        assert os.read(ready_fd, 1) == b""
        _assert_no_startup(adb_log.read_text())


@pytest.mark.parametrize("power", ["malformed", "Unknown", "adb-error"])
def test_invalid_power_query_aborts_without_starting(tmp_path, power) -> None:
    with _launcher(tmp_path, power) as (launcher, _, _, adb_log, ready_fd):
        assert launcher.wait(timeout=8) != 0
        assert os.read(ready_fd, 1) == b""
        calls = adb_log.read_text()
        assert "shell dumpsys power" in calls
        _assert_no_startup(calls)


def test_unconfirmed_sleep_times_out_without_starting(tmp_path) -> None:
    with _launcher(tmp_path, "Awake", ignore_sleep=True) as (
        launcher,
        _,
        _,
        adb_log,
        ready_fd,
    ):
        assert launcher.wait(timeout=15) != 0
        assert os.read(ready_fd, 1) == b""
        assert "could not confirm Quest sleep" in launcher.stdout.read()
        calls = adb_log.read_text()
        assert calls.count("KEYCODE_SLEEP") == 2  # Wake gate, then shutdown.
        assert calls.rstrip().endswith("shell input keyevent KEYCODE_SLEEP")
        _assert_no_startup(calls)


def _assert_no_native_operations(calls: str) -> None:
    for command in (
        "shell pm path",
        "shell am start",
        "shell am force-stop",
        "shell pidof",
        " reverse ",
        "KEYCODE_HOME",
        "KEYCODE_WAKEUP",
        "KEYCODE_POWER",
        "disable_guardian:b:",
        "disable_dialogs:b:",
    ):
        assert command not in calls


@pytest.mark.parametrize("initial_power", ["Awake", "Asleep"])
@pytest.mark.parametrize("original", ["true", "false"])
def test_browser_hold_restores_explicit_pin_settings_without_native_launch(
    tmp_path, initial_power, original
) -> None:
    with _launcher(
        tmp_path,
        initial_power,
        meta_pin="7890",
        properties=f"disable_autosleep={original}, set_proximity_close={original}",
    ) as (launcher, _, power_file, adb_log, ready_fd):
        _wait(lambda: adb_log.read_text().count("POWER:Asleep") >= 4)
        waiting = adb_log.read_text()
        assert waiting.index("GET_PROPERTY") < waiting.index("prox_far")
        assert "SET_PROPERTY" not in waiting
        assert os.read(ready_fd, 1) == b"W"
        with pytest.raises(BlockingIOError):
            os.read(ready_fd, 1)
        _wake(power_file)
        _wait(lambda: "prox_close" in adb_log.read_text())
        _wait(lambda: bool(select.select([ready_fd], [], [], 0)[0]))
        assert os.read(ready_fd, 1) == b"1"
        calls = adb_log.read_text()
        assert calls.index("POWER:Awake") < calls.index("disable_autosleep:b:true")
        assert calls.index("disable_autosleep:b:true") < calls.index("prox_close")
        _stop(launcher)
        calls = adb_log.read_text()
        _assert_no_native_operations(calls)
        assert f"disable_autosleep:b:{original}" in calls.split("prox_close", 1)[1]
        last_proximity = [
            line for line in calls.splitlines() if "vrpowermanager.prox_" in line
        ][-1]
        assert last_proximity.endswith(
            "prox_close" if original == "true" else "prox_far"
        )
        assert calls.rstrip().endswith(
            "-s quest-test shell input keyevent KEYCODE_SLEEP"
        )
        assert power_file.read_text() == "Asleep"


@pytest.mark.parametrize("initial_power", ["Awake", "Asleep"])
@pytest.mark.parametrize("cancel", ["term", "owner_exit"])
def test_browser_hold_without_pin_ignores_saved_pin_and_sets_no_properties(
    tmp_path, initial_power, cancel
) -> None:
    with _launcher(
        tmp_path,
        initial_power,
        meta_pin=None,
        properties="disable_autosleep=false",
        require_physical_wake=False,
    ) as (launcher, owner, power_file, adb_log, ready_fd):
        _wait(lambda: "prox_close" in adb_log.read_text())
        _wait(lambda: bool(select.select([ready_fd], [], [], 0)[0]))
        assert os.read(ready_fd, 1) == b"1"
        if cancel == "term":
            launcher.terminate()
        else:
            _stop(owner)
        assert launcher.wait(timeout=8) == 0
        calls = adb_log.read_text()
        _assert_no_native_operations(calls)
        assert "GET_PROPERTY" in calls
        assert "SET_PROPERTY" not in calls
        assert "prox_far" in calls
        assert calls.count("KEYCODE_SLEEP") == 1
        assert calls.rstrip().endswith(
            "-s quest-test shell input keyevent KEYCODE_SLEEP"
        )
        assert power_file.read_text() == "Asleep"
        output = launcher.stdout.read()
        assert "without changing protected settings" in output
        assert "normal headset behavior restored" not in output


def test_shutdown_sleep_failure_warns_after_restoring_proximity(tmp_path) -> None:
    with _launcher(tmp_path, "Awake", require_physical_wake=False, fail_sleep=True) as (
        launcher,
        _,
        _,
        adb_log,
        ready_fd,
    ):
        _wait(lambda: bool(select.select([ready_fd], [], [], 0)[0]))
        assert os.read(ready_fd, 1) == b"1"
        _stop(launcher)
        assert launcher.returncode == 0
        calls = adb_log.read_text()
        assert calls.index("prox_far") < calls.index("KEYCODE_SLEEP")
        assert "could not put Quest to sleep" in launcher.stdout.read()


@pytest.mark.parametrize("cancel", ["term", "owner_exit"])
def test_browser_wake_cancellation_restores_proximity_without_startup(
    tmp_path, cancel
) -> None:
    with _launcher(tmp_path, "Asleep") as (
        launcher,
        owner,
        power_file,
        adb_log,
        ready_fd,
    ):
        _wait(lambda: adb_log.read_text().count("POWER:Asleep") >= 4)
        assert os.read(ready_fd, 1) == b"W"
        if cancel == "term":
            launcher.terminate()
        else:
            _stop(owner)
        _wake(power_file)
        launcher.wait(timeout=8)
        assert os.read(ready_fd, 1) == b""
        calls = adb_log.read_text()
        _assert_no_native_operations(calls)
        assert "SET_PROPERTY" not in calls
        assert "prox_close" not in calls
        assert calls.count("prox_far") == 2


@pytest.mark.parametrize(
    "properties", ["disable_autosleep=unknown", "error=permission denied"]
)
def test_browser_pin_requires_a_readable_original_autosleep(
    tmp_path, properties
) -> None:
    with _launcher(tmp_path, "Awake", meta_pin="7890", properties=properties) as (
        launcher,
        _,
        _,
        adb_log,
        ready_fd,
    ):
        assert launcher.wait(timeout=8) != 0
        assert os.read(ready_fd, 1) == b""
        calls = adb_log.read_text()
        _assert_no_native_operations(calls)
        assert "SET_PROPERTY" not in calls
        assert "prox_far" not in calls
        assert "KEYCODE_SLEEP" not in calls


def test_browser_rejected_pin_never_becomes_ready_and_attempts_restore(
    tmp_path,
) -> None:
    with _launcher(
        tmp_path,
        "Awake",
        meta_pin="7890",
        set_success=False,
        require_physical_wake=False,
    ) as (launcher, _, _, adb_log, ready_fd):
        assert launcher.wait(timeout=8) != 0
        assert os.read(ready_fd, 1) == b""
        calls = adb_log.read_text()
        _assert_no_native_operations(calls)
        assert "disable_autosleep:b:true" in calls
        assert "disable_autosleep:b:false" in calls
        assert "prox_close" not in calls
        assert calls.rstrip().endswith("shell input keyevent KEYCODE_SLEEP")
        output = launcher.stdout.read()
        assert "rejected the auto-sleep change" in output
        assert "could not restore previous Quest auto-sleep" in output


@pytest.mark.parametrize("require_physical_wake", [False, True])
def test_browser_ready_waits_for_hold_and_excludes_overlapping_launcher(
    tmp_path, require_physical_wake
) -> None:
    with _launcher(
        tmp_path,
        "Awake",
        require_physical_wake=require_physical_wake,
        block_hold=True,
    ) as (launcher, _, power_file, adb_log, ready_fd):
        if require_physical_wake:
            _wait(lambda: adb_log.read_text().count("POWER:Asleep") >= 4)
            assert os.read(ready_fd, 1) == b"W"
            _wake(power_file)
        _wait(lambda: "prox_close" in adb_log.read_text())
        with pytest.raises(BlockingIOError):
            os.read(ready_fd, 1)
        (tmp_path / "release-hold").touch()
        _wait(lambda: bool(select.select([ready_fd], [], [], 0)[0]))
        assert os.read(ready_fd, 1) == b"1"
        sleep_count = adb_log.read_text().count("KEYCODE_SLEEP")
        overlap = subprocess.run(
            launcher.args,
            env={
                **os.environ,
                "QUEST_ADB": str(tmp_path / "adb"),
                "QUEST_SERIAL": "quest-test",
                "QUEST_OWNER_PID": str(os.getpid()),
                "QUEST_LAUNCHER_LOCK_FILE": str(tmp_path / "launcher.lock"),
                "QUEST_TEST_ADB_LOG": str(adb_log),
            },
            capture_output=True,
            text=True,
            timeout=8,
        )
        assert overlap.returncode != 0
        assert "another Quest launcher is already running" in overlap.stderr
        assert launcher.poll() is None
        assert adb_log.read_text().count("KEYCODE_SLEEP") == sleep_count
        assert power_file.read_text() == "Awake"
        _assert_no_native_operations(adb_log.read_text())
