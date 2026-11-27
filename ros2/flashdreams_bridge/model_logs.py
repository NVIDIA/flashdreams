# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
"""Forward the local model's systemd user journal to the ROS log topic."""

import json
import re
import subprocess
from threading import Event, Thread

from builtin_interfaces.msg import Time
from rcl_interfaces.msg import Log
from rclpy.logging import LoggingSeverity as Severity
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


MODEL_UNIT = "flashdreams-ros-model.service"
_SYSLOG = (
    Severity.FATAL,
    Severity.FATAL,
    Severity.FATAL,
    Severity.ERROR,
    Severity.WARN,
    Severity.INFO,
    Severity.INFO,
    Severity.DEBUG,
)
_LOGURU = re.compile(
    r"^\d{4}-\d\d-\d\d [\d:.]+ \| (TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s*\|"
)
_LEVELS = dict(
    TRACE=Severity.DEBUG,
    DEBUG=Severity.DEBUG,
    INFO=Severity.INFO,
    SUCCESS=Severity.INFO,
    WARNING=Severity.WARN,
    ERROR=Severity.ERROR,
    CRITICAL=Severity.FATAL,
)


def _message(entry):
    text = entry["MESSAGE"]
    if isinstance(text, list):  # journalctl encodes non-UTF8 messages as byte arrays.
        text = bytes(text).decode(errors="replace")
    priority = int(entry.get("PRIORITY", 6))
    level = _SYSLOG[priority] if 0 <= priority < len(_SYSLOG) else Severity.INFO
    match = _LOGURU.match(text)
    if match:
        level = _LEVELS[match[1]]
    elif level == Severity.INFO and (
        text.startswith("Traceback (most recent call last):")
        or re.match(r"^\w*(?:Error|Exception|Interrupt):", text)
    ):
        level = Severity.ERROR
    sec, us = divmod(int(entry["__REALTIME_TIMESTAMP"]), 1_000_000)
    return Log(
        stamp=Time(sec=sec, nanosec=us * 1000),
        level=level,
        name=MODEL_UNIT,
        msg=text.replace("\0", "�"),
    )


class ModelLogs:
    def __init__(self, node, unit):
        self.process = None
        if not unit:
            return
        if unit != MODEL_UNIT:
            raise ValueError(f"model_log_unit must be empty or {MODEL_UNIT}")
        self.node = node
        self.stopping = Event()
        self.publisher = node.create_publisher(
            Log,
            "/rosout",
            QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        try:
            self.process = subprocess.Popen(
                [
                    "journalctl",
                    "--user",
                    f"--user-unit={unit}",
                    "--boot",
                    "--follow",
                    "--all",
                    "--lines=100",
                    "--output=json",
                    "--output-fields=MESSAGE,PRIORITY",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            )
        except OSError as exc:
            node.get_logger().warning(f"Cannot follow {unit} journal: {exc}")
            return
        self.thread = Thread(target=self._read, name="model-journal", daemon=True)
        self.thread.start()

    def _read(self):
        reported = False
        with self.process.stdout as records:
            for line in records:
                if self.stopping.is_set() or not self.node.context.ok():
                    break
                try:
                    message = _message(json.loads(line))
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    AssertionError,
                    OverflowError,
                ) as exc:
                    if not reported:
                        self.node.get_logger().warning(
                            f"Cannot read {MODEL_UNIT} journal record: {line.strip()[:160]} ({exc})"
                        )
                        reported = True
                    continue
                try:
                    self.publisher.publish(message)
                except RuntimeError as exc:
                    if not self.stopping.is_set() and self.node.context.ok():
                        self.node.get_logger().error(
                            f"Cannot forward {MODEL_UNIT} logs: {exc}"
                        )
                    break
        if not self.stopping.is_set() and self.node.context.ok():
            self.node.get_logger().warning(f"Journal stream for {MODEL_UNIT} ended")

    def close(self):
        if self.process is None:
            return
        self.stopping.set()
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join()
        self.process = None
