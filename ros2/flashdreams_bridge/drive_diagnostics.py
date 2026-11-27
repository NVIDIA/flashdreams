# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Report observation rates against both wall time and the model clock."""

from collections import deque
from threading import Lock
from time import monotonic

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.clock import Clock, ClockType


class DriveDiagnostics:
    def __init__(self, node):
        self.node = node
        self.publisher = node.create_publisher(DiagnosticArray, "/diagnostics", 1)
        self.lock = Lock()
        self.cameras = ()
        self.frames = 0
        self.source_us = None
        self.period_us = None
        self.observed_at = None
        self.chunk_duration = None
        self.history = deque(maxlen=6)
        node.create_timer(1.0, self.publish, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def observe(self, source_us, cameras, reset):
        with self.lock:
            if reset:
                self.cameras = tuple(cameras)
                self.frames = 0
                self.history.clear()
                self.source_us = self.period_us = self.chunk_duration = None
            elif self.source_us is not None:
                self.period_us = source_us - self.source_us
            self.source_us = source_us
            self.observed_at = monotonic()
            self.frames += 1

    def step_done(self, duration):
        with self.lock:
            self.chunk_duration = duration

    def publish(self):
        now = monotonic()
        with self.lock:
            rates = {"episode_frames": self.frames}
            if self.source_us is not None:
                self.history.append((now, self.frames, self.source_us))
                since, frames, source = self.history[0]
                elapsed = now - since
                if elapsed > 0:
                    rates["publish_hz_wall"] = round((self.frames - frames) / elapsed, 2)
                    rates["realtime_factor"] = round((self.source_us - source) / (1e6 * elapsed), 3)
                if self.period_us:
                    rates["sample_hz_sim"] = round(1e6 / self.period_us, 2)
                rates["last_source_us"] = self.source_us
                rates["last_frame_age_s"] = round(now - self.observed_at, 2)
            cameras = self.cameras
            chunk_duration = self.chunk_duration
        node = self.node
        active = node.active_request
        joy = node.joy
        error = node.last_error
        if error:
            phase = error
        elif active is not None:
            phase = active["type"]
        elif node.needs_reset:
            phase = "waiting for reset"
        elif joy is not None and (joy.latest is None or now - joy.received > joy.timeout):
            phase = "waiting for fresh joy"
        else:
            phase = "waiting for control"
        values = dict(rates, episode=node.episode)
        if joy is not None and joy.received:
            values["joy_age_s"] = round(now - joy.received, 3)
        if chunk_duration is not None:
            values["last_step_wall_s"] = round(chunk_duration, 3)

        def status(name, message, values):
            return DiagnosticStatus(
                name=name, hardware_id="interactive_drive", message=message,
                level=DiagnosticStatus.WARN if error else DiagnosticStatus.OK,
                values=[KeyValue(key=key, value=str(value)) for key, value in values.items()],
            )

        msg = DiagnosticArray()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.status = [status("flashdreams/drive", phase, values)]
        for camera in cameras:
            msg.status.append(status("flashdreams/camera/" + camera, "image_raw + camera_info", rates))
        self.publisher.publish(msg)
