# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Drive the simulation from the latest ROS Joy sample at each chunk boundary."""

import math
from threading import Condition, Lock, Thread
from time import monotonic
from uuid import uuid4

from flashdreams_bridge.cockpit_joy import CockpitJoy
from flashdreams_interfaces.srv import Reset, Step
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Joy


class JoyDriver:
    def __init__(self, node):
        self.node = node
        self.timeout = node.declare_parameter("joy_timeout_s", 0.75).value
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("joy_timeout_s must be finite and positive")
        self.mapper = CockpitJoy()
        self.inputs = Condition()
        self.latest = None
        self.received = 0.0
        self.reset_pending = False
        self.brake_pending = False
        self.closed = False
        self.selection = Lock()
        self.selected_index = 0
        self.views = []
        self.selected_image = node.create_publisher(
            Image, "/sim/camera/selected/image_raw", node.qos
        )
        self.selected_info = node.create_publisher(
            CameraInfo, "/sim/camera/selected/camera_info", node.qos
        )
        # The default mutually exclusive callback group consumes button edges in order.
        # GPU work runs separately so short taps arrive during generation.
        node.create_subscription(
            Joy,
            "/joy",
            self.on_joy,
            QoSProfile(depth=64, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.worker = Thread(target=self.drive, name="joy-drive")
        self.worker.start()

    def release_input(self):
        self.mapper.release()
        self.latest = None
        self.brake_pending = False
        self.reset_pending = False

    def on_joy(self, msg):
        now = monotonic()
        with self.inputs:
            if now - self.received > self.timeout:
                self.release_input()
            try:
                sample = self.mapper.update(msg.axes, msg.buttons)
            except ValueError as exc:
                # An empty/disconnected device or bad layout must release held inputs.
                self.release_input()
                self.node.get_logger().warning(str(exc), throttle_duration_sec=2.0)
                return
            if sample.handbrake and (self.latest is None or not self.latest.handbrake):
                self.brake_pending = True
            self.reset_pending |= sample.reset
            self.latest = sample
            self.received = now
            self.inputs.notify()
        if sample.camera_delta:
            with self.selection:
                self.selected_index += sample.camera_delta
                self.publish_selected()

    def present(self, views):
        """Publish the selected camera and retain the images for immediate switching."""
        with self.selection:
            self.views = views
            self.publish_selected()

    def publish_selected(self):
        if self.views:
            self.selected_index %= len(self.views)
            pixels, info = self.views[self.selected_index]
            self.selected_info.publish(info)
            self.selected_image.publish(pixels)

    def drive(self):
        next_step = None
        while self.node.context.ok():
            with self.inputs:
                self.inputs.wait_for(
                    lambda: self.closed
                    or (
                        self.latest is not None
                        and monotonic() - self.received <= self.timeout
                    )
                )
                if self.closed:
                    return
                reset = self.reset_pending or next_step is None
                self.reset_pending = False
                sample = self.latest
                brake = self.brake_pending
                if not reset:
                    self.brake_pending = False
            if reset:
                request = {
                    "type": "reset",
                    "episode_id": str(uuid4()),
                    "previous_episode_id": self.node.episode,
                }
                response = self.node.exchange(request, Reset.Response())
            else:
                # Direct manual steering uses a 0.4 rad full scale in Interactive Drive.
                request = {
                    "type": "step",
                    "episode_id": self.node.episode,
                    "expected_step": next_step,
                    "manual_gear": sample.gear,
                    "steering_angle_rad": 0.4 * sample.steer,
                    "throttle": sample.throttle,
                    "brake": 1.0 if brake else sample.brake,
                    "reverse": sample.gear == "R",
                }
                response = self.node.exchange(request, Step.Response())
            if response.code == "OK":
                next_step = response.next_step
                continue
            if self.node.stopping():
                return
            self.node.get_logger().warning(
                f'Joy {request["type"]} failed ({response.code}): {response.message}',
                throttle_duration_sec=2.0,
            )
            if response.code != "INVALID_REQUEST":
                next_step = None
            # Retry recoverable failures against a newly observed state, without spinning.
            with self.inputs:
                if self.closed:
                    return
                self.inputs.wait_for(lambda: self.closed, timeout=0.5)

    def close(self):
        with self.inputs:
            self.closed = True
            self.inputs.notify()
        self.worker.join()
