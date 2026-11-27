#!/usr/bin/python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
"""ROS 2 Humble adapter for a single Interactive Drive lockstep endpoint."""

from __future__ import annotations

from array import array
import math
import re
import socket
import threading
from time import monotonic

from builtin_interfaces.msg import Time
from flashdreams_bridge.camera_calibration import camera_info
from flashdreams_bridge.drive_diagnostics import DriveDiagnostics
from flashdreams_bridge.ego_visualization import EgoVisualization
from flashdreams_bridge.joy_driver import JoyDriver
from flashdreams_bridge.model_logs import ModelLogs
from flashdreams_bridge.scene_visualization import SceneVisualization
from flashdreams_bridge.ros_endpoint import (
    NIL_UUID,
    read_message,
    validate_request,
    write_message,
)
from flashdreams_interfaces.msg import CameraModel, Control, VehicleState
from flashdreams_interfaces.srv import Reset, Step
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry
import PyKDL as kdl
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def pose_matrix(matrix):
    rows = matrix[:3]
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Expected a 4x4 pose matrix")
    return kdl.Frame(
        kdl.Rotation(*(v for row in rows for v in row[:3])),
        kdl.Vector(*(row[3] for row in rows)),
    )


def vector(values):
    return Vector3(x=values[0], y=values[1], z=values[2])


def transform(parent, child, stamp, pose):
    result = TransformStamped(
        header=Header(frame_id=parent, stamp=stamp), child_frame_id=child
    )
    result.transform.translation = vector(pose.p)
    x, y, z, w = pose.M.GetQuaternion()
    result.transform.rotation = Quaternion(x=x, y=y, z=z, w=w)
    return result


class DriveBridge(Node):
    def __init__(self):
        super().__init__("flashdreams_bridge")
        self.path = self.declare_parameter("socket_path", "/tmp/flashdreams.sock").value
        depth = self.declare_parameter("queue_depth", 32).value
        if depth < 1:
            raise ValueError("queue_depth must be positive")
        self.qos = QoSProfile(
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.clock_pub = self.create_publisher(Clock, "/clock", latched)
        self.odom_pub = self.create_publisher(Odometry, "/sim/ego/odom", self.qos)
        self.state_pub = self.create_publisher(
            VehicleState, "/sim/ego/vehicle_state", self.qos
        )
        self.visualization = EgoVisualization(self, latched)
        self.scene = SceneVisualization(self, latched)
        self.condition_qos = latched
        self.condition_pubs = ()
        self.condition_capable = False
        self.static_tf = StaticTransformBroadcaster(self)
        # A paused session can outlive a viewer; replay the observed poses on reconnect.
        self.dynamic_tf = TransformBroadcaster(self, qos=self.qos)
        self.cam_pubs = {}
        self.cam_ids = {}
        self.calibrations = {}
        self.conn = None
        self.episode = NIL_UUID
        self.observed_episode = None
        self.needs_reset = True
        self.last_clock_us = 0
        self.origin_us = self.offset_us = 0
        self.lock = threading.Lock()
        self.active_request = None
        self.last_error = ""
        self.joy = None
        mode = self.declare_parameter("control_mode", "step").value
        if mode == "joy":
            self.joy = JoyDriver(self)
        elif mode == "step":
            group = ReentrantCallbackGroup()
            self.create_service(Reset, "/sim/reset", self.reset, callback_group=group)
            self.create_service(Step, "/sim/step", self.step, callback_group=group)
        else:
            raise ValueError("control_mode must be 'step' or 'joy'")
        self.diagnostics = DriveDiagnostics(self)
        self.model_logs = ModelLogs(self, self.declare_parameter("model_log_unit", "").value)
        self.get_logger().info(f"Waiting for {mode} control, model socket {self.path}")

    def reset(self, req, response):
        return self.exchange(
            {
                "type": "reset",
                "episode_id": req.episode_id,
                "previous_episode_id": req.previous_episode_id or NIL_UUID,
            },
            response,
        )

    def step(self, req, response):
        control = req.control
        return self.exchange(
            {
                "type": "step",
                "episode_id": req.episode_id,
                "expected_step": req.expected_step,
                "steering_angle_rad": control.steering_angle_rad,
                "throttle": control.throttle,
                "brake": control.brake,
                "reverse": control.reverse,
            },
            response,
        )

    def connect(self):
        if self.conn is not None:
            return
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(1.0)
        self.conn.connect(self.path)
        self.conn.settimeout(0.25)
        hello, _ = read_message(self.conn, allow_images=False, stopping=self.stopping)
        if hello["type"] != "hello" or hello["version"] != 1:
            raise ValueError("Unsupported Interactive Drive endpoint")
        self.episode = hello["episode_id"]
        self.condition_capable = "front_condition" in hello.get("features", ())
        self.needs_reset |= (
            hello["needs_reset"] or self.observed_episode != self.episode
        )
        self.get_logger().info(f"Connected to model, episode={self.episode}")

    def stopping(self):
        return not self.context.ok()

    def close_connection(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def exchange(self, request, response):
        try:
            validate_request(request)
        except ValueError as exc:
            response.code = "INVALID_REQUEST"
            response.message = str(exc)
            response.episode_id = self.episode
            return response
        with self.lock:
            if self.active_request is not None:
                response.code = (
                    "IN_PROGRESS" if self.active_request == request else "BUSY"
                )
                response.episode_id = self.episode
                response.message = "Another simulation request is in progress"
                return response
            self.active_request = request
        count = 0
        started = monotonic()
        try:
            self.connect()
            response.episode_id = self.episode
            if request["type"] == "reset" and self.joy is not None:
                # The joystick owns this session, including a pre-existing endpoint
                # whose current episode is only known after reading its hello.
                request["previous_episode_id"] = self.episode
            if self.needs_reset and (
                request["type"] == "step" or request["episode_id"] == self.episode
            ):
                response.code = "RESET_REQUIRED"
                response.message = (
                    "Reset to a fresh episode UUID to obtain complete observations"
                )
                return response
            wire_request = request
            # Keep the most recent geometry available to views opened while paused.
            if request["type"] == "step" and self.condition_capable:
                wire_request = dict(request, front_condition=True)
            write_message(self.conn, wire_request, stopping=self.stopping)
            expected_step = request.get("expected_step", -1)
            while True:
                msg, images = read_message(self.conn, stopping=self.stopping)
                if msg["type"] == "error":
                    response.code = msg["code"]
                    response.message = msg["message"]
                    self.last_error = f"{response.code}: {response.message}"
                    self.get_logger().warning(self.last_error, throttle_duration_sec=2.0)
                    self.needs_reset |= (
                        bool(msg.get("fatal"))
                        or count > 0
                        or response.code == "RESET_REQUIRED"
                    )
                    return response
                if (msg["episode_id"], msg["step"]) != (
                    request["episode_id"],
                    expected_step,
                ):
                    raise ValueError("Endpoint returned a different episode or step")
                if msg["type"] == "frame":
                    self.publish_frame(msg, images)
                    count += 1
                    continue
                if msg["type"] != "done":
                    raise ValueError("Expected a frame or completion")
                if msg.get("replayed"):
                    if count or self.observed_episode != msg["episode_id"]:
                        raise ValueError(
                            "Cannot replay a result from an unobserved episode"
                        )
                elif count != msg["frame_count"]:
                    raise ValueError("Incomplete observation batch")
                self.fill_response(response, msg)
                self.last_error = ""
                if msg["step"] >= 0 and not msg.get("replayed"):
                    duration = monotonic() - started
                    self.diagnostics.step_done(duration)
                    self.get_logger().debug(
                        f"episode={self.episode} step={msg['step']} frames={count} wall={duration:.3f}s"
                    )
                return response
        except (
            AssertionError,  # Generated ROS messages validate their field types.
            EOFError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
        ) as exc:
            self.close_connection()
            self.needs_reset = True
            response.code = "TRANSPORT_ERROR"
            response.message = str(exc)
            self.last_error = f"{response.code}: {response.message}"
            self.get_logger().warning(self.last_error, throttle_duration_sec=2.0)
            response.episode_id = self.episode
            return response
        finally:
            with self.lock:
                self.active_request = None

    def stamp(self, source_us):
        sec, us = divmod(source_us + self.offset_us, 1_000_000)
        return Time(sec=sec, nanosec=us * 1000)

    def fill_response(self, response, msg):
        response.episode_id = self.episode = msg["episode_id"]
        response.next_step = msg["next_step"]
        response.next_apply_stamp = self.stamp(msg["next_apply_us"])
        if msg["step"] == -1:
            response.cameras = [self.cam_ids[name] for name in msg["cameras"]]
            response.source_origin_us = self.origin_us
            response.initial_observation_stamp = self.stamp(msg["last_observation_us"])
            self.needs_reset = False
        else:
            response.step = msg["step"]
            response.applied_from = self.stamp(msg["applied_from_us"])
            response.last_observation_stamp = self.stamp(msg["last_observation_us"])
            response.frame_count = msg["frame_count"]
            response.applied_control = Control(**msg["control"])
        response.code = "OK"

    def publish_calibration(self, msg, stamp):
        self.cam_ids = {}
        optical_to_flu = kdl.Frame(kdl.Rotation(0, 0, 1, -1, 0, 0, 0, -1, 0))
        static = []
        self.calibrations = {cal["clipgt_name"]: cal for cal in msg["calibration"]}
        for image in msg["images"]:
            cal = self.calibrations[image["name"]]
            name = cal["logical_name"]
            if (
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                or name in self.cam_ids.values()
            ):
                raise ValueError(f"Invalid or duplicate ROS camera name: {name!r}")
            self.cam_ids[image["name"]] = name
            frame_id = name + "_optical_frame"
            static.append(
                transform(
                    "base_link",
                    frame_id,
                    stamp,
                    pose_matrix(cal["sensor_to_rig_flu"]) * optical_to_flu,
                )
            )
            if name not in self.cam_pubs:
                base = "/sim/camera/" + name + "/"
                self.cam_pubs[name] = (
                    self.create_publisher(Image, base + "image_raw", self.qos),
                    self.create_publisher(CameraInfo, base + "camera_info", self.qos),
                    self.create_publisher(CameraModel, base + "calibration", self.qos),
                )
            model = CameraModel(
                header=Header(stamp=stamp, frame_id=frame_id),
                calibration_width=cal["width"],
                calibration_height=cal["height"],
                image_width=image["width"],
                image_height=image["height"],
                cx=float(cal["cx"]),
                cy=float(cal["cy"]),
                polynomial=cal["polynomial"],
                is_backward_polynomial=cal["is_backward_polynomial"],
                linear_cde=cal["linear_cde"],
            )
            self.cam_pubs[name][2].publish(model)
        for retired in self.cam_pubs.keys() - set(self.cam_ids.values()):
            for publisher in self.cam_pubs.pop(retired):
                self.destroy_publisher(publisher)
        for publisher in self.condition_pubs:
            self.destroy_publisher(publisher)
        self.condition_pubs = ()
        if self.condition_capable and self.cam_ids:
            anchor = next(iter(self.cam_ids.values()))
            base = f"/sim/camera/{anchor}/condition/"
            self.condition_pubs = (
                self.create_publisher(Image, base + "image_raw", self.condition_qos),
                self.create_publisher(CameraInfo, base + "camera_info", self.condition_qos),
            )
        if static:
            self.static_tf.sendTransform(static)

    def publish_frame(self, msg, images):
        source = msg["timestamp_us"]
        if msg["step"] == -1:
            self.origin_us = source
            self.offset_us = self.last_clock_us + 1 - source
            self.observed_episode = msg["episode_id"]
            self.publish_calibration(msg, self.stamp(source))
        if source + self.offset_us <= self.last_clock_us:
            raise ValueError("Non-increasing simulation timestamp")
        camera_names = list(self.cam_ids)
        descriptors = msg["images"]
        n = len(camera_names)
        if [img["name"] for img in descriptors[:n]] != camera_names:
            raise ValueError("Frame cameras differ from reset")
        if len(descriptors) > n and not (
            self.condition_pubs and len(descriptors) == n + 1
            and descriptors[n].get("role") == "condition"
            and descriptors[n]["name"] == "condition/" + camera_names[0]
        ):
            raise ValueError("Unexpected condition image from model")
        stamp = self.stamp(source)
        pose = pose_matrix(msg["rig_to_world"])
        tf = transform("odom", "base_link", stamp, pose)
        odom = Odometry(header=tf.header, child_frame_id=tf.child_frame_id)
        (
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z,
        ) = pose.p
        odom.pose.pose.orientation = tf.transform.rotation
        vehicle = msg["vehicle"]
        speed, yaw = vehicle["speed_mps"], vehicle["yaw_rad"]
        vx, vy = vehicle["velocity_x_mps"], vehicle["velocity_y_mps"]
        world_v = kdl.Vector(
            speed * math.cos(yaw) if vx is None else vx,
            speed * math.sin(yaw) if vy is None else vy,
            0,
        )
        odom.twist.twist.linear = vector(pose.M.Inverse() * world_v)
        odom.twist.twist.angular = vector(
            pose.M.Inverse() * kdl.Vector(0, 0, vehicle["yaw_rate_radps"])
        )
        self.odom_pub.publish(odom)
        if "gear" in vehicle:  # Older model endpoints omit actuator feedback.
            self.state_pub.publish(
                VehicleState(
                    header=Header(stamp=stamp, frame_id=tf.child_frame_id),
                    speed_mps=float(speed),
                    steering_angle_rad=float(vehicle["steer_rad"]),
                    yaw_rate_radps=float(vehicle["yaw_rate_radps"]),
                    gear=vehicle["gear"],
                    throttle=float(vehicle["throttle"]),
                    brake=float(vehicle["brake"]),
                )
            )
        self.dynamic_tf.sendTransform(tf)
        if msg["step"] == -1:
            self.scene.reset(msg, tf.header)
            self.get_logger().info(
                f"Initial observation episode={self.observed_episode}, cameras={len(self.cam_ids)}"
            )
        self.scene.publish_traffic(msg, tf.header)
        self.visualization.publish(
            odom, reset=msg["step"] == -1, dimensions_lwh=msg.get("ego_dimensions_lwh"),
            rig_ground_clearance_m=msg.get("rig_ground_clearance_m", 0.0),
        )
        views = []
        for index, (image, data) in enumerate(zip(descriptors, images, strict=True)):
            condition = index == n
            source_name = camera_names[0] if condition else image["name"]
            name = self.cam_ids[source_name]
            header = Header(stamp=stamp, frame_id=name + "_optical_frame")
            width, height = image["width"], image["height"]
            info = camera_info(
                header, width, height, self.calibrations[source_name], opening=msg["step"] == -1
            )
            pixels = Image(
                header=header,
                width=width,
                height=height,
                encoding="rgb8",
                step=width * 3,
                data=array("B", data),
            )
            image_pub, info_pub = self.condition_pubs if condition else self.cam_pubs[name][:2]
            info_pub.publish(info)
            image_pub.publish(pixels)
            if self.joy is not None and not condition:
                views.append((pixels, info))
        if self.joy is not None:
            self.joy.present(views)
        self.clock_pub.publish(Clock(clock=stamp))
        self.last_clock_us = source + self.offset_us
        self.diagnostics.observe(source, self.cam_ids.values(), reset=msg["step"] == -1)


def main():
    rclpy.init()
    node = DriveBridge()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.model_logs.close()
        rclpy.try_shutdown()
        if node.joy is not None:
            node.joy.close()
        executor.shutdown()
        node.close_connection()
        node.destroy_node()


if __name__ == "__main__":
    main()
