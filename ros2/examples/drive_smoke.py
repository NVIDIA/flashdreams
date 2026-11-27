# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

"""Drive a stationary RGB scene through three ROS steps and check the observations."""

import argparse
import time
import uuid

import rclpy
from flashdreams_interfaces.srv import Reset, Step
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--previous-episode", default="", help="ID of the last active episode"
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="Wall seconds per operation"
    )
    parser.add_argument(
        "--snapshot",
        help="Save the last first-camera observation as PNG (requires Pillow)",
    )
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node("flashdreams_drive_smoke")
    qos = QoSProfile(depth=32, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    odom, clocks = {}, set()
    images, infos = {}, {}
    last_image = None

    def wait(predicate, description):
        deadline = time.monotonic() + args.timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise TimeoutError(description)
            rclpy.spin_once(node, timeout_sec=0.1)

    def call(client, request):
        future = client.call_async(request)
        wait(future.done, client.srv_name)
        result = future.result()
        if result.code != "OK":
            raise RuntimeError(
                f"{client.srv_name} returned {result.code}: {result.message}"
            )
        return result

    def read_image(msg, camera):
        nonlocal last_image
        assert msg.encoding == "rgb8" and msg.width and msg.height
        assert len(msg.data) == msg.step * msg.height
        images[camera][ns(msg.header.stamp)] = (
            msg.width,
            msg.height,
            msg.header.frame_id,
        )
        if args.snapshot and camera == next(iter(images)):
            last_image = msg

    def read_info(msg, samples):
        samples[ns(msg.header.stamp)] = (msg.width, msg.height, msg.header.frame_id)

    def observed(t):
        return (
            t in odom
            and t in clocks
            and all(t in samples for samples in images.values())
            and all(t in samples for samples in infos.values())
        )

    try:
        reset = node.create_client(Reset, "/sim/reset")
        step = node.create_client(Step, "/sim/step")
        node.create_subscription(
            Odometry,
            "/sim/ego/odom",
            lambda m: odom.update({ns(m.header.stamp): m}),
            qos,
        )
        node.create_subscription(
            Clock, "/clock", lambda m: clocks.add(ns(m.clock)), qos
        )
        for client in (reset, step):
            if not client.wait_for_service(timeout_sec=args.timeout):
                raise TimeoutError(client.srv_name)
        episode = str(uuid.uuid4())
        initial = call(
            reset,
            Reset.Request(
                episode_id=episode, previous_episode_id=args.previous_episode
            ),
        )
        print(f"episode={episode} cameras={list(initial.cameras)}", flush=True)
        assert (
            initial.episode_id == episode and initial.next_step == 0 and initial.cameras
        )
        for camera in initial.cameras:
            images[camera], infos[camera] = {}, {}
            prefix = f"/sim/camera/{camera}"
            node.create_subscription(
                Image, prefix + "/image_raw", lambda m, c=camera: read_image(m, c), qos
            )
            node.create_subscription(
                CameraInfo,
                prefix + "/camera_info",
                lambda m, c=camera: read_info(m, infos[c]),
                qos,
            )
        previous = ns(initial.initial_observation_stamp)
        apply_from = ns(initial.next_apply_stamp)
        wait(lambda: observed(previous), "reset observations")
        speeds = []
        for i, (angle, throttle, brake) in enumerate(
            ((0.0, 1.0, 0.0), (0.15, 1.0, 0.0), (0.0, 0.0, 1.0))
        ):
            request = Step.Request(episode_id=episode, expected_step=i)
            request.control.steering_angle_rad = angle
            request.control.throttle = throttle
            request.control.brake = brake
            started = time.monotonic()
            reply = call(step, request)
            last = ns(reply.last_observation_stamp)
            assert reply.step == i and reply.next_step == i + 1
            assert (
                ns(reply.applied_from) == apply_from
                and reply.applied_control == request.control
            )
            wait(lambda: observed(last), f"observations for step {i}")
            for camera in initial.cameras:
                stamps = {t for t in images[camera] if previous < t <= last}
                assert len(stamps) == reply.frame_count
                assert all(
                    t in odom and images[camera][t] == infos[camera][t] for t in stamps
                )
            speed = odom[last].twist.twist.linear.x
            speeds.append(speed)
            print(
                f"step={i} frames={reply.frame_count} sim_dt={(ns(reply.next_apply_stamp) - apply_from) / 1e9:.3f}s "
                f"wall={time.monotonic() - started:.3f}s speed={speed:.3f}m/s "
                f"yaw_rate={odom[last].twist.twist.angular.z:.4f}rad/s",
                flush=True,
            )
            if i == 0:
                assert call(step, request) == reply, (
                    "A step retry must return its original receipt"
                )
            previous, apply_from = last, ns(reply.next_apply_stamp)
        assert speeds[1] > speeds[0] > 0 and speeds[2] < speeds[1]
        if args.snapshot:
            from PIL import Image as PILImage

            assert last_image is not None and ns(last_image.header.stamp) == previous
            PILImage.frombytes(
                "RGB", (last_image.width, last_image.height), last_image.data.tobytes()
            ).save(args.snapshot, format="PNG")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
