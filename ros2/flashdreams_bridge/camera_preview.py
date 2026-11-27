#!/usr/bin/python3
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Encode only the Interactive Drive cameras being viewed remotely."""

from array import array
from io import BytesIO

from PIL import Image as PillowImage
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


def jpeg_image(image, quality):
    if image.encoding != "rgb8" or image.step < image.width * 3:
        raise ValueError(
            f"Expected RGB8 camera pixels, got {image.encoding}, step={image.step}"
        )
    if len(image.data) != image.height * image.step:
        raise ValueError("Camera buffer size differs from its dimensions")
    pixels = PillowImage.frombytes(
        "RGB", (image.width, image.height), image.data.tobytes(), "raw", "RGB", image.step, 1
    )
    encoded = BytesIO()
    pixels.save(encoded, "JPEG", quality=quality)
    return CompressedImage(
        header=image.header,
        format="rgb8; jpeg compressed bgr8",
        data=array("B", encoded.getbuffer()),
    )


class CameraPreview(Node):
    def __init__(self):
        super().__init__("flashdreams_camera_preview")
        self.quality = self.declare_parameter("jpeg_quality", 75).value
        if type(self.quality) is not int or not 0 <= self.quality <= 100:
            raise ValueError("jpeg_quality must be an integer between 0 and 100")
        self.qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.outputs = {}
        self.inputs = {}
        self.create_timer(0.25, self.refresh, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(f"Camera JPEG previews ready (quality {self.quality})")

    def refresh(self):
        cameras = {
            name
            for name, types in self.get_topic_names_and_types()
            if name.startswith("/sim/camera/")
            and name.endswith("/image_raw")
            and "sensor_msgs/msg/Image" in types
            and self.count_publishers(name)
        }
        for name in self.outputs.keys() - cameras:
            if name in self.inputs:
                self.destroy_subscription(self.inputs.pop(name))
            self.destroy_publisher(self.outputs.pop(name))
        for name in cameras:
            if name not in self.outputs:
                self.outputs[name] = self.create_publisher(
                    CompressedImage, name + "/compressed", self.qos
                )
            pub = self.outputs[name]
            watching = pub.get_subscription_count() > 0
            if watching and name not in self.inputs:
                # Transient-local also delivers the current raw frame while driving is paused.
                self.inputs[name] = self.create_subscription(
                    Image, name, lambda image, pub=pub: self.encode(pub, image), self.qos
                )
                self.get_logger().info(f"Streaming {pub.topic_name}")
            elif not watching and name in self.inputs:
                self.destroy_subscription(self.inputs.pop(name))
                self.get_logger().info(f"Paused {pub.topic_name}")

    def encode(self, pub, image):
        try:
            preview = jpeg_image(image, self.quality)
        except (ValueError, OSError) as error:
            self.get_logger().warning(
                f"Cannot encode {pub.topic_name}: {error}", throttle_duration_sec=5.0
            )
            return
        pub.publish(preview)


def main():
    rclpy.init()
    node = CameraPreview()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
