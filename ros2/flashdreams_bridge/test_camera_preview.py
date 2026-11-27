# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

from array import array
from io import BytesIO
import unittest
from unittest.mock import Mock

import pytest

try:
    from PIL import Image as PillowImage
    from sensor_msgs.msg import Image
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Run with ROS 2 Humble and Pillow") from exc

from ros2.flashdreams_bridge.camera_preview import CameraPreview, jpeg_image

pytestmark = pytest.mark.ci_cpu


def test_preview_preserves_capture_geometry_and_rgb_channels():
    width, height, padding = 32, 16, 4
    row = bytes((231, 90, 15)) * width + bytes(padding)
    image = Image(
        width=width, height=height, step=len(row),
        encoding="rgb8", data=array("B", row * height),
    )
    image.header.stamp.sec = 312
    image.header.frame_id = "rear_optical"
    compressed = jpeg_image(image, 75)
    assert compressed.header == image.header
    with PillowImage.open(BytesIO(compressed.data)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.size == (width, height)
        assert decoded.getpixel((2, 14)) == pytest.approx((231, 90, 15), abs=3)
    image.encoding = "bgr8"
    with pytest.raises(ValueError, match="RGB8"):
        jpeg_image(image, 75)


def test_only_viewed_sources_are_subscribed_and_retired_sources_disappear():
    name = "/sim/camera/front/condition/image_raw"
    node = Mock(outputs={}, inputs={})
    node.get_topic_names_and_types.return_value = [(name, ["sensor_msgs/msg/Image"])]
    node.count_publishers.return_value = 1
    pub = node.create_publisher.return_value
    pub.get_subscription_count.return_value = 0
    CameraPreview.refresh(node)
    assert node.outputs == {name: pub} and not node.inputs
    assert node.create_publisher.call_args.args[1] == name + "/compressed"
    pub.get_subscription_count.return_value = 1
    CameraPreview.refresh(node)
    sub = node.create_subscription.return_value
    assert node.inputs == {name: sub}
    pub.get_subscription_count.return_value = 0
    CameraPreview.refresh(node)
    node.destroy_subscription.assert_called_once_with(sub)
    node.count_publishers.return_value = 0
    CameraPreview.refresh(node)
    node.destroy_publisher.assert_called_once_with(pub)
    assert not node.outputs and not node.inputs
