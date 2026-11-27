# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
"""Check layered map snapshots and traffic visibility without a model or a ROS daemon."""

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import pytest

try:
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Source ROS 2 Humble first") from exc

from ros2.flashdreams_bridge.scene_visualization import SceneVisualization

pytestmark = pytest.mark.ci_cpu


class TestSceneVisualization(unittest.TestCase):
    def test_snapshots_use_source_layers_and_render_time(self):
        node = Mock()
        pubs = {}

        def create(_type, topic, _qos):
            pubs[topic] = Mock()
            return pubs[topic]

        node.create_publisher.side_effect = create
        scene = SceneVisualization(node, 1)
        header = Header(frame_id="odom")
        header.stamp.sec = 12
        with TemporaryDirectory() as temp:
            root = Path(temp)

            def asset(name, data):
                payload = json.dumps(data).encode()
                (root / name).write_bytes(payload)
                return hashlib.sha256(payload).hexdigest()

            car = {
                "object_type": "Car",
                "source_object_id": "A",
                "timestamps_us": ["10", "30"],
                "translations": [[1, 2, 3], [5, 6, 7]],
                "quaternions_xyzw": [[0, 0, 0, 1], [0, 0, 1, 0]],
                "scale": [4, 2, 1],
            }
            person = {
                **car,
                "object_type": "Pedestrian",
                "source_object_id": "B",
                "timestamps_us": ["20"],
                "translations": [[0, 1, 2]],
                "quaternions_xyzw": [[0, 0, 0, 1]],
            }
            digests = {
                "tracks.json": asset(
                    "tracks.json", [{"prim": 5, "tracks": [car, person]}]
                ),
            }
            segment = [[-1, -2, 0], [1, 2, 0]]

            def reset(layers):
                digests["map.json"] = asset(
                    "map.json",
                    {"lines": [{"prim": p, "vertices": [segment]} for p in layers]},
                )
                manifest_hash = asset(
                    "scene.json",
                    {
                        "palette": {
                            "28": [207, 255, 191, 255],
                            "33": [255, 255, 255, 255],
                        },
                        "files_sha256": digests,
                    },
                )
                msg = {
                    "scene_geometry": {
                        "root": str(root),
                        "manifest_sha256": manifest_hash,
                    }
                }
                scene.reset(msg, header)
                return msg

            reset([28, 33, 28])
            green = pubs["/sim/map/prim_28/markers"]
            white = pubs["/sim/map/prim_33/markers"]
            (marker,) = green.publish.call_args.args[0].markers
            self.assertEqual((marker.header, marker.type), (header, Marker.LINE_LIST))
            self.assertEqual([p.x for p in marker.points], [-1, 1, -1, 1])
            self.assertAlmostEqual(marker.color.r, 207 / 255)
            self.assertAlmostEqual(marker.color.b, 191 / 255)

            def observe(time_us):
                scene.publish_traffic({"scene_time_us": time_us}, header)
                clear, *boxes = (
                    pubs["/sim/traffic/markers"].publish.call_args.args[0].markers
                )
                self.assertEqual(clear.action, Marker.DELETEALL)
                self.assertTrue(
                    all(
                        m.header == header and m.type == Marker.LINE_LIST for m in boxes
                    )
                )
                return boxes

            vehicle, walker = observe(20)
            self.assertEqual(len(vehicle.points), 26)
            self.assertEqual(len(walker.points), 24)
            self.assertEqual(vehicle.ns, "Car/A")
            self.assertEqual(vehicle.pose.position.x, 3)
            self.assertAlmostEqual(abs(vehicle.pose.orientation.z), math.sqrt(0.5))
            self.assertEqual(observe(30)[0].pose.position.x, 5)
            self.assertEqual(observe(31), [])
            green.publish.assert_called_once()

            msg = reset([28])
            self.assertEqual(
                white.publish.call_args.args[0].markers[0].action, Marker.DELETEALL
            )
            node.destroy_publisher.assert_called_once_with(white)
            self.assertEqual(len(green.publish.call_args.args[0].markers[0].points), 2)
            (root / "tracks.json").write_text("[]")
            scene.reset(msg, header)
            node.get_logger().error.assert_called_once()
            self.assertEqual(
                green.publish.call_args.args[0].markers[0].action, Marker.DELETEALL
            )
            self.assertEqual(
                pubs["/sim/traffic/markers"]
                .publish.call_args.args[0]
                .markers[0]
                .action,
                Marker.DELETEALL,
            )
