# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

import unittest
from itertools import combinations, product
from math import sqrt
from unittest.mock import Mock

import pytest

try:
    from nav_msgs.msg import Odometry
    from visualization_msgs.msg import Marker
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Source ROS 2 Humble first") from exc

from ego_visualization import EgoVisualization, wireframe_points

pytestmark = pytest.mark.ci_cpu


class TestEgoVisualization(unittest.TestCase):
    def test_cuboid_edges_and_center_to_front(self):
        corners = product((-3, 3), (-2, 2), (-1, 1))
        edges = {
            frozenset((a, b)) for a, b in combinations(corners, 2)
            if sum(i != j for i, j in zip(a, b)) == 1
        }
        outline = wireframe_points(6, 4, 2, heading=False)
        coords = [(p.x, p.y, p.z) for p in outline]
        self.assertEqual(len(coords), 24)
        self.assertEqual({frozenset(pair) for pair in zip(coords[::2], coords[1::2])}, edges)
        oriented = wireframe_points(6, 4, 2)
        self.assertEqual(oriented[:-2], outline)
        self.assertEqual([(p.x, p.y, p.z) for p in oriented[-2:]], [(0, 0, 0), (3, 0, 0)])

    def test_ego_uses_ground_pose_and_body_heading_in_reverse(self):
        view = EgoVisualization(Mock(), qos=1)
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.header.stamp.sec = 5
        odom.pose.pose.position.x = 10.0
        odom.pose.pose.orientation.y = sqrt(0.5)
        odom.pose.pose.orientation.w = sqrt(0.5)
        odom.twist.twist.linear.x = -2.0
        view.publish(odom, reset=True, dimensions_lwh=[5.0, 2.0, 1.8])
        body, *removed = view.publisher.publish.call_args.args[0].markers
        self.assertEqual({(m.ns, m.action) for m in removed}, {
            ("heading", Marker.DELETE), ("trail", Marker.DELETE),
        })
        self.assertEqual((body.type, body.action), (Marker.LINE_LIST, Marker.ADD))
        self.assertEqual(body.header, odom.header)
        self.assertEqual(body.pose.orientation, odom.pose.pose.orientation)
        self.assertAlmostEqual(body.pose.position.x, 10.9)
        self.assertAlmostEqual(body.pose.position.z, 0.0)
        self.assertEqual(body.points, wireframe_points(5, 2, 1.8))
        view.publish(odom, reset=False)
        self.assertEqual(view.publisher.publish.call_args.args[0].markers, [body])
        view.publish(odom, reset=True)
        legacy_body = view.publisher.publish.call_args.args[0].markers[0]
        self.assertEqual(legacy_body.points, wireframe_points(4.8, 2.0, 1.6))
        with self.assertRaises(ValueError):
            view.publish(odom, reset=True, dimensions_lwh=[5.0, -2.0, 1.8])

    def test_registered_rig_height_is_cached_per_episode(self):
        view = EgoVisualization(Mock(), qos=1)
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        odom.pose.pose.position.z = 1.21
        view.publish(odom, reset=True, rig_ground_clearance_m=0.84)
        view.publish(odom, reset=False)
        body = view.publisher.publish.call_args.args[0].markers[0]
        self.assertAlmostEqual(body.pose.position.z + min(p.z for p in body.points), 0.37)
        view.publish(odom, reset=True)
        body = view.publisher.publish.call_args.args[0].markers[0]
        self.assertAlmostEqual(body.pose.position.z + min(p.z for p in body.points), 1.21)
