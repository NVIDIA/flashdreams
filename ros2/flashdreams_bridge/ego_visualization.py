# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
"""Draw the observed ego vehicle and build wireframes for other scene objects."""

from math import isfinite

from geometry_msgs.msg import Point, Pose, Vector3
import PyKDL as kdl
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_DEFAULT_EGO_DIMENSIONS = (4.8, 2.0, 1.6)


def wireframe_points(
    length: float, width: float, height: float, *, heading: bool = True
) -> list[Point]:
    """Return independent line segments in a box-centered, +X-forward frame."""
    if not all(isfinite(d) and d > 0 for d in (length, width, height)):
        raise ValueError("Box dimensions must be finite positive lengths")
    x, y, z = length / 2, width / 2, height / 2
    corners = [
        Point(x=px, y=py, z=pz)
        for pz in (-z, z)
        for px, py in ((-x, -y), (x, -y), (x, y), (-x, y))
    ]
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    points = [corners[index] for edge in edges for index in edge]
    if heading:
        points.extend((Point(), Point(x=x)))
    return points


class EgoVisualization:
    def __init__(self, node, qos):
        self.publisher = node.create_publisher(MarkerArray, "/sim/ego/markers", qos)
        self.height = _DEFAULT_EGO_DIMENSIONS[2]
        self.rig_ground_clearance_m = 0.0
        self.points = wireframe_points(*_DEFAULT_EGO_DIMENSIONS)

    def publish(self, odom, *, reset, dimensions_lwh=None, rig_ground_clearance_m=0.0):
        if reset:
            dimensions = _DEFAULT_EGO_DIMENSIONS if dimensions_lwh is None else dimensions_lwh
            if not isfinite(rig_ground_clearance_m):
                raise ValueError("Rig ground clearance must be finite")
            self.points = wireframe_points(*dimensions)
            self.height = dimensions[2]
            self.rig_ground_clearance_m = rig_ground_clearance_m
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        up = kdl.Rotation.Quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        ).UnitZ()
        # The registered road lies a world-vertical distance below the rig.
        center = Point(
            x=position.x + self.height / 2 * up.x(),
            y=position.y + self.height / 2 * up.y(),
            z=position.z - self.rig_ground_clearance_m + self.height / 2 * up.z(),
        )
        body = Marker(
            header=odom.header, ns="body", type=Marker.LINE_LIST,
            pose=Pose(position=center, orientation=orientation),
            points=self.points,
            scale=Vector3(x=0.07), color=ColorRGBA(g=0.90, b=1.0, a=1.0),
        )
        markers = [body]
        if reset:
            # Clear persistent markers in layouts still connected across an upgrade.
            markers.extend(
                Marker(header=odom.header, ns=ns, action=Marker.DELETE)
                for ns in ("heading", "trail")
            )
        self.publisher.publish(MarkerArray(markers=markers))
