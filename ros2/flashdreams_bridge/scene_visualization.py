# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
"""Publish prepared scene geometry in the same odom coordinates as the renderer."""

from bisect import bisect_left
import hashlib
import json
from pathlib import Path

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
import PyKDL as kdl
from rclpy.duration import Duration
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from .ego_visualization import wireframe_points


_GEOMETRY_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AssertionError,
    OverflowError,
)


def _asset(root, name, digest):
    path = root / name
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f"Scene geometry checksum mismatch for {path}")
    return json.loads(payload)


def _point(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def _load_scene(descriptor, header):
    root = Path(descriptor["root"])
    if not root.is_absolute():
        raise ValueError("Expected an absolute prepared scene path")
    manifest = _asset(root, "scene.json", descriptor["manifest_sha256"])
    digests = manifest["files_sha256"]
    maps = _asset(root, "map.json", digests["map.json"])
    traffic = _asset(root, "tracks.json", digests["tracks.json"])
    layers = {}
    for group in maps["lines"]:
        prim = group["prim"]
        if type(prim) is not int or prim < 0:
            raise ValueError(f"Invalid map primitive ID {prim!r}")
        if prim not in layers:
            r, g, b, a = manifest["palette"][str(prim)]
            layers[prim] = Marker(
                header=header,
                ns="map",
                type=Marker.LINE_LIST,
                pose=Pose(orientation=Quaternion(w=1.0)),
                scale=Vector3(x=0.035),
                color=ColorRGBA(r=r / 255, g=g / 255, b=b / 255, a=a / 255),
            )
        for start, end in group["vertices"]:
            layers[prim].points.extend((_point(*start), _point(*end)))

    tracks = []
    for group in traffic:
        if group["prim"] != 5:  # Ludus PRIM_OBSTACLE (other pools include road signs).
            continue
        for track in group["tracks"]:
            track["timestamps_us"] = tuple(map(int, track["timestamps_us"]))
            tracks.append(track)
    return layers, tracks


def _actor_marker(track, index, time_us, header):
    stamps = track["timestamps_us"]
    # Prepared scenes use zero extrapolation. A one-sample track is visible at its stamp.
    if time_us < stamps[0] or time_us > stamps[-1]:
        return None
    right = bisect_left(stamps, time_us)
    position = track["translations"][right]
    orientation = track["quaternions_xyzw"][right]
    if stamps[right] != time_us:
        left = right - 1
        fraction = (time_us - stamps[left]) / (stamps[right] - stamps[left])
        position = [
            a + (b - a) * fraction
            for a, b in zip(track["translations"][left], position)
        ]
        first = kdl.Rotation.Quaternion(*track["quaternions_xyzw"][left])
        last = kdl.Rotation.Quaternion(*orientation)
        orientation = kdl.addDelta(
            first, kdl.diff(first, last), fraction
        ).GetQuaternion()
    x, y, z, w = map(float, orientation)
    kind = track.get("object_type", "obstacle")
    if not isinstance(kind, str):
        raise ValueError("Expected a string actor category")
    source = track.get("source_object_id", index)
    return Marker(
        header=header,
        ns=f"{kind}/{source}",
        id=index,
        type=Marker.LINE_LIST,
        pose=Pose(
            position=_point(*position), orientation=Quaternion(x=x, y=y, z=z, w=w)
        ),
        points=wireframe_points(
            *track["scale"], heading=kind.lower() in {"car", "truck", "bus", "van"}
        ),
        scale=Vector3(x=0.045),
        color=ColorRGBA(r=0.81, g=0.86, b=0.87, a=1.0),
    )


class SceneVisualization:
    def __init__(self, node, qos):
        self.node, self.qos = node, qos
        self.map_pubs = {}
        self.traffic_pub = node.create_publisher(
            MarkerArray, "/sim/traffic/markers", qos
        )
        self.tracks = None

    def reset(self, msg, header):
        layers, self.tracks = {}, None
        descriptor = msg.get("scene_geometry")
        if descriptor is not None:
            try:
                layers, self.tracks = _load_scene(descriptor, header)
            except _GEOMETRY_ERRORS as exc:
                self.node.get_logger().error(f"Cannot load 3D scene geometry: {exc}")
            else:
                segments = {
                    prim: len(marker.points) // 2 for prim, marker in layers.items()
                }
                self.node.get_logger().info(
                    f"3D scene loaded: map segments {segments}, {len(self.tracks)} actor tracks"
                )
        empty = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
        if self.tracks is None:
            self.traffic_pub.publish(empty)
        for prim in self.map_pubs.keys() - layers.keys():
            pub = self.map_pubs[prim]
            pub.publish(empty)
            # Keep the latched deletion for slow readers if it has not been acknowledged.
            if pub.wait_for_all_acked(Duration(seconds=0.25)):
                self.node.destroy_publisher(pub)
                del self.map_pubs[prim]
        for prim, marker in layers.items():
            if prim not in self.map_pubs:
                topic = f"/sim/map/prim_{prim}/markers"
                self.map_pubs[prim] = self.node.create_publisher(
                    MarkerArray, topic, self.qos
                )
            self.map_pubs[prim].publish(MarkerArray(markers=[marker]))

    def clear(self, header):
        self.reset({}, header)

    def publish_traffic(self, msg, header):
        if self.tracks is None:
            return
        markers = [Marker(header=header, action=Marker.DELETEALL)]
        try:
            time_us = msg["scene_time_us"]
            # ponytail: scan scene tracks per generated frame; index intervals for larger clips.
            for index, track in enumerate(self.tracks):
                marker = _actor_marker(track, index, time_us, header)
                if marker is not None:
                    markers.append(marker)
        except _GEOMETRY_ERRORS as exc:
            self.node.get_logger().error(f"Cannot sample 3D traffic: {exc}")
            self.tracks = None
            del markers[1:]
        self.traffic_pub.publish(MarkerArray(markers=markers))
