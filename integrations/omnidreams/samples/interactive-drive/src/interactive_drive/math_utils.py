# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import math
from collections.abc import Iterable


def _np():
    import numpy as np

    return np


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(v: Iterable[float]) -> object:
    np = _np()
    arr = np.asarray(tuple(v), dtype=np.float32)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-8:
        return arr
    return arr / nrm


def quaternion_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> object:
    np = _np()
    half = np.radians(np.array([roll_deg, pitch_deg, yaw_deg], dtype=np.float32)) * 0.5
    cr, cp, cy = np.cos(half)
    sr, sp, sy = np.sin(half)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def quaternion_to_matrix_xyzw(quat_xyzw: Iterable[float]) -> object:
    np = _np()
    x, y, z, w = normalize(quat_xyzw)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def matrix_from_quaternion_translation(
    quat_xyzw: Iterable[float], translation_xyz: Iterable[float]
) -> object:
    np = _np()
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = quaternion_to_matrix_xyzw(quat_xyzw)
    matrix[:3, 3] = np.asarray(tuple(translation_xyz), dtype=np.float32)
    return matrix


def matrix_from_rpy_translation(
    roll_deg: float, pitch_deg: float, yaw_deg: float, translation_xyz: Iterable[float]
) -> object:
    quat = quaternion_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
    return matrix_from_quaternion_translation(quat, translation_xyz)


def matrix_from_xy_yaw(x_m: float, y_m: float, z_m: float, yaw_rad: float) -> object:
    np = _np()
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = cy
    matrix[0, 1] = -sy
    matrix[1, 0] = sy
    matrix[1, 1] = cy
    matrix[0, 3] = x_m
    matrix[1, 3] = y_m
    matrix[2, 3] = z_m
    return matrix


def compose(a: object, b: object) -> object:
    np = _np()
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)


def invert_rigid(transform: object) -> object:
    np = _np()
    t = np.asarray(transform, dtype=np.float32)
    rot = t[:3, :3]
    trans = t[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = rot.T
    out[:3, 3] = -(rot.T @ trans)
    return out


def polyline_to_segments(points: object) -> object:
    np = _np()
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def sample_polyline(points: object, spacing_m: float) -> object:
    np = _np()
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return pts.copy()
    diffs = pts[1:] - pts[:-1]
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = float(seg_lengths.sum())
    if total_length < 1e-6:
        return pts[:1].copy()
    distances = [0.0]
    for seg_length in seg_lengths:
        distances.append(distances[-1] + float(seg_length))
    target_distances = np.arange(0.0, total_length + spacing_m * 0.5, spacing_m, dtype=np.float32)
    sampled = []
    seg_index = 0
    for target in target_distances:
        while seg_index + 1 < len(distances) and distances[seg_index + 1] < float(target):
            seg_index += 1
        if seg_index >= len(seg_lengths):
            sampled.append(pts[-1])
            continue
        start_d = distances[seg_index]
        end_d = distances[seg_index + 1]
        alpha = 0.0 if end_d <= start_d else (float(target) - start_d) / (end_d - start_d)
        sampled.append(pts[seg_index] * (1.0 - alpha) + pts[seg_index + 1] * alpha)
    if not np.allclose(sampled[-1], pts[-1]):
        sampled.append(pts[-1])
    return np.asarray(sampled, dtype=np.float32)


def dash_polyline(
    points: object, pattern: list[tuple[bool, float]], spacing_m: float = 0.25
) -> list[object]:
    np = _np()
    sampled = sample_polyline(points, spacing_m=spacing_m)
    if len(sampled) < 2:
        return []
    segments = polyline_to_segments(sampled)
    segment_lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    pattern_length = sum(length for _, length in pattern)
    if pattern_length <= 1e-6:
        return [segments]
    visible = []
    distance = 0.0
    for segment, seg_length in zip(segments, segment_lengths, strict=False):
        phase = distance % pattern_length
        accum = 0.0
        draw = False
        for is_visible, length in pattern:
            if accum <= phase < accum + length:
                draw = is_visible
                break
            accum += length
        if draw:
            visible.append(segment)
        distance += float(seg_length)
    if not visible:
        return []
    return [np.asarray(visible, dtype=np.float32)]


def offset_segments(segments: object, offset_m: float) -> object:
    np = _np()
    segs = np.asarray(segments, dtype=np.float32)
    if len(segs) == 0:
        return segs.copy()
    offsets = []
    for p0, p1 in segs:
        direction = p1 - p0
        direction_xy = direction[:2]
        norm = float(np.linalg.norm(direction_xy))
        if norm < 1e-6:
            offsets.append(np.zeros(3, dtype=np.float32))
            continue
        tangent = direction_xy / norm
        normal_xy = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        offsets.append(
            np.array([normal_xy[0] * offset_m, normal_xy[1] * offset_m, 0.0], dtype=np.float32)
        )
    offset_arr = np.asarray(offsets, dtype=np.float32)
    return segs + offset_arr[:, None, :]


def triangle_fan(vertices: object) -> object:
    np = _np()
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) < 3:
        return np.empty((0, 3, 3), dtype=np.float32)
    tris = []
    anchor = verts[0]
    for idx in range(1, len(verts) - 1):
        tris.append([anchor, verts[idx], verts[idx + 1]])
    return np.asarray(tris, dtype=np.float32)


def oriented_box_triangles(
    center: Iterable[float], dimensions: Iterable[float], quat_xyzw: Iterable[float]
) -> object:
    np = _np()
    cx, cy, cz = center
    dx, dy, dz = dimensions
    half = np.array([dx, dy, dz], dtype=np.float32) * 0.5
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float32,
    )
    rot = quaternion_to_matrix_xyzw(quat_xyzw)
    world = (rot @ local.T).T + np.array([cx, cy, cz], dtype=np.float32)
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    tris = []
    for i0, i1, i2, i3 in faces:
        tris.append([world[i0], world[i1], world[i2]])
        tris.append([world[i0], world[i2], world[i3]])
    return np.asarray(tris, dtype=np.float32)
