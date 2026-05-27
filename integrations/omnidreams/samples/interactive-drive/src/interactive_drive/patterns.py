# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


def subdivide_polyline(
    points_xyz: npt.NDArray[np.float32], interval_m: float
) -> npt.NDArray[np.float32]:
    if len(points_xyz) <= 1:
        return points_xyz.astype(np.float32)

    samples: list[npt.NDArray[np.float32]] = [points_xyz[0].astype(np.float32)]
    for p0, p1 in zip(points_xyz[:-1], points_xyz[1:], strict=False):
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        steps = max(1, int(np.ceil(length / interval_m)))
        for i in range(1, steps + 1):
            t = i / steps
            samples.append((p0 + direction * t).astype(np.float32))
    return np.stack(samples, axis=0).astype(np.float32)


def segments_from_polyline(points_xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if len(points_xyz) < 2:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack([points_xyz[:-1], points_xyz[1:]], axis=1).astype(np.float32)


def resample_polyline(
    points_xyz: npt.NDArray[np.float32], interval_m: float
) -> npt.NDArray[np.float32]:
    if len(points_xyz) <= 1:
        return points_xyz.astype(np.float32)
    if interval_m <= 1e-6:
        return points_xyz.astype(np.float32)

    deltas = points_xyz[1:] - points_xyz[:-1]
    lengths = np.linalg.norm(deltas, axis=1).astype(np.float32)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(lengths, dtype=np.float32)]
    )
    total_length = float(cumulative[-1])
    if total_length <= interval_m:
        return np.stack([points_xyz[0], points_xyz[-1]], axis=0).astype(np.float32)

    sample_distances = np.arange(0.0, total_length, interval_m, dtype=np.float32)
    if total_length - float(sample_distances[-1]) > 1e-4:
        sample_distances = np.concatenate(
            [sample_distances, np.array([total_length], dtype=np.float32)]
        )

    sampled: list[npt.NDArray[np.float32]] = []
    segment_index = 0
    for distance in sample_distances:
        while segment_index < len(lengths) - 1 and distance > cumulative[segment_index + 1]:
            segment_index += 1

        segment_length = float(lengths[segment_index])
        if segment_length <= 1e-6:
            sampled.append(points_xyz[segment_index].astype(np.float32))
            continue

        local_t = float((distance - cumulative[segment_index]) / segment_length)
        local_t = float(np.clip(local_t, 0.0, 1.0))
        point = (
            points_xyz[segment_index] * (1.0 - local_t) + points_xyz[segment_index + 1] * local_t
        )
        sampled.append(point.astype(np.float32))

    if np.linalg.norm(sampled[-1] - points_xyz[-1]) > 1e-4:
        sampled.append(points_xyz[-1].astype(np.float32))
    return np.stack(sampled, axis=0).astype(np.float32)


def _segment_lengths(
    line_segments: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], float]:
    lengths = np.linalg.norm(line_segments[:, 1] - line_segments[:, 0], axis=1).astype(np.float32)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(lengths, dtype=np.float32)]
    )
    return lengths, cumulative, float(cumulative[-1])


def _clip_pattern(
    line_segments: npt.NDArray[np.float32], on_length_m: float, off_length_m: float
) -> npt.NDArray[np.float32]:
    if len(line_segments) == 0:
        return line_segments

    lengths, cumulative, total_length = _segment_lengths(line_segments)
    result: list[npt.NDArray[np.float32]] = []
    start = 0.0
    while start < total_length:
        end = min(start + on_length_m, total_length)
        for index, segment in enumerate(line_segments):
            seg_start = float(cumulative[index])
            seg_end = float(cumulative[index + 1])
            if seg_end <= start or seg_start >= end or lengths[index] <= 1e-6:
                continue

            clip_start = max(start, seg_start)
            clip_end = min(end, seg_end)
            t0 = (clip_start - seg_start) / float(lengths[index])
            t1 = (clip_end - seg_start) / float(lengths[index])
            p0 = segment[0] + (segment[1] - segment[0]) * t0
            p1 = segment[0] + (segment[1] - segment[0]) * t1
            result.append(np.stack([p0, p1], axis=0).astype(np.float32))
        start += on_length_m + off_length_m

    if not result:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.stack(result, axis=0).astype(np.float32)


def offset_segments(
    line_segments: npt.NDArray[np.float32],
    offset_distance_m: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if len(line_segments) == 0:
        empty = np.empty((0, 2, 3), dtype=np.float32)
        return empty, empty

    left: list[npt.NDArray[np.float32]] = []
    right: list[npt.NDArray[np.float32]] = []
    for segment in line_segments:
        direction = segment[1] - segment[0]
        direction_xy = direction[:2]
        norm = float(np.linalg.norm(direction_xy))
        if norm < 1e-6:
            continue
        tangent_xy = direction_xy / norm
        normal = np.array([-tangent_xy[1], tangent_xy[0], 0.0], dtype=np.float32)
        offset = normal * offset_distance_m
        left.append((segment + offset).astype(np.float32))
        right.append((segment - offset).astype(np.float32))

    if not left:
        empty = np.empty((0, 2, 3), dtype=np.float32)
        return empty, empty
    return np.stack(left, axis=0).astype(np.float32), np.stack(right, axis=0).astype(np.float32)


def apply_pattern(
    line_segments: npt.NDArray[np.float32],
    pattern: str,
    dual_pattern: tuple[str, str] | None = None,
    dual_offset_m: float = 0.10,
) -> list[npt.NDArray[np.float32]]:
    if pattern == "solid":
        return [line_segments]
    if pattern == "long_dashed":
        return [_clip_pattern(line_segments, on_length_m=3.0, off_length_m=9.0)]
    if pattern == "short_dashed":
        return [_clip_pattern(line_segments, on_length_m=1.5, off_length_m=1.5)]
    if pattern == "dot_dashed":
        base = _clip_pattern(line_segments, on_length_m=0.91, off_length_m=2.74)
        return [base[::3] if len(base) > 0 else base]
    if pattern == "dotted_1_9":
        return [line_segments[::10]]
    if pattern == "dual":
        left, right = offset_segments(line_segments, dual_offset_m)
        left_pattern, right_pattern = dual_pattern or ("solid", "solid")
        result: list[npt.NDArray[np.float32]] = []
        result.extend(apply_pattern(left, left_pattern, dual_offset_m=dual_offset_m))
        result.extend(apply_pattern(right, right_pattern, dual_offset_m=dual_offset_m))
        return result
    return [line_segments]


def triangulate_polygon_fan(points_xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    unique_points: list[npt.NDArray[np.float32]] = []
    for point in points_xyz:
        if unique_points and np.linalg.norm(unique_points[-1] - point) < 1e-4:
            continue
        unique_points.append(point.astype(np.float32))
    if len(unique_points) >= 2 and np.linalg.norm(unique_points[0] - unique_points[-1]) < 1e-4:
        unique_points = unique_points[:-1]
    if len(unique_points) < 3:
        return np.empty((0, 3, 3), dtype=np.float32)

    base = unique_points[0]
    triangles = [
        np.stack([base, unique_points[i], unique_points[i + 1]], axis=0).astype(np.float32)
        for i in range(1, len(unique_points) - 1)
    ]
    return np.stack(triangles, axis=0).astype(np.float32)


def triangulate_polygon_xy(points_xyz: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    points = np.asarray(points_xyz, dtype=np.float32)
    unique_points: list[npt.NDArray[np.float32]] = []
    for point in points:
        if unique_points and np.linalg.norm(unique_points[-1] - point) < 1e-4:
            continue
        unique_points.append(point.astype(np.float32))
    if len(unique_points) >= 2 and np.linalg.norm(unique_points[0] - unique_points[-1]) < 1e-4:
        unique_points = unique_points[:-1]
    if len(unique_points) < 3:
        return np.empty((0, 3, 3), dtype=np.float32)

    polygon = np.stack(unique_points, axis=0).astype(np.float32)
    polygon_xy = polygon[:, :2]

    signed_area = 0.5 * float(
        np.sum(
            polygon_xy[:, 0] * np.roll(polygon_xy[:, 1], -1)
            - np.roll(polygon_xy[:, 0], -1) * polygon_xy[:, 1]
        )
    )
    if abs(signed_area) < 1e-6:
        return triangulate_polygon_fan(polygon)
    if signed_area < 0.0:
        polygon = polygon[::-1].copy()
        polygon_xy = polygon_xy[::-1].copy()

    def cross2d(
        a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], c: npt.NDArray[np.float32]
    ) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def point_in_triangle(
        point: npt.NDArray[np.float32],
        a: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
        c: npt.NDArray[np.float32],
    ) -> bool:
        ab = cross2d(a, b, point)
        bc = cross2d(b, c, point)
        ca = cross2d(c, a, point)
        eps = 1e-6
        return ab >= -eps and bc >= -eps and ca >= -eps

    indices = list(range(len(polygon)))
    triangles: list[npt.NDArray[np.float32]] = []
    safety_counter = 0
    max_iterations = len(indices) * len(indices)
    while len(indices) > 3 and safety_counter < max_iterations:
        ear_found = False
        for offset, current_idx in enumerate(indices):
            prev_idx = indices[(offset - 1) % len(indices)]
            next_idx = indices[(offset + 1) % len(indices)]
            a = polygon_xy[prev_idx]
            b = polygon_xy[current_idx]
            c = polygon_xy[next_idx]
            if cross2d(a, b, c) <= 1e-6:
                continue

            contains_other = False
            for test_idx in indices:
                if test_idx in (prev_idx, current_idx, next_idx):
                    continue
                if point_in_triangle(polygon_xy[test_idx], a, b, c):
                    contains_other = True
                    break
            if contains_other:
                continue

            triangles.append(
                np.stack(
                    [polygon[prev_idx], polygon[current_idx], polygon[next_idx]], axis=0
                ).astype(np.float32)
            )
            del indices[offset]
            ear_found = True
            break
        if not ear_found:
            return triangulate_polygon_fan(polygon)
        safety_counter += 1

    if len(indices) == 3:
        triangles.append(
            np.stack(
                [polygon[indices[0]], polygon[indices[1]], polygon[indices[2]]], axis=0
            ).astype(np.float32)
        )

    if not triangles:
        return np.empty((0, 3, 3), dtype=np.float32)
    return np.stack(triangles, axis=0).astype(np.float32)


def split_segment_runs(
    line_segments: npt.NDArray[np.float32], atol: float = 1e-4
) -> list[npt.NDArray[np.float32]]:
    if len(line_segments) == 0:
        return []

    runs: list[list[npt.NDArray[np.float32]]] = [
        [line_segments[0, 0].astype(np.float32), line_segments[0, 1].astype(np.float32)]
    ]
    for segment in line_segments[1:]:
        current_run = runs[-1]
        if np.linalg.norm(current_run[-1] - segment[0]) <= atol:
            current_run.append(segment[1].astype(np.float32))
        else:
            runs.append([segment[0].astype(np.float32), segment[1].astype(np.float32)])
    return [np.stack(run, axis=0).astype(np.float32) for run in runs if len(run) >= 2]


def concatenate_segments(
    segment_groups: Iterable[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    non_empty = [group.astype(np.float32) for group in segment_groups if len(group) > 0]
    if not non_empty:
        return np.empty((0, 2, 3), dtype=np.float32)
    return np.concatenate(non_empty, axis=0).astype(np.float32)
