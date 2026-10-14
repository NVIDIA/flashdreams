# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundled-map artwork compositing with calibrated semantic geometry."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import shapely
from omnidreams_game_engine.game_map import load_game_map
from omnidreams_game_engine.game_map import spawn_render as render
from PIL import Image
from shapely.geometry import LineString
from shapely.ops import substring


def main() -> None:
    """Rebuild the seed frame and check its visible semantic paint coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "map",
        choices=("pacific_breakwater", "flashdreams_raceway", "boulevard_district"),
    )
    name = parser.parse_args().map
    # Artwork-specific registration landmarks, measured at 1280x704.
    if name == "pacific_breakwater":
        edge_y = [367, 362, 350, 340, 333, 350, 379, 424]
        hood_y = [638, 612, 598, 586, 580, 576, 575, 579, 590, 611, 640]
        horizon, texture_scale, hood_scale = 333, 0.88, 1.0
    elif name == "flashdreams_raceway":
        edge_y = [351, 344, 330, 319, 314, 324, 338, 352]
        hood_y = [594, 561, 546, 537, 527, 520, 519, 523, 534, 552, 594]
        horizon, texture_scale, hood_scale = 314, 0.70, 0.70
    else:
        edge_y = [388, 374, 349, 329, 317, 339, 368, 409]
        hood_y = [597, 576, 562, 551, 545, 541, 542, 548, 563, 580, 605]
        horizon, texture_scale, hood_scale = 317, 0.80, 0.80
    root = Path(__file__).resolve().parents[1]
    maps = root / "crazy_robotaxi" / "maps"
    game_map = load_game_map(maps / f"{name}.robotaxi.yaml")
    spawn = game_map.default_spawn
    width, height = 1280, 704
    source = np.asarray(
        Image.open(Path(__file__).with_name(f"{name}_artwork.png"))
        .convert("RGB")
        .resize((width, height), Image.Resampling.LANCZOS)
    )
    geometry = render.render_spawn_first_frame(game_map, spawn)
    Image.fromarray(geometry).save(maps / f"{name}_spawn_geometry.png")
    ground_x, ground_y, _, valid = render._camera_ground_intersections(
        spawn, width, height
    )

    # Use the same ground intersections and geometry as the spawn renderer.
    def coverage(shape: shapely.Geometry) -> np.ndarray:
        """Project a ground-plane shape using the compiled camera rays.

        Args:
            shape: Map-space surface or buffered marking geometry.

        Returns:
            Boolean pixel mask at the seed frame resolution.
        """
        mask = np.zeros(width * height, dtype=bool)
        mask[valid] = shapely.intersects_xy(shape, ground_x[valid], ground_y[valid])
        return mask.reshape(height, width)

    road = coverage(
        render._polygon_geometry(
            [element.surface_world for element in game_map.elements]
        )
    )
    curb = coverage(
        render._line_geometry(
            [
                curb.polyline_world
                for element in game_map.elements
                for curb in element.curbs
            ],
            render._CURB_WIDTH_M,
        )
    )
    paint_masks = {}
    for color in ("WHITE", "YELLOW"):
        paint_lines = [
            mark.polyline_world
            for mark in game_map.line_markings
            if mark.color == color
        ]
        for divider in game_map.lane_dividers:
            if divider.color != color:
                continue
            if divider.style == "DASHED_SINGLE":
                line = LineString(divider.polyline_world[:, :2])
                for offset in np.arange(0, line.length, 9.0):
                    dash = substring(line, offset, min(offset + 3.0, line.length))
                    if isinstance(dash, LineString):
                        paint_lines.append(np.asarray(dash.coords))
            else:
                paint_lines.append(divider.polyline_world)
        paint_masks[color] = coverage(
            render._line_geometry(paint_lines, render._PAINT_WIDTH_M)
        )

    # Register the generated barrier feet to the actual paved boundary.
    # These landmarks belong only to the checked-in artwork, not the map.
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    source_edge = np.interp(
        np.arange(width),
        [0, 100, 300, 500, 640, 800, 1000, 1279],
        edge_y,
    )
    # Use the connected foreground surface, excluding distant track slivers.
    assert road[-1].all(), "Artwork requires road across the foreground"
    target_edge = height - np.argmax((~road)[::-1], axis=0)
    assert np.all(target_edge > 0), "Unexpected sky-only columns"
    if name != "pacific_breakwater":
        # Homes and grandstands occlude remote ground beyond the foreground
        # road; do not paint distant map lines through above-ground scenery.
        foreground = y >= target_edge[None, :]
        road &= foreground
        curb &= y >= target_edge[None, :] - 1
        paint_masks = {color: mask & foreground for color, mask in paint_masks.items()}
    scenery_y = (y * source_edge[None, :] / target_edge[None, :]).astype(np.float32)
    scenery = cv2.remap(
        source,
        x.astype(np.float32),
        scenery_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Remove the generated paint before reusing its asphalt texture.
    hood_top = np.interp(
        np.arange(width),
        [0, 100, 200, 300, 400, 500, 640, 800, 950, 1100, 1279],
        hood_y,
    )
    source_road = (y > source_edge[None, :] + 2) & (y < hood_top[None, :] - 3)
    generated_paint = source_road & (source.min(axis=2) > 140)
    generated_paint = cv2.dilate(
        generated_paint.astype(np.uint8), np.ones((7, 7), np.uint8)
    )
    texture = source.copy()
    for row in range(height):
        transitions = np.diff(np.pad(generated_paint[row].astype(np.int16), (1, 1)))
        for start, end in zip(
            np.flatnonzero(transitions == 1),
            np.flatnonzero(transitions == -1),
            strict=True,
        ):
            size = end - start
            for candidate in (start - size - 5, end + 5):
                if (
                    candidate >= 0
                    and candidate + size <= width
                    and source_road[row, candidate : candidate + size].all()
                    and not generated_paint[row, candidate : candidate + size].any()
                ):
                    texture[row, start:end] = source[row, candidate : candidate + size]
                    break
    asphalt = cv2.remap(
        texture,
        (640 + (x - 640) * texture_scale).astype(np.float32),
        (horizon + (y - 347) * texture_scale).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    asphalt = np.minimum(asphalt, 160)
    result = scenery.copy()
    result[road] = asphalt[road]
    result[curb] = [168, 165, 153]
    result[paint_masks["WHITE"]] = render._WHITE_PAINT_RGB
    result[paint_masks["YELLOW"]] = render._YELLOW_PAINT_RGB

    # Preserve the photographed hood; it naturally occludes near-field road.
    hood_sample_y = (height - (height - y) / hood_scale).astype(np.float32)
    hood_source = cv2.remap(
        source,
        x.astype(np.float32),
        hood_sample_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    target_hood_top = height - (height - hood_top) * hood_scale
    hood_alpha = np.clip(y - target_hood_top[None, :] + 0.5, 0.0, 1.0)
    result = np.rint(
        result * (1 - hood_alpha[..., None]) + hood_source * hood_alpha[..., None]
    ).astype(np.uint8)
    visible = road & (hood_alpha == 0) & ~curb
    observed_paint = (result.min(axis=2) > 190) & visible
    expected_paint = paint_masks["WHITE"] & visible
    assert np.array_equal(observed_paint, expected_paint), "Paint projection drift"
    observed_yellow = (
        (result[..., 0] > 190) & (result[..., 1] > 140) & (result[..., 2] < 90)
    ) & visible
    assert np.array_equal(observed_yellow, paint_masks["YELLOW"] & visible), (
        "Yellow divider projection drift"
    )
    assert np.count_nonzero(expected_paint) > 500, "Missing road markings"
    output = maps / f"{name}_initial_frame.png"
    Image.fromarray(result).save(output)
    # Show semantic boundaries and full divider trajectories through dash gaps.
    overlay = result.copy()
    guide = (
        np.all(geometry == render._WHITE_PAINT_RGB, axis=2)
        | np.all(geometry == render._YELLOW_PAINT_RGB, axis=2)
        | np.all(geometry == render._CURB_RGB, axis=2)
        | np.all(geometry == render._BOUNDARY_RGB, axis=2)
    ) & (hood_alpha == 0)
    if name != "pacific_breakwater":
        guide &= road | curb
    overlay[guide] = [30, 240, 210]
    Image.fromarray(overlay).save(maps / f"{name}_alignment.png")
    print(f"Saved {output}; white and yellow semantic paint masks match exactly.")


if __name__ == "__main__":
    main()
