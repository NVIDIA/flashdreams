# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a supplied OBJ vehicle and rasterize its fixed chase-camera view."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def chase_projection(
    points: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Project right/forward/up coordinates through the presentation chase camera."""
    width, height = size
    x, y, z = np.moveaxis(points, -1, 0)
    depth = (y + 14) * 0.809 - (z - 12) * 0.588
    up = (z - 12) * 0.809 + (y + 14) * 0.588
    xy = np.stack(
        (
            width / 2 + width * 0.85 * x / np.maximum(depth, 0.1),
            height * 0.60 - width * 0.85 * up / np.maximum(depth, 0.1),
        ),
        axis=-1,
    )
    return xy, depth


def prepare_obj(path: Path, output: Path, *, forward_sign: int = -1) -> None:
    """Convert an OBJ with millimeter coordinates into a compact FLU mesh cache."""
    materials: dict[str, dict] = {}
    current: dict = {}
    mtl = path.with_suffix(".mtl")
    for line in mtl.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "newmtl":
            current = {"color": [1.0, 1.0, 1.0], "alpha": 1.0}
            materials[fields[1]] = current
        elif fields[0] == "Kd":
            current["color"] = list(map(float, fields[1:4]))
        elif fields[0] == "d":
            current["alpha"] = float(fields[1])
        elif fields[0] == "map_Kd":
            texture = (path.parent / " ".join(fields[1:])).resolve()
            if not texture.is_relative_to(path.parent.resolve()):
                raise ValueError("Texture path escapes the model directory")
            with Image.open(texture) as image:
                current["texture"] = np.asarray(image.convert("RGB"))
    vertices, uvs, faces, face_uvs, face_materials = [], [], [], [], []
    names = list(materials)
    material = 0
    with path.open() as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith("vt "):
                uvs.append(list(map(float, line.split()[1:3])))
            elif line.startswith("usemtl "):
                material = names.index(line.split()[1])
            elif line.startswith("f "):
                corners = [field.split("/") for field in line.split()[1:]]
                indices = [
                    int(field[0]) - 1
                    if int(field[0]) > 0
                    else len(vertices) + int(field[0])
                    for field in corners
                ]
                texture_indices = [
                    int(field[1]) - 1 if len(field) > 1 and field[1] else -1
                    for field in corners
                ]
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[i], indices[i + 1]])
                    face_uvs.append(
                        [texture_indices[0], texture_indices[i], texture_indices[i + 1]]
                    )
                    face_materials.append(material)
    points = np.asarray(vertices, dtype=np.float32) / 1000
    if not np.isfinite(points).all():
        raise ValueError("Mesh contains non-finite vertices")
    points[:, :2] -= (points[:, :2].max(axis=0) + points[:, :2].min(axis=0)) / 2
    points[:, 2] -= points[:, 2].min()
    points = np.stack(
        (points[:, 1] * forward_sign, -points[:, 0] * forward_sign, points[:, 2]),
        axis=-1,
    )
    faces = np.asarray(faces, dtype=np.int32)
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Mesh face index is out of range")
    colors = np.ones((len(faces), 3), dtype=np.float32)
    alpha = np.ones(len(faces), dtype=np.float32)
    material_ids = np.asarray(face_materials)
    face_uvs = np.asarray(face_uvs)
    uv = np.asarray(uvs)
    for i, name in enumerate(names):
        mask = material_ids == i
        entry = materials[name]
        colors[mask] = entry["color"]
        alpha[mask] = entry["alpha"]
        if "texture" in entry:
            selected = mask & np.all(face_uvs >= 0, axis=1)
            coordinates = uv[face_uvs[selected]].mean(axis=1)
            texture = entry["texture"]
            tx = np.clip(
                (coordinates[:, 0] * (texture.shape[1] - 1)).astype(int),
                0,
                texture.shape[1] - 1,
            )
            ty = np.clip(
                ((1 - coordinates[:, 1]) * (texture.shape[0] - 1)).astype(int),
                0,
                texture.shape[0] - 1,
            )
            colors[selected] *= texture[ty, tx] / 255
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, vertices=points, faces=faces, colors=colors, alpha=alpha
    )
    print(
        f"Prepared {len(points)} vertices / {len(faces)} triangles; FLU bounds {np.ptp(points, axis=0).tolist()}",
        flush=True,
    )


def render_vehicle(path: Path, size: tuple[int, int]) -> Image.Image:
    """Rasterize the supplied vehicle with a depth buffer and material colors."""
    with np.load(path, allow_pickle=False) as model:
        vertices, faces = model["vertices"], model["faces"]
        colors, alpha = model["colors"], model["alpha"]
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.size == 0
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or colors.shape != (len(faces), 3)
        or alpha.shape != (len(faces),)
        or not np.isfinite(colors).all()
        or not np.isfinite(alpha).all()
    ):
        raise ValueError("Invalid vehicle mesh arrays")
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    light = np.array([-0.3, -0.45, 0.84])
    colors = np.clip(colors * (0.6 + 0.4 * np.abs(normals @ light))[:, None], 0, 1)
    # ponytail: The heading-following chase camera permits one cached ego view;
    # animated body pitch/roll requires rerasterizing the mesh per frame.
    factor = 3
    resolution = (size[0] * factor, size[1] * factor)
    # The prepared rig's 1.6 m body is centered 0.8 m above its vehicle floor.
    camera_points = np.stack(
        (-vertices[:, 1], vertices[:, 0], vertices[:, 2] - 0.8), axis=-1
    )
    projected, depths = chase_projection(camera_points, resolution)
    pixel_triangles, triangle_depth = projected[faces], depths[faces]
    min_xy = np.floor(pixel_triangles.min(axis=1)).astype(int)
    max_xy = np.ceil(pixel_triangles.max(axis=1)).astype(int)
    visible = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < resolution[0])
        & (min_xy[:, 1] < resolution[1])
        & np.all(triangle_depth > 0.05, axis=1)
    )
    indices = np.flatnonzero(visible)
    indices = indices[np.argsort(triangle_depth[indices].mean(axis=1))[::-1]]
    canvas = np.zeros((resolution[1], resolution[0], 4), dtype=np.float32)
    zbuffer = np.full((resolution[1], resolution[0]), np.inf)
    for i in indices:
        x0, y0 = np.maximum(min_xy[i], 0)
        x1, y1 = np.minimum(max_xy[i], [resolution[0] - 1, resolution[1] - 1])
        points = pixel_triangles[i]
        denominator = (points[1, 1] - points[2, 1]) * (points[0, 0] - points[2, 0]) + (
            points[2, 0] - points[1, 0]
        ) * (points[0, 1] - points[2, 1])
        if abs(denominator) < 1e-9:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a = (
            (points[1, 1] - points[2, 1]) * (xx - points[2, 0])
            + (points[2, 0] - points[1, 0]) * (yy - points[2, 1])
        ) / denominator
        b = (
            (points[2, 1] - points[0, 1]) * (xx - points[2, 0])
            + (points[0, 0] - points[2, 0]) * (yy - points[2, 1])
        ) / denominator
        c = 1 - a - b
        depth = 1 / (
            a / triangle_depth[i, 0]
            + b / triangle_depth[i, 1]
            + c / triangle_depth[i, 2]
        )
        region = canvas[y0 : y1 + 1, x0 : x1 + 1]
        zregion = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        mask = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6) & (depth < zregion)
        opacity = alpha[i]
        if opacity >= 0.99:
            region[mask, :3] = colors[i]
            region[mask, 3] = 1
            zregion[mask] = depth[mask]
        else:
            region[mask, :3] = colors[i] * opacity + region[mask, :3] * (1 - opacity)
            region[mask, 3] = opacity + region[mask, 3] * (1 - opacity)
    image = Image.fromarray(np.round(np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    return image.resize(size, Image.Resampling.LANCZOS)


def main() -> None:
    """Prepare a local OBJ asset without loading a model or using a GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-sign", type=int, choices=(-1, 1), default=-1)
    args = parser.parse_args()
    prepare_obj(args.obj, args.output, forward_sign=args.forward_sign)
    render_vehicle(args.output, (604, 330)).save(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
