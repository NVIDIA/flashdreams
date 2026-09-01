# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Qwen generation of semantic-map spawn images."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from omnidreams_game_engine.game_map import (
    compile_game_map,
    load_game_map,
    render_spawn_first_frame_with_road_mask,
    resolve_seed_asset,
)
from omnidreams_game_engine.game_map.types import (
    GameMapSpawn,
    GameMapVisualVariant,
    ResolvedGameMap,
)
from PIL import Image
from yaml.nodes import MappingNode, ScalarNode, SequenceNode

SPAWN_IMAGE_AUTHORING_VERSION = "2"
"""Version mixed into deterministic generation seeds."""


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in value
    ).strip("-")


def spawn_image_seed(
    game_map: ResolvedGameMap,
    spawn: GameMapSpawn,
    variant: GameMapVisualVariant,
) -> int:
    """Return a stable per-map, spawn, and variant seed."""
    digest = hashlib.sha256(
        repr(
            (
                SPAWN_IMAGE_AUTHORING_VERSION,
                game_map.map_id,
                spawn.spawn_id,
                variant.name,
                variant.prompt,
                variant.time_of_day,
            )
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def spawn_image_prompt(variant: GameMapVisualVariant) -> str:
    """Build the image-edit instruction paired with the semantic road render."""
    return (
        "Use the input image as a fixed perspective template. The dark gray paved "
        "region is the only road and must remain in exactly the same screen-space "
        "shape, position, direction, and perspective. Do not add, remove, bend, "
        "branch, intersect, or relocate any road. Keep an eye-level, straight-ahead "
        "dashcam viewpoint with the horizon at the same height as the input; no "
        "aerial, elevated, or downward-looking view. The road must be completely "
        "empty: no cars, vehicles, people, or movable objects. Add photorealistic "
        "scenery only around the fixed road and in the sky. "
        f"The scene is at {variant.time_of_day}. Surrounding visual theme: "
        f"{variant.prompt.strip()}"
    )


_SPAWN_NEGATIVE_PROMPT = (
    "cars, vehicles, traffic, pedestrians, people, objects on the road, aerial "
    "view, elevated view, downward-looking camera, extra roads, intersecting "
    "roads, branching roads, invented track, altered road geometry"
)

_ROAD_NOISE_OFFSET = {"dawn": 4.0, "day": 10.0, "dusk": -4.0, "night": -24.0}
_ROAD_NOISE_STANDARD_DEVIATION = 8.0


def add_spawn_road_noise(
    image_rgb: np.ndarray,
    semantic_rgb: np.ndarray,
    road_mask: np.ndarray,
    *,
    time_of_day: str,
    seed: int,
) -> np.ndarray:
    """Lock semantic road geometry while adding grain only to dark pavement."""
    image = np.asarray(image_rgb)
    semantic = np.asarray(semantic_rgb)
    mask = np.asarray(road_mask)
    if image.shape != semantic.shape or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb and semantic_rgb must be matching HWC RGB arrays")
    if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("road_mask must be a bool array matching the image dimensions")
    try:
        offset = _ROAD_NOISE_OFFSET[time_of_day]
    except KeyError as error:
        values = ", ".join(_ROAD_NOISE_OFFSET)
        raise ValueError(f"time_of_day must be one of: {values}") from error

    result = image.astype(np.uint8, copy=True)
    result[mask] = semantic[mask]
    paved = mask & (semantic.max(axis=2) < 100) & (np.ptp(semantic, axis=2) < 10)
    grain = np.random.default_rng(seed).normal(
        offset, _ROAD_NOISE_STANDARD_DEVIATION, (int(paved.sum()), 1)
    )
    result[paved] = np.clip(semantic[paved].astype(np.float32) + grain, 0, 255)
    return result


def _managed_relative_path(
    game_map: ResolvedGameMap,
    spawn: GameMapSpawn,
    variant: GameMapVisualVariant,
) -> Path:
    return Path(f"{game_map.map_id}.spawn-images") / (
        f"{_slug(spawn.spawn_id)}--{_slug(variant.name)}.png"
    )


def _is_managed(game_map: ResolvedGameMap, image: str | None) -> bool:
    return image is None or Path(image).parts[:1] == (
        f"{game_map.map_id}.spawn-images",
    )


def _atomic_save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".png"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.convert("RGB").save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping_value(node: MappingNode, name: str) -> Any:
    for key, value in node.value:
        if isinstance(key, ScalarNode) and key.value == name:
            return value
    raise KeyError(name)


def _write_image_fields(path: Path, images: dict[tuple[str, str], str]) -> None:
    text = path.read_text(encoding="utf-8")
    root = yaml.compose(text)
    if not isinstance(root, MappingNode):
        raise ValueError(f"Expected a YAML mapping in {path}")
    spawns = _mapping_value(root, "spawns")
    if not isinstance(spawns, SequenceNode):
        raise ValueError(f"Expected spawns to be a sequence in {path}")
    lines = text.splitlines(keepends=True)
    edits: list[tuple[int, bool, str]] = []
    for spawn_node in spawns.value:
        if not isinstance(spawn_node, MappingNode):
            continue
        spawn_id = _mapping_value(spawn_node, "id").value
        variants = _mapping_value(spawn_node, "variants")
        if not isinstance(variants, MappingNode):
            continue
        for name_node, variant_node in variants.value:
            if not isinstance(name_node, ScalarNode) or not isinstance(
                variant_node, MappingNode
            ):
                continue
            key = (spawn_id, name_node.value)
            if key not in images:
                continue
            image_node = None
            for field, value in variant_node.value:
                if isinstance(field, ScalarNode) and field.value == "image":
                    image_node = value
                    break
            rendered = (
                " " * (name_node.start_mark.column + 2) + f"image: {images[key]}\n"
            )
            edits.append(
                (
                    variant_node.start_mark.line
                    if image_node is None
                    else image_node.start_mark.line,
                    image_node is not None,
                    rendered,
                )
            )
    for line_index, replace, rendered in sorted(edits, reverse=True):
        lines[line_index : line_index + int(replace)] = [rendered]
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(path)


def _copy_bundle(source: Path, destination: Path, game_map: ResolvedGameMap) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for spawn in game_map.spawns:
        for variant in spawn.variants:
            image = variant.image
            if image is None or image.startswith("package://"):
                continue
            resolved = resolve_seed_asset(source, image)
            target = destination.parent / image
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved, target)


def generate_spawn_images(
    map_path: str | Path,
    editor: Any,
    *,
    output_map: str | Path | None = None,
    resolution_wh: tuple[int, int] = (1280, 704),
    force: bool = False,
    num_inference_steps: int | None = None,
    progress: Callable[[int, int, str, str], None] | None = None,
) -> Path:
    """Generate managed spawn images, update YAML, and compile the map.

    Existing authored images are never overwritten. A variant is managed only
    when its image is absent or already points inside ``<map-id>.spawn-images``.
    """
    source = Path(map_path).expanduser().resolve()
    game_map = load_game_map(source)
    destination = (
        source if output_map is None else Path(output_map).expanduser().resolve()
    )
    if destination != source:
        _copy_bundle(source, destination, game_map)
        game_map = load_game_map(destination)

    candidates = [
        (spawn, variant)
        for spawn in game_map.spawns
        for variant in spawn.variants
        if _is_managed(game_map, variant.image)
        and (
            force
            or variant.image is None
            or not (destination.parent / variant.image).is_file()
        )
    ]
    generated_fields: dict[tuple[str, str], str] = {}
    for index, (spawn, variant) in enumerate(candidates, start=1):
        relative = _managed_relative_path(game_map, spawn, variant)
        if progress is not None:
            progress(index, len(candidates), spawn.spawn_id, variant.name)
        seed = spawn_image_seed(game_map, spawn, variant)
        semantic, road_mask = render_spawn_first_frame_with_road_mask(
            game_map, spawn, resolution_wh=resolution_wh
        )
        image = editor.generate(
            Image.fromarray(semantic, mode="RGB"),
            spawn_image_prompt(variant),
            output_size=resolution_wh,
            seed=seed,
            num_inference_steps=num_inference_steps,
            negative_prompt=_SPAWN_NEGATIVE_PROMPT,
        )
        generated = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if generated.shape != semantic.shape:
            raise ValueError(
                f"Qwen returned {generated.shape}, expected {semantic.shape}"
            )
        generated = add_spawn_road_noise(
            generated,
            semantic,
            road_mask,
            time_of_day=variant.time_of_day,
            seed=seed,
        )
        image = Image.fromarray(generated, mode="RGB")
        _atomic_save(image, destination.parent / relative)
        generated_fields[(spawn.spawn_id, variant.name)] = relative.as_posix()

    for spawn in game_map.spawns:
        for variant in spawn.variants:
            if variant.image is not None and _is_managed(game_map, variant.image):
                generated_fields.setdefault(
                    (spawn.spawn_id, variant.name), variant.image
                )
    if generated_fields:
        _write_image_fields(destination, generated_fields)
    compile_game_map(destination, force=True)
    return destination


def save_settled_spawn_image(
    map_path: str | Path,
    variant_name: str,
    image_rgb: np.ndarray,
) -> Path:
    """Replace one managed default-spawn image with a world-model result."""
    source = Path(map_path).expanduser().resolve()
    game_map = load_game_map(source)
    variant = next(
        (
            candidate
            for candidate in game_map.default_spawn.variants
            if candidate.name == variant_name
        ),
        None,
    )
    if variant is None:
        raise ValueError(f"Unknown spawn-image variant {variant_name!r}")
    if variant.image is None or not _is_managed(game_map, variant.image):
        raise ValueError("World-model settlement only replaces managed spawn images")
    array = np.asarray(image_rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "Settled spawn image must be an HWC uint8 RGB array; got "
            f"shape={array.shape}, dtype={array.dtype}"
        )
    destination = resolve_seed_asset(source, variant.image)
    _atomic_save(Image.fromarray(array, mode="RGB"), destination)
    compile_game_map(source, force=True)
    return destination


__all__ = [
    "add_spawn_road_noise",
    "generate_spawn_images",
    "save_settled_spawn_image",
    "spawn_image_prompt",
    "spawn_image_seed",
]
