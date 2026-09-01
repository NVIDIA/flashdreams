# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for explicit Qwen spawn-image authoring."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
from crazy_robotaxi.spawn_images import (
    add_spawn_road_noise,
    generate_spawn_images,
    save_settled_spawn_image,
    spawn_image_prompt,
    spawn_image_seed,
)
from omnidreams_game_engine.game_map import (
    load_game_map,
    render_spawn_first_frame_with_road_mask,
)
from PIL import Image

pytestmark = pytest.mark.ci_cpu

_SOURCE = (
    Path(__file__).parents[1]
    / "crazy_robotaxi"
    / "maps"
    / "flashdreams_raceway.robotaxi.yaml"
)


def _source_without_managed_image() -> str:
    return "".join(
        line
        for line in _SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
        if not line.lstrip().startswith("image: crazy-robotaxi-")
    )


class _Editor:
    def __init__(self) -> None:
        self.calls: list[tuple[Image.Image, str, dict[str, object]]] = []

    def generate(
        self, image: Image.Image, prompt: str, **kwargs: object
    ) -> Image.Image:
        self.calls.append((image.copy(), prompt, kwargs))
        return Image.new(
            "RGB", cast(tuple[int, int], kwargs["output_size"]), (11, 22, 33)
        )


def test_generates_semantic_spawn_and_updates_yaml_without_reformatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / _SOURCE.name
    path.write_text(_source_without_managed_image(), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        "crazy_robotaxi.spawn_images.compile_game_map", lambda *a, **k: None
    )
    editor = _Editor()

    result = generate_spawn_images(
        path,
        editor,
        resolution_wh=(160, 96),
    )

    assert result == path.resolve()
    assert len(editor.calls) == 1
    source_image, prompt, kwargs = editor.calls[0]
    game_map = load_game_map(path)
    variant = game_map.default_spawn.variants[0]
    assert variant.image == (f"{game_map.map_id}.spawn-images/race-start--default.png")
    assert variant.image is not None
    semantic, road_mask = render_spawn_first_frame_with_road_mask(
        game_map, game_map.default_spawn, resolution_wh=(160, 96)
    )
    assert np.array_equal(np.asarray(source_image), semantic)
    assert road_mask.any()
    assert not road_mask.all()
    assert "at day" in prompt
    assert "only road" in prompt
    assert "no aerial" in prompt
    assert kwargs["seed"] == spawn_image_seed(game_map, game_map.default_spawn, variant)
    assert "cars" in cast(str, kwargs["negative_prompt"])
    image_path = path.parent / variant.image
    generated = np.asarray(Image.open(image_path))
    painted = road_mask & (semantic.max(axis=2) >= 100)
    paved = road_mask & ~painted
    assert np.array_equal(generated[painted], semantic[painted])
    assert not np.array_equal(generated[paved], semantic[paved])
    assert np.all(generated[~road_mask] == np.asarray([11, 22, 33]))
    after = path.read_text(encoding="utf-8")
    assert after.replace(f"        image: {variant.image}\n", "") == before

    second = _Editor()
    generate_spawn_images(path, second, resolution_wh=(160, 96))
    assert second.calls == []

    settled = np.full((96, 160, 3), 44, dtype=np.uint8)
    assert save_settled_spawn_image(path, "default", settled) == image_path
    assert np.array_equal(np.asarray(Image.open(image_path)), settled)


def test_custom_bundle_preserves_authored_image_and_generates_only_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source" / _SOURCE.name
    source.parent.mkdir()
    source.write_text(
        _source_without_managed_image().replace(
            "        time_of_day: day\n",
            "        image: authored.png\n        time_of_day: day\n",
            1,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (4, 4), "red").save(source.parent / "authored.png")
    destination = tmp_path / "bundle" / source.name
    monkeypatch.setattr(
        "crazy_robotaxi.spawn_images.compile_game_map", lambda *a, **k: None
    )
    editor = _Editor()

    generate_spawn_images(source, editor, output_map=destination)

    assert editor.calls == []
    assert (destination.parent / "authored.png").is_file()
    assert load_game_map(destination).default_spawn.variants[0].image == "authored.png"
    with pytest.raises(ValueError, match="only replaces managed"):
        save_settled_spawn_image(
            destination,
            "default",
            np.zeros((4, 4, 3), dtype=np.uint8),
        )


def test_prompt_includes_time_of_day() -> None:
    game_map = load_game_map(_SOURCE)
    variant = game_map.default_spawn.variants[0]
    assert variant.prompt in spawn_image_prompt(variant)
    assert variant.time_of_day in spawn_image_prompt(variant)


def test_road_noise_preserves_scenery_and_markings() -> None:
    game_map = load_game_map(_SOURCE)
    spawn = game_map.default_spawn
    semantic, road_mask = render_spawn_first_frame_with_road_mask(
        game_map, spawn, resolution_wh=(160, 96)
    )
    qwen = np.full_like(semantic, 180)

    day = add_spawn_road_noise(
        qwen,
        semantic,
        road_mask,
        time_of_day="day",
        seed=123,
    )
    repeated = add_spawn_road_noise(
        qwen,
        semantic,
        road_mask,
        time_of_day="day",
        seed=123,
    )
    night = add_spawn_road_noise(
        qwen,
        semantic,
        road_mask,
        time_of_day="night",
        seed=123,
    )

    painted = road_mask & (semantic.max(axis=2) >= 100)
    paved = road_mask & ~painted
    assert np.array_equal(day, repeated)
    assert np.all(day[~road_mask] == 180)
    assert np.array_equal(day[painted], semantic[painted])
    grain = day[paved].astype(np.float32) - semantic[paved]
    assert 6.0 < grain.std() < 10.0
    assert day[paved].mean() > night[paved].mean() + 25.0
