# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PNG export for semantic-map spawn previews."""

from pathlib import Path

from PIL import Image

from omnidreams_game_engine.game_map.spawn_render import render_spawn_first_frame


def write_spawn_first_frame_preview(
    source: Path,
    destination: Path,
    *,
    spawn_id: str | None = None,
) -> Path:
    """Write the deterministic fallback render for one authored spawn.

    Args:
        source: Semantic map YAML path.
        destination: PNG path to create.
        spawn_id: Spawn identifier; ``None`` selects the first spawn.

    Returns:
        Resolved output path.

    Raises:
        GameMapError: ``spawn_id`` does not identify a map spawn.
    """
    from omnidreams_game_engine.game_map._schema import GameMapError
    from omnidreams_game_engine.game_map.loader import load_game_map

    game_map = load_game_map(source)
    if spawn_id is None:
        spawn = game_map.default_spawn
    else:
        spawn = next(
            (
                candidate
                for candidate in game_map.spawns
                if candidate.spawn_id == spawn_id
            ),
            None,
        )
        if spawn is None:
            available = ", ".join(item.spawn_id for item in game_map.spawns)
            raise GameMapError(
                f"Unknown spawn {spawn_id!r}; available spawns: {available}"
            )
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_spawn_first_frame(game_map, spawn)).save(output)
    return output
