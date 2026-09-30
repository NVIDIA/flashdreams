# Launch-only arguments

[Options index](../OPTIONS.md)

These arguments do not correspond to an Options field:

| CLI flag | Purpose |
| --- | --- |
| `--config PATH` | Selects the user settings YAML file. |
| `--game-mode` | Chooses `taxi` or `race` and skips the mode menu. |
| `--map PATH` | Chooses a map and skips map selection. |
| `--race-course ID` | Chooses a race course; requires race mode. |
| `--force-map-recompile` | Rebuilds the compiled map. |
| `--ui`, `--no-ui` | Enables or disables the ImGui UI. |
| `--controls-dir PATH` | Selects the separate controls settings directory. |

The runner name selects the starting model preset before application arguments are parsed. Controller bindings are edited under **CONTROLS** on the mode menu and saved in separate `controls/keyboard.yaml`, `controls/gamepad.yaml`, and `controls/wheel.yaml` files.
