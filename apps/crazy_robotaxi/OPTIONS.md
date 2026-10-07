# Crazy Robotaxi Options menu

Open **OPTIONS** from **SELECT GAME MODE**. Use the guide for the page you are viewing. Each guide lists the exact on-screen labels, setting ownership, and available CLI flags. Repeated labels such as **Seed:** and **Enabled:** are identified by their surrounding menu headings.

Hover an option's label or editor to see its description and full YAML path.

The guides list the supported overrides for both Options and `config.yaml`. Model architecture and embedding formats come from the selected runner preset. Internal checkpoint hooks, benchmark-only fields, and raster fields unused by the game are excluded. Deprecated overrides in existing YAML files are ignored and removed on the next save. Unknown keys produce an error.

| Options page | Guide |
| --- | --- |
| GAME | [Gameplay, taxi rules, vehicle physics, race, and effects](options/game.md) |
| MODEL | [Model overview and component guides](options/model.md) |
| RENDERER | [Main-camera raster and top-down view](options/renderer.md) |
| PRESENTATION | [Display size and HUD](options/presentation.md) |
| LIVE EDIT | [Style, coins, items, weather, obstacles, and map context](options/live-edit.md) |
| RUNTIME | [Generation limits and warmup](options/runtime.md) |
| DIAGNOSTICS | [Profiling and input latency](options/diagnostics.md) |

The MODEL guide links to separate documents for the transformer, optimized attention, scheduler, and encoders.

## Settings ownership

**Defined by** identifies the project that declares each setting's name and type. **Crazy Robotaxi** fields belong to this app and can be renamed here. **Game engine** fields come from the shared `omnidreams_game_engine` package. **OmniDreams** and **FlashDreams** fields come from the model integration and framework. Crazy Robotaxi saves all four kinds in the same `config.yaml`; the source names of shared engine, OmniDreams, and FlashDreams fields should be kept intact. A more readable menu label for one of those fields may therefore differ from its YAML key. For inherited fields, **Defined by** names the project that first declared the field, even when Crazy Robotaxi overrides its default value.

## Editing and saving

A checkbox changes a Boolean, a drop-down presents a fixed set of choices, and other fields accept text. For lists and structured values, enter YAML flow syntax; a list looks like `[1000, 500]`. A blank optional field means `None`.

**SAVE** writes the draft and stays on this screen. **RESET TO DEFAULTS** resets the draft to the selected runner's defaults; press **SAVE** to persist that reset. **EXIT** returns to the mode menu. When edits are pending it says **EXIT WITHOUT SAVING** and discards them. Presentation changes take effect on save; other changes need a process restart, as the screen warns. A **COMMAND-LINE OVERRIDE ACTIVE** note means a launch argument controls the current run even if you save a different value.

## Configuration file and CLI overrides

The screen displays the settings file path. By default it is `$XDG_CONFIG_HOME/crazy-robotaxi/config.yaml`, or `~/.config/crazy-robotaxi/config.yaml` if `XDG_CONFIG_HOME` is unset. `--config PATH` selects another file. The YAML path in each tooltip gives the sequence of nested keys in the settings file. Only values different from the runner preset are saved, so the values visible on **MODEL** can vary by runner. Paths entered relative to the settings file resolve relative to its directory. Units are in field names: `_m` means meters, `_s` seconds, `_mps` meters per second, `_rad` radians, and `_deg` degrees.

CLI flags are application arguments after the runner's `--`. An em dash means there is no direct CLI flag for that setting; use Options or `config.yaml`. Where both `--flag` and `--no-flag` are listed, the latter explicitly turns the setting off. An explicit CLI value takes precedence over the saved YAML for the current run. `--seed` sets both taxi and model seeds; `--game-seed` and `--model-seed` select them independently, with `--model-seed` taking precedence over `--seed` for the model and `--seed` taking precedence over `--game-seed` for the taxi if both are passed. `--profile-input-latency [TRACE_PATH]` enables profiling and sets the trace path together; without a path it uses the default trace file.

## Launch and controls

See [Launch arguments and controls files](options/launch.md) for arguments that select the configuration file, game mode, map, UI, or controls directory.
