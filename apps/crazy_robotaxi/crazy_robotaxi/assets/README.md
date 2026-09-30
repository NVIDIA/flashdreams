# Assets

This directory contains map-independent runtime data for Crazy Robotaxi.

## `obstacle_vehicle_tracks_v1.npz`

This catalog contains numeric vehicle trajectories used by the optional
live-edit obstacle ability. The archive stores relative timestamps, local
center translations, orientations, first-sample dimensions, object-type
codes, sample offsets, and initial heights for 668 car and truck tracks.

Runtime loading uses `allow_pickle=False`. The source scene used to derive the
catalog is not distributed with the package.

## Downloaded checkpoints

Additional style and corrector checkpoints for enabled live-edit features are
resolved during application startup. Downloads go to
`artifacts/crazy_robotaxi/live_edit` relative to the working directory; they are
separate from these packaged assets. Explicit checkpoint paths in settings
take precedence. See the [live-edit abilities section](../../README.md#optional-live-edit-abilities).

## Maps

Bundled maps live in [../maps/](../maps/). They use the
[shared node-graph map format](../../../omnidreams_game_engine/NODE_GRAPH_MAP_FORMAT.md).
Spawn images may be map-relative files or `package://package/resource`
references. Both bundled maps use
`package://omnidreams_game_engine/screenshot.jpg`. Spawns without an image receive
a deterministic first-person road preview during compilation.
