# OmniDreams Game Engine

This package owns the reusable scene, simulation, input, conditioning,
presentation, and inference runtime used by standalone OmniDreams games.
Games inject an application policy into `InteractiveDriveApp`.

Standalone games may author semantic road networks with the engine's
[node-graph map format](NODE_GRAPH_MAP_FORMAT.md). The schema keeps authored
topology, generated surfaces, curb collision geometry, and the derived directed
lane graph as separate runtime concepts.
