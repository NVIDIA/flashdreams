# Future Game-Map Conditioning

The node-graph map schema currently describes roads, lanes, intersections,
driveways, parking lots, curb barriers, spawns, and visual seed variants. The
following conditioning classes are candidates for future schema versions.
They are not committed additions to the format.

## Roadside objects

- Poles, including placement, height, radius, and visual category.
- Traffic signs, including sign type, facing direction, and supporting pole.
- Traffic lights, including signal heads, mounting, orientation, and controlled
  approaches or lanes.

## Road-surface features

- Wait or stop lines associated with a lane or intersection approach.
- Crosswalk polygons and their relationship to intersection approaches.
- Road islands and medians, including traversability, curb treatment, and
  whether they divide opposing traffic.

## Traffic

- Authored traffic sources, destinations, routes, spawn rates, and vehicle
  classes.
- Rules for connecting authored traffic to lane successors, signals, and game
  difficulty settings.
- A controller-owned runtime representation shared by routing, physics, and
  rendering.

Each addition should define authoring semantics first, then specify validation,
compiled conditioning output, preview rendering, collision behavior, and
runtime ownership.
