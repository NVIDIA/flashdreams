# Assets

This directory contains map-independent runtime data for Crazy Robotaxi.

## `obstacle_vehicle_tracks_v1.npz`

This catalog contains numeric vehicle trajectories used by the optional
live-edit obstacle ability. The archive stores relative timestamps, local
center translations, orientations, first-sample dimensions, object-type
codes, sample offsets, and initial heights for 668 car and truck tracks.

Runtime loading uses `allow_pickle=False`. The source scene used to derive the
catalog is not distributed with the package.

## Maps

Crazy Robotaxi's `.robotaxi.yaml` maps live in `crazy_robotaxi/maps/`. Seed
images referenced by a map may be map-relative files or packaged assets.
