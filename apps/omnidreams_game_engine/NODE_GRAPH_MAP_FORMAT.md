# Node-Graph Map Format

Schema version 1 is the authoring format for standalone OmniDreams games. It
models the road network directly: structural places are nodes, public roads
are graph edges, and parking access uses special driveway relationships.

## Coordinate and topology model

Every node has an explicit map-space pose:

```yaml
pose: {x_m: 12, y_m: -4, rotation_deg: 30}
```

`x_m` and `y_m` use metres. `rotation_deg` rotates the node's own footprint
counterclockwise from map +x. It does not rotate, snap, or constrain incident
roads. A four-way intersection has graph degree four, and every incident road
retains its independently authored geometry.

```text
              road C (independent tangent)
                       /
 road A ───── [ rotated intersection ] ───── road B
                    /
              road D
```

The persisted `GameMapTopology` contains typed nodes, roads, direct links,
inline road attachments, and adjacency. The compiler separately derives a
directed lane graph for fare routing and future NPC traffic. Turn connectors
in that routing graph are not emitted into ClipGT map conditioning. The player
remains physics-controlled and may drive in any physically reachable lane or
surface.

## Document shape

A document contains exactly these root keys:

```yaml
schema_version: 1
id: example-map
name: Example Map
compiler: {}
profiles: {}
nodes: []
roads: []
links: []
road_attachments: []
spawns: []
```

Unknown or missing fields are errors. See either bundled Crazy Robotaxi map for
complete `compiler` and `profiles` blocks. A profile defines lane width,
roadside curb offset, ordered lane directions, speed, curbs, and divider paint.
Its paved surface width is all lane widths plus both curb offsets.

## Nodes

All nodes require `id`, `type`, `pose`, and `geometry`.

### Intersections

Intersection footprints auto-fit their incident road and driveway widths:

```yaml
- id: askew_junction
  type: intersection
  pose: {x_m: 0, y_m: 0, rotation_deg: 45}
  geometry: {}
```

By default, the surface is the union of a central paved core and one
road-width arm along each actual incident bearing. Consequently, adjacent road
edges form the intersection corners directly. An intersection may override
both auto-fit dimensions to use a larger rotated rectangular core; its
incident arms still retain their independently authored bearings.

```yaml
geometry: {width_m: 24, depth_m: 16}
```

Road paths still determine their own endpoint tangents. A road may therefore
enter a 45-degree rotated intersection at 63 degrees without any per-arm angle
override.

### Cul-de-sacs

A cul-de-sac terminates exactly one authored road. Its circular road surface
has a flat curb opening whose chord exactly matches the incident road width,
so the road surface meets it without a gap. The circle has no visible
centerline or lane divisions and derives a routing-only turnaround.

```yaml
- id: oak_court_end
  type: cul_de_sac
  pose: {x_m: 80, y_m: 20, rotation_deg: 0}
  geometry: {radius_m: 10}
```

### Parking lots

A parking lot requires an access profile and rectangular dimensions. Its pose
is the footprint center, and it may be served by multiple driveway nodes.

```yaml
- id: market_lot
  type: parking_lot
  profile: parking_access
  pose: {x_m: 30, y_m: -20, rotation_deg: 15}
  geometry: {width_m: 28, depth_m: 34}
```

The compiler derives an aisle and turnaround for each entrance. The full lot
surface becomes a green ClipGT roadnet mask; parking stall lines are not
generated.

### Driveways

A driveway is a special access node with an explicit opening width. Its access
profile supplies lanes, paint, curb policy, and speed.

```yaml
- id: market_west_driveway
  type: driveway
  profile: parking_access
  pose: {x_m: 17, y_m: -7, rotation_deg: 270}
  geometry: {width_m: 6.4}
```

Every driveway serves exactly one parking lot and has exactly one public-road
source: either an intersection direct link or an inline road attachment. It is
a topological connection point rather than a separately paved rectangle; the
access surface and lanes continue through its center without a gap.

## Roads and curves

An authored road is one topological edge between two intersections and/or
cul-de-sacs:

```yaml
- id: oak_street
  from: west_junction
  to: east_junction
  profile: neighborhood
```

With no `path`, its centerline is the straight segment between node poses. A
self-loop therefore requires a path.

A path is one or more map-space cubic Bézier spans. Each span starts at the
previous endpoint (the first starts at the `from` node), has exactly two
controls, and has an explicit endpoint. The final endpoint must equal the `to`
node pose.

```yaml
- id: river_road
  from: west_junction
  to: east_junction
  profile: neighborhood
  path:
    - control_points: [{x_m: 20, y_m: 0}, {x_m: 35, y_m: 12}]
      end: {x_m: 45, y_m: 15}
    - control_points: [{x_m: 55, y_m: 18}, {x_m: 70, y_m: 5}]
      end: {x_m: 80, y_m: 5}
```

Intermediate span anchors and controls are geometry only; they do not become
graph nodes. Endpoint tangents come from the Bézier controls and remain
independent from both endpoint node rotations.

## Parking access

Direct links are not authored roads. They generate boundary-to-boundary paved
driveway spans using the driveway node's explicit width.

An explicit intersection entrance uses two links:

```text
intersection ══ driveway node ══ parking-lot node
```

```yaml
links:
  - {id: market_public_access, a: market_junction, b: market_driveway}
  - {id: market_lot_access, a: market_driveway, b: market_lot}
road_attachments: []
```

An inline driveway attaches to one uninterrupted road edge:

```text
intersection A ───────── one authored road ───────── intersection B
                                ║
                         driveway ══ parking lot
```

The driveway pose is the curb-opening center. Its rotation points outward from
the road toward the lot. The attachment cuts the curb and adds access routing,
but does not split the road, change its through lanes, or create an
intersection.

```yaml
links:
  - {id: market_lot_access, a: market_driveway, b: market_lot}
road_attachments:
  - {driveway: market_driveway, road: oak_street}
```

## Spawns and visual variants

A spawn names an authored road lane and a distance along its directed
centerline. Lane indices follow the profile's authored `lanes` order.

```yaml
spawns:
  - id: taxi_start
    road: oak_street
    lane: 1
    distance_m: 5
    variants:
      default:
        image: seed.png
        prompt: A forward-facing taxi view in a quiet neighborhood at daylight.
```

Every spawn requires a `default` variant. Images may be map-relative paths or
`package://package/resource` references. The resolved geometry, compiler
implementation, seed images, and prompts participate in the compiled-map
cache key.

## Validation summary

The loader rejects unknown fields and references, duplicate identifiers,
unsupported endpoint node types, zero-length straight roads, malformed or
discontinuous final Bézier endpoints, cul-de-sacs with the wrong degree,
parking lots without driveways, driveways serving zero or multiple lots,
driveways with multiple public-road sources, and inline driveway poses that are
not on the selected curb or do not face outward.
