# Node-Graph Map Format

Schema version 1 is the authoring format for standalone OmniDreams games. It
models structural places as nodes, public roads as graph edges, and parking
access through driveway relationships.

## Document shape

```yaml
schema_version: 1
id: example-map
name: Example Map
compiler:
  sample_spacing_m: 2.0
  ground_margin_m: 20.0
  intersection_connector_samples: 8
profiles: {}
nodes: []
roads: []
parking_accesses: []
spawns: []
```

`profiles` is optional. All other root fields are required, and unknown root
fields are errors.

The compiler settings control road sampling, ground extent, and routing-only
turn-connector resolution. They do not configure the renderer archive.

## Attributes and profiles

Profiles are optional, partial sets of defaults. An element may provide any
applicable attribute directly at its top level, reference a profile, or do
both. A directly supplied value wins over the profile value. Profile fields
that do not apply to an element are ignored.

After combining direct values and profile defaults, every required attribute
must have a value or compilation fails. Identity, pose, topology, and road
geometry are not profile attributes.

Linear elements use these attributes:

```yaml
lane_width_m: 3.6
curb_offset_m: 0.6
lanes: [backward, forward]
speed_limit_mps: 13.4
curb: true
lane_marking: {style: SOLID_GROUP, color: YELLOW}
divider_markings:
  - {style: SOLID_GROUP, color: YELLOW}
```

There must be one divider marking for every adjacent lane pair. The paved
surface width is `lane_width_m * len(lanes) + 2 * curb_offset_m`.

Every element emits semantic road-boundary polylines around its surface,
excluding declared connections. Those boundaries are always included in HD-map
conditioning. `curb: true` also makes them physical collision barriers;
`curb: false` leaves them non-colliding.

For example, a road can inherit most values while overriding its width:

```yaml
- id: oak_street
  from: west_junction
  to: east_junction
  profile: neighborhood
  lane_width_m: 4.0
```

## Coordinates and topology

Every node except a parking lot has an explicit map-space pose:

```yaml
pose: {x_m: 12, y_m: -4, rotation_deg: 30}
```

`x_m` and `y_m` use metres. `rotation_deg` rotates the node's own footprint
counterclockwise from map +x. It does not rotate or snap incident roads.

The persisted `GameMapTopology` retains typed nodes, roads, parking accesses,
and adjacency. The compiler separately derives a
directed lane graph for routing. Routing-only turn connectors are not emitted
into ClipGT map conditioning.

Each node and edge owns its surface and curb geometry. Connected elements meet
at equal-width openings without overlapping. Unrelated elements may not have
positive-area overlap or share a boundary edge; isolated point tangency is
allowed. Roads, parking lots, and other surfaces therefore cannot be layered
over one another to repair topology.

## Nodes

All non-parking nodes require `id`, `type`, and `pose`. Their remaining required
attributes may be supplied directly or by profile.

### Intersections

An intersection requires `curb`:

```yaml
- id: askew_junction
  type: intersection
  pose: {x_m: 0, y_m: 0, rotation_deg: 45}
  curb: true
```

The compiler infers the intersection footprint from its incident roads and
access paths. Each opening uses that element's paved width and endpoint tangent.
Adjacent road-edge lines determine how far each arm must reach, so orthogonal
roads form a compact rectangular junction while acute approaches extend far
enough to meet without gaps. Intersection dimensions and arm lengths are not
authored. Road centerlines determine their endpoint tangents independently of
node rotation.

### Road joints

A road joint connects exactly two compatible authored roads without creating an
intersection. It requires `curve_length_m`:

```yaml
- id: diagonal_bend
  type: road_joint
  pose: {x_m: 40, y_m: 20, rotation_deg: 0}
  curve_length_m: 12
```

The compiler measures `curve_length_m` independently along each incident road,
trims those portions, and replaces them with one tangent-continuous cubic
Bézier. Each control point is one `curve_length_m` from its cut point along the
trimmed road's local tangent. For straight approaches, both controls coincide
with the joint pose. Curved `path` and `bezier` approaches are supported, and a
trim may cross multiple curve spans.

`curve_length_m: 0` keeps a sharp change in direction at the joint pose. A
positive-length curve does not necessarily pass through the pose; the pose is
the original sharp-corner vertex. Node rotation does not affect the join.

Both roads must resolve to compatible lane directions, widths, curb offsets,
markings, divider markings, and curb modes when oriented through the joint.
Their speed limits may differ. The joint emits conditioning-visible lanes and
markings, and each directed joint lane inherits its incoming road's speed.
Requested trims that consume an entire road or produce invalid or overlapping
geometry are errors.

### Cul-de-sacs

A cul-de-sac requires `curb` and `culdesac_radius_m` and must terminate exactly
one road:

```yaml
- id: oak_court_end
  type: cul_de_sac
  pose: {x_m: 80, y_m: 20, rotation_deg: 0}
  culdesac_radius_m: 10
  curb: true
```

Its circular surface has a flat opening matching the incident road width. The
circle has no visible centerline or lane divisions and derives a routing-only
turnaround.

### Parking lots

A parking lot is an absolute map-space polygon. It has no pose, profile, or
linear attributes:

```yaml
- id: market_lot
  type: parking_lot
  vertices:
    - {x_m: 10, y_m: -30}
    - {x_m: 10, y_m: -10}
    - {x_m: 18, y_m: -10}
    - {x_m: 26, y_m: -10}
    - {x_m: 40, y_m: -10}
    - {x_m: 40, y_m: -30}
```

Vertices must describe a simple clockwise polygon. Concave polygons are
supported; holes, self-intersections, duplicate vertices, and degenerate edges
are not. Multiple parking accesses may serve one lot. The lot has physical
curbs and semantic boundaries on every edge except selected access openings.
It has no inferred aisle or turnaround lanes. Its surface becomes a green
ClipGT roadnet mask; parking-stall lines are not generated.

### Driveways

A driveway is a degree-two road node with one parking access:

```yaml
- id: market_west_driveway
  type: driveway
  pose: {x_m: 17, y_m: -7, rotation_deg: 270}
```

Its two roads must have compatible cross-sections, markings, and curb modes.
The compiler infers a minimal through-road surface large enough to contain the
curb opening, preserves conditioning-visible through lanes, and adds hidden
turn connectors to the access. A driveway is not emitted as an intersection.
Its entrance width comes from the selected parking-lot polygon edge.

## Road geometry

An authored road is one topological edge between intersections, road joints,
driveways, and/or cul-de-sacs:

```yaml
- id: oak_street
  from: west_junction
  to: east_junction
  profile: neighborhood
```

It uses the linear attributes. Without `path` or `bezier`, its centerline is
the straight segment between node poses. A self-loop therefore requires one of
those fields.

For normal hand-authored maps, `path` is a list of map-space points the road
centerline passes through. The `from` node pose is the implicit first point and
the `to` node pose is the implicit final point. The compiler derives smooth
cubic spans through the authored points.

```yaml
- id: river_road
  from: west_junction
  to: east_junction
  profile: neighborhood
  path:
    - {x_m: 45, y_m: 15}
    - {x_m: 70, y_m: 5}
```

The resulting centerline is:

```text
west_junction pose -> (45, 15) -> (70, 5) -> east_junction pose
```

Intermediate path points are geometry only; they do not become graph nodes.

For imported or precision-authored geometry, `bezier` supplies exact cubic
Bézier spans. Each span starts at the previous endpoint and has exactly two
control points plus an endpoint:

```yaml
- id: imported_curve
  from: west_junction
  to: east_junction
  profile: neighborhood
  bezier:
    - control_points: [{x_m: 20, y_m: 0}, {x_m: 35, y_m: 12}]
      end: {x_m: 45, y_m: 15}
    - control_points: [{x_m: 55, y_m: 18}, {x_m: 70, y_m: 5}]
      end: {x_m: 80, y_m: 5}
```

An `end` closes its span and becomes the next span's implicit start. The final
`end` must match the `to` node pose within 0.05m. Control points pull the curve
toward themselves; the centerline does not generally pass through them.

A road may include both fields. Both must be valid, and `bezier` determines the
compiled geometry when present. This lets a generated or precision-authored
curve override a simpler editable `path` without conflating the two formats.

## Parking access

`parking_accesses` generate boundary-to-boundary access spans. They are not
authored roads:

```yaml
parking_accesses:
  - id: market_lot_access
    source: market_driveway
    parking_lot: market_lot
    opening_vertex: 3
```

`source` must be an intersection or driveway. `opening_vertex` is one-based and
selects the complete polygon edge from that vertex to the next, wrapping from
the last vertex to the first. Authors insert vertices around a narrower
opening. One edge may be selected only once.

The compiler infers a tangent cubic to the opening midpoint and validates that
the source is outside the lot on the edge's exterior side. The exact opening
width becomes two equal opposing lanes with no shoulder, virtual white
markings, physical curbs, and a 5.5m/s speed limit. Intersection sources include
the access as an inferred footprint arm. Access lanes end at the lot boundary;
parking lots contain no internal routing lanes.

## Spawns and visual variants

A spawn names an authored road lane and a distance along its directed
centerline. Lane indices follow the effective `lanes` order.

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
`package://package/resource` references. Resolved geometry, compiler code, seed
images, and prompts participate in the compiled-map cache key.

## Validation summary

Compilation rejects unknown fields and references, missing effective
attributes, duplicate element identifiers, invalid endpoint types, malformed
or discontinuous road paths, invalid node degrees, invalid driveway
relationships, invalid parking polygons or openings, overlapping or
edge-sharing unrelated surfaces, overlapping connected surfaces, mismatched
connection openings, and parking accesses placed on an opening's interior side.
