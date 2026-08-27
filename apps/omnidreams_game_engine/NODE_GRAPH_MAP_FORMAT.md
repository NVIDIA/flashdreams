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
race_courses: []
traffic_count: 12
traffic: []
spawns: []
```

`profiles`, `race_courses`, `traffic_count`, and `traffic` are optional. All
other root fields are required, and unknown root fields are errors.

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
lane_marking: {style: SOLID_GROUP, color: YELLOW}
divider_markings:
  - {style: SOLID_GROUP, color: YELLOW}
```

There must be one divider marking for every adjacent lane pair. The paved
surface width is `lane_width_m * len(lanes) + 2 * curb_offset_m`.

Every element emits semantic road-boundary polylines around its surface,
excluding declared connections. Those boundaries are always included in HD-map
conditioning. `curb: true` also makes them physical collision barriers;
`curb: false` leaves them non-colliding. The `curb` attribute defaults to
`true` when neither the element nor its profile supplies it.

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
pose: {x_m: 12, y_m: -4}
```

`x_m` and `y_m` use metres. Connected road geometry determines every node's
approach directions and footprint orientation.

The persisted `GameMapTopology` retains typed nodes, roads, derived parking
accesses, and adjacency. The compiler separately derives a
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

An intersection connects at least three incident road arms and has no required
attributes beyond its identity and pose:

```yaml
- id: askew_junction
  type: intersection
  pose: {x_m: 0, y_m: 0}
  lane_transition_length_m: 20
```

Use a road joint for a degree-two connection and a cul-de-sac for a degree-one
road ending; one- and two-arm intersections are rejected. The compiler infers
the intersection footprint from its incident roads and access paths. Each
opening uses that element's paved width and endpoint tangent.
Adjacent road-edge lines determine how far each arm must reach, so orthogonal
roads form a compact rectangular junction while acute approaches extend far
enough to meet without gaps. Intersection dimensions and arm lengths are not
authored. Road centerlines determine their endpoint tangents independently of
node rotation.

`lane_transition_length_m` is optional and defaults to zero. For each pair of
opposing through-road arms, the compiler independently selects the cross-section
with the greater lane count (or wider lanes when the counts match) at the
intersection. A narrower arm then widens over this distance: its incoming lane
splits before the intersection and its outgoing local lanes merge after it.
Perpendicular through roads are paired separately, so a north-south lane-count
change does not add lanes to a matching east-west street. The taper is part of
the intersection surface and its lanes and markings are conditioning-visible.
Each approach retains its own `curb_offset_m` outside the changing lane
envelope, and its authored curb mode controls the physical curb along the taper.

Opposing arms are inferred from their endpoint tangents; authors do not label
through-road pairs. A positive transition length is required only when a pair
changes lane count or lane width. It is measured into the authored road from
the inferred intersection opening and must not consume the complete road arm.

### Road joints

A road joint connects exactly two compatible authored roads without creating an
intersection:

```yaml
- id: diagonal_bend
  type: road_joint
  pose: {x_m: 40, y_m: 20}
  lane_transition_length_m: 20
```

The compiler independently infers the shortest trim on each incident road from
the roads' endpoint tangents and paved widths. It replaces those portions with
one tangent-continuous cubic Bézier and traces the joint surface from the
resulting roadside boundaries. The outside boundary remains curved rather than
forming a straight miter between the two approaches. Curved `path` and `bezier`
approaches are supported.

To author a longer curve, place a `bezier` road between two road joints. The
joints provide the minimal tangent connections while the road owns the extended
curve geometry.

`lane_transition_length_m` is optional and defaults to zero. It permits the two
roads to differ in lane count, lane width, or both. The joint uses the dominant
cross-section at the curve, then tapers into each narrower incident road over
the authored distance. As at an intersection, an incoming narrow lane splits
toward the joint and outgoing local lanes merge into the narrow road. The taper
is measured along the incident road after its inferred joint trim, including
across curved `path` or `bezier` approaches.

When oriented through the joint, both roads must still have compatible
direction ordering and opposing dividers. Speed limits, outer markings, curb
offsets, and curb modes may differ. Each approach keeps its authored curb offset
outside the changing lane envelope throughout its taper; adding lanes never
widens or narrows that offset. A lane-count or lane-width change requires a
positive `lane_transition_length_m`; otherwise its default of zero preserves
the previous exact-width behavior. The joint and taper emit
conditioning-visible lanes and markings, and each directed joint lane inherits
its incoming road's speed. Inferred curve trims or lane transitions that consume
an entire road or produce invalid or overlapping geometry are errors.

### Cul-de-sacs

A cul-de-sac requires `culdesac_radius_m` and must terminate exactly one road:

```yaml
- id: oak_court_end
  type: cul_de_sac
  pose: {x_m: 80, y_m: 20}
  culdesac_radius_m: 10
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
  connected_to: market_west_driveway
  opening_vertex: 3
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
are not. `connected_to` must name an intersection or driveway node.
`opening_vertex` is one-based and selects the complete polygon edge from that
vertex to the next, wrapping from the final vertex to the first. Authors may
insert vertices around a narrower opening. The lot has physical curbs and
semantic boundaries on every edge except its selected access opening.
It has no inferred aisle or turnaround lanes. Its surface becomes a green
ClipGT roadnet mask; parking-stall lines are not generated.

### Driveways

A driveway is a degree-two road node with one parking access:

```yaml
- id: market_west_driveway
  type: driveway
  pose: {x_m: 17, y_m: -7}
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

Each authored road has one uniform cross-section. To change lane count or lane
width along a contiguous street, end one road and begin another at a road joint,
then set the joint's `lane_transition_length_m`. Intersections provide the same
transition independently for each inferred through-road pair.

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

## Inferred parking access

The parking lot's `connected_to` and `opening_vertex` fields generate a
boundary-to-boundary access span; authors do not declare a separate road or
top-level access object. The compiler derives its stable identifier from the
lot identifier.

The compiler infers a tangent cubic to the opening midpoint and validates that
the connected node is outside the lot on the edge's exterior side. The exact
opening width becomes two equal opposing lanes with no shoulder, virtual white
markings, physical curbs, and a 5.5m/s speed limit. Intersection connections
include the access as an inferred footprint arm. Access lanes end at the lot
boundary; parking lots contain no internal routing lanes.

## Race courses

The optional `race_courses` list defines one or more ordered courses using
globally unique node or road IDs:

```yaml
race_courses:
  - id: neighborhood-loop
    start: south_intersection
    checkpoints: [east_road, north_intersection, west_road]
    lap_count: 3
    checkpoint_markers: true
```

Each referenced node or road supplies geometry for a fixed cross-course gate.
The start line crosses its element near the exit toward the first checkpoint;
checkpoint lines cross their elements near the entrance approached from the
preceding course element. The player registers a gate by crossing that line,
not merely by entering the element. `start` and every checkpoint must be
distinct valid IDs, and checkpoints must be non-empty. A zero
`lap_count` defines a point-to-point race that ends at the final checkpoint. A
positive count defines a lap race: after reaching the final checkpoint, the
player must return to `start` to complete that lap and begin the next one.

`checkpoint_markers` is optional and defaults to `true`. Set it to `false` to
hide the course's camera-world gate overlays without changing race progression
or timing. The BEV map always displays the active gate as a thick red line.

## NPC traffic

The optional `traffic` list defines vehicles that continuously follow the
compiled public-road lane graph:

```yaml
traffic:
  - id: neighborhood_car
    nodes: [west_junction, central_intersection, east_junction]
    end_behavior: reverse
    vehicle_type: car
    speed_mps: 11
    start_distance_m: 20
```

`traffic_count` optionally fixes the final number of NPC vehicles. When it is
omitted, the compiler uses the authored `traffic` list unchanged. It must be a
nonnegative integer at least as large as the authored list; a smaller value is
a conflict and fails compilation. When it is larger, the compiler fills the
difference with deterministic default cars distributed across legal public-road
loops and cul-de-sac routes. Generated cars avoid playable spawns and unsafe
initial overlap. Compilation fails with the map's safe capacity when the
requested count cannot be placed.

The compiled fleet advances logically across the full public-road graph, but
only graph-nearby vehicles enter PhysX and HD-map conditioning. While the ego
is on a road, nearby starts at both endpoint nodes; while the ego is on a node,
it starts at that node. The neighborhood includes public roads attached to
those nodes, the nodes at the other ends of those roads, and every public road
attached to that expanded node set. It stops before adding otherwise-unreached
nodes at the far ends of that final road ring. Leaving mapped road/node
surfaces retains the last valid neighborhood. Invisible vehicles continue
following their routes, speed limits, and same-direction headway.

`nodes` requires at least two non-parking nodes. Consecutive nodes do not need
to be adjacent: the compiler selects the shortest routable sequence of public
roads and rejects routes that cannot be connected. Parking accesses and
parking-lot interiors are never considered. At each intersection, traffic uses
the rightmost lane for right turns, the leftmost lane for left turns, and
preserves its relative lane for straight travel. Lane-count changes are joined
with a smooth lateral transition.

`end_behavior: wrap` routes from the final node back to the first without
teleporting. `reverse` traverses the waypoint list in the opposite order while
the vehicle continues to drive forward; the resulting route must have a legal
turnaround. A cul-de-sac endpoint supplies one automatically.

`vehicle_type` is optional and defaults to `car`; accepted values are `car`,
`truck`, and `bus`. Their dimensions can be overridden with
`dimensions_lwh_m: [length, width, height]`. `speed_mps` is an optional cap on
road speed limits. `start_distance_m` offsets the initial position along the
compiled cyclic route and defaults to zero. Vehicles are physical, collidable,
and maintain simple same-lane headway; traffic signals and right-of-way are not
currently modeled.

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

Every spawn requires a `default` variant. `image` is optional; when omitted (or
set to `null`), the compiler generates a deterministic synthetic first-person
view by projecting the semantic map from that spawn through the runtime front
camera. This fallback shows aligned road surfaces, boundaries, curbs, and
markings, but does not synthesize scenery. Use it as a robust placeholder, not
as a photorealistic authoring result.

Authored images may be map-relative paths or `package://package/resource`
references. Resolved geometry, compiler and fallback-renderer code, seed
images, and prompts participate in the compiled-map cache key.

## Validation summary

Compilation rejects unknown fields and references, missing effective
attributes, duplicate element identifiers, invalid endpoint types, malformed
or discontinuous road paths, invalid node degrees, invalid driveway
relationships, invalid parking polygons or openings, overlapping or
edge-sharing unrelated surfaces, overlapping connected surfaces, mismatched
connection openings, incompatible lane-transition cross-sections, transitions
that consume a road arm, and parking accesses placed on an opening's interior
side.
