<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Independent LiDAR geometry and projection: test specification

## Status, scope and handoff

This behavioral acceptance specification was prepared for the separate
implementation of `_lidar_geometry.py` and `_lidar_projection.py` while both files
were nonfunctional Apache-2.0 interfaces. The public integration now contains
independently supplied implementations and analytic tests; rendering was enabled
in commit `127566cd`.
The 83 case requirements below remain the acceptance contract. A requirement is
not a passing result: use the recorded tests and their stated CPU/GPU/manual scope.
The historical handoff constraints below still describe the required separation
from removed source; this status update does not certify independent authorship.

The specification author has seen the previous implementation. This document
is a requirements handoff, not an assertion that a clean-room process has already
been completed. The implementers should receive a separately prepared snapshot
containing this document, the new stubs, approved caller interfaces, synthetic
fixtures and independently authored tests. Exclude removed source, repository
history, source-provenance research, reference checkouts, prior implementation
comparisons and their generated numerical outputs. Deletion from the current
checkout does not remove source from Git history. Keep provenance and review
records outside that handoff. Record who supplied each requirement, fixture,
oracle and implementation contribution.

The acceptance oracle is analytic geometry and the stated interfaces. Do not
copy old routines, port their control flow, use them to generate golden outputs,
or require their intermediate arrays, traversal order, decomposition method,
allocation calls, convergence algorithm or kernel selection. The replacement
may choose independent algorithms and dependencies subject to normal review.
No private reference repository, checkpoint, recorded dataset, network access
or GPU is needed for the required unit cases.

The requirements below preserve caller-visible coordinates, shapes, range
limits, timing and occlusion behavior. Sampling counts/order and ambiguous
numeric edge cases are specified as sets/invariants, rather than historical
array equality. Additional validation rules are proposed acceptance requirements;
review them before implementing and record intentional compatibility differences.

## Interface inventory

The stubs are the source of truth for parameter names, positional/keyword
calling conventions and defaults. Public numerical entry points are:

| Surface | Inputs and outputs |
| --- | --- |
| `PointBudget(remaining).reserve(count)` | Shared point-capacity record; reservation changes remaining capacity or rejects overflow |
| `sample_points_along_line` | Two 3-vectors, density in metres, optional budget → NumPy `[N,3]` |
| `sample_points_in_cylinder` | Two 3-vectors, density, radius in metres, optional budget → `[N,3]` |
| `points_from_polylines` | Sequence of `[K,3]` vertex arrays, density/radius/dtype/budget → `[N,3]` |
| `points_from_polygons` | Sequence of planar `[K,3]` polygon boundaries, density/dtype/budget → `[N,3]` |
| `points_from_closed_polylines` | Sequence of closed edge paths, density/radius/dtype/budget → `[N,3]` |
| `minimap_points` | Layer name and its primitive sequence, density/radius/dtype/budget → `[N,3]` |
| `cuboid_polyline` | Positive length/width/height `[3]`, object-to-world pose `[4,4]` → an edge path `[K,3]` |
| `SpinningLidarModel` | Elevations `[R]`, firing offsets `[R]`, columns `C`, frequency and optional sensor/lookup settings |
| Model metadata properties | `n_rows`, `shape`, nominal leading-edge azimuths, elevations in degrees, sweep duration in microseconds |
| `ray_directions` | Model → NumPy `[R,C,3]` unit directions through nominal range-image cell centres |
| `nearest_rows` | Elevations `[N]` → integer row indices `[N]` on the query device |
| `sensor_angles` | Sensor-frame tensor `[N,3]` → elevation radians `[N]`, azimuth radians `[N]`, validity bool `[N]` |
| `relative_frame_times` | Sensor-frame tensor `[N,3]` → normalized firing time `[N]`, validity bool `[N]` |
| `interpolate_rigid` | Two sensor-to-world poses `[4,4]` and times `[N]` → rotations `[N,3,3]`, translations `[N,3]` |
| `apply_rigid_inverse` | Per-point rotations/translations and world points → sensor points `[N,3]` |
| `world_points_to_shutter_sensor_frame` | Model, world points, endpoint poses and accuracy options → `ShutterProjection` |
| `ShutterProjection` | `points_sensor [M,3]`, `relative_time [M]`, original `valid_indices [M]`, all on the input device |
| `points_to_range_map` | Sensor points, model, maximum range and optional intensity `[N]` → two FP32 tensors `[R,C]` |

Metadata-only construction in the current stubs does not validate calibration;
`prepared_lidar.load` continues to validate scene metadata. The future functional
model must validate its own direct-construction inputs as specified below.
Private helper names are outside the compatibility contract.

## Shared conventions and independent fixtures

Coordinates use metres and a right-handed forward/left/up sensor frame. A pose
maps sensor coordinates into world coordinates. Its translation is the sensor's
world position. Angles are radians unless an API explicitly says degrees.

A scan starts at the rear and progresses clockwise when viewed from above.
There are `C` equally spaced nominal firing columns. Column zero's leading edge
is +pi, and each next leading edge advances clockwise by one column. Nominal
range-image rays pass through cell centres. Firing offsets affect timing;
range-image columns and inverse rays use the nominal lattice. A positive ring
offset adds to that column's nominal physical azimuth: a point at a nominal
leading-edge angle plus the ring offset fires at that column's time. The first and last
firing columns have normalized times 0 and 1. Ring midpoint ties choose the lower
original row index. Empty range/intensity cells contain zero.

Prepare these new, synthetic fixtures with a fixed seed (17 for randomized
cases). Store generator version, seed, units, array shape/dtype and a digest.
Use no copied sensor calibration or recorded point cloud.

| Fixture | Definition and independent expected facts |
| --- | --- |
| F01 line | Segment `(0,0,0)`→`(1,0,0)`, density 0.1 m; endpoints and one-dimensional support are known |
| F02 cylinder | Segment `(0,0,0)`→`(0,0,2)`, radius 0.2 m; axial coordinate in `[0,2]`, radial distance bounded by 0.2 |
| F03 polygons | Unit square; right triangle `(0,0),(1,0),(0,1)`; L shape with vertices `(0,0),(2,0),(2,1),(1,1),(1,2),(0,2)`; each lies at z=0 |
| F04 box | Dimensions `(4,2,2)`, identity pose; corners are all combinations of x=±2, y=±1, z=±1; exactly twelve distinct geometric edges |
| F05 toy sensor | Three rings at `(0.25,0,-0.25)` rad, zero offsets, eight columns, 10 Hz; all expected angular bins can be enumerated independently |
| F06 offset sensor | F05 with offsets `(one column,0,-one column)`; nominal rays stay unchanged, firing times shift by the stated offset convention |
| F07 analytic poses | Identity; translation `(1,2,3)`; +90° yaw; and identity→+90° yaw with translation `(0,0,0)`→`(2,0,0)` |
| F08 return pairs | Several points on one known ray at 2, 5, 12, 20, 99.999, 100 and 100.001 m, carrying distinct intensities |
| F09 synthetic scene | Seven cameras, four RGB frames, 128 synthetic descending ring angles, 1800 columns, 30 FPS, explicit identity mounts, eight empty semantic layers |
| F10 sparse actors | Two independently defined boxes at explicit frames `[0,2]`, with a gap at frame 1 and distinguishable dimensions/poses |

For tilted geometry apply an independently constructed 90° rotation and integer
translation to F01–F04. For interpolated poses use hand-computed 0°, 45° and 90°
rotations and translations at times 0, 0.5 and 1. Never obtain expected values by
calling a candidate helper that the test is meant to verify.

### Numeric comparison policy

* Exact: shapes, dtype/device, integer indices, validity masks, empty-cell zeros,
  exception class and input immutability. Boundary tests name the accepted bin.
* CPU float64 analytic coordinates: absolute tolerance 1e-9 m/rad, relative 1e-9.
* CPU/GPU float32 coordinates within 100 m: absolute 1e-4 m, relative 1e-5;
  unit rays absolute 2e-6; scalar angles/time absolute 2e-6 away from boundaries.
* Moving-scene range comparisons: absolute 1e-3 m, relative 1e-5, with identical
  accepted return masks. Inspect boundary cases separately; aggregate MAE must
  not hide a missed or extra return.
* Test exactly at thresholds and with adjacent representable values, not only a
  loose tolerance around a cutoff. Do not widen a tolerance automatically after
  failure. Any exception needs a written cause and review.
* FP16/BF16 support is optional and must be advertised explicitly. If offered,
  run dedicated finite-value/device/error-budget checks; never silently use
  lower precision to satisfy the float32/64 contract.

## Geometry unit specification — CPU required

Run each applicable case with NumPy float32 and float64. Compare geometric sets,
coverage and bounds, not point order. Neither inputs nor poses may be mutated.

| ID | Stimulus | Required assertions / oracle |
| --- | --- | --- |
| G01 | F01 at origin and after rigid transformation | Finite `[N,3]`; endpoints represented; points lie on the segment; inverse transformation restores the same geometric set within tolerance |
| G02 | Zero length, 0.009 m and 0.01 m segments | No divide-by-zero/nonfinite values; sub-centimetre segments contribute no samples; at 0.01 m a nondegenerate segment is represented; empty shape is `[0,3]` |
| G03 | F01 at density 0.2, 0.1 and 0.05 | Coverage does not become coarser as density decreases; maximum uncovered axial gap at most twice density; no historical exact count required |
| G04 | F02 along each principal axis, oblique and reversed | Points inside the finite cylindrical envelope; both endpoint neighbourhoods represented; axis orientation does not change geometric coverage |
| G05 | Radius zero and positive radius | Zero radius reduces to segment support; positive radius provides off-axis samples; no out-of-envelope point |
| G06 | Empty primitive lists and mixed empty/nonempty arrays | Correct empty dtype/shape; empty records do not consume budget or affect nonempty geometry |
| G07 | Two disconnected polylines | Every point belongs near a supplied segment; no artificial segment connects separate records |
| G08 | Closed path with and without repeated first vertex | Closing edge represented once geometrically; no diagonal connection across the shape; duplicate endpoint does not create nonfinite points |
| G09 | F03 square and triangle | All samples lie in the polygon or on its boundary and plane; interior test locations have samples within twice density; triangle excludes x+y>1 beyond tolerance |
| G10 | Concave F03 L shape | The missing quadrant contains no samples outside boundary tolerance; both arms and their boundaries are represented |
| G11 | Reversed polygon winding, repeated closure and repeated consecutive vertex | Same covered geometric region; no fill outside the boundary; deterministic results for repeated identical input |
| G12 | Rotated/translated planar polygon | Samples remain in the transformed plane/region; use the analytic inverse pose for verification, not candidate projection helpers |
| G13 | Collinear and fewer-than-three-distinct-vertex polygon | No false two-dimensional interior, huge allocation or nonfinite output; an empty interior or boundary-only representation is acceptable and documented |
| G14 | Self-intersection or materially nonplanar polygon | Reject with ValueError, or document an explicitly supported fill rule and add independently defined cases before accepting support |
| G15 | F04 and its transformed version | All emitted vertices are box corners; consecutive nonzero edges change exactly one local coordinate; unordered edge set covers all twelve box edges with no face/body diagonal |
| G16 | All eight map-layer names | lanelines/road_boundaries/wait_lines/poles use line geometry; crosswalks/road_markings preserve polygon interiors; signs/lights preserve closed box edges; unsupported names reject clearly |
| G17 | NaN/Inf, wrong ranks/shapes, nonpositive density, negative radius, invalid box dimensions or nonrigid pose | ValueError before large allocations; no partial successful output or input mutation |
| G18 | Explicit dtype and mixed primitive inputs | Honor requested float32/64 output dtype; return a consistent `[N,3]` even when every primitive is empty |
| G19 | Repeated invocation and reordered independent primitives | Same-input determinism; reordered inputs preserve the output geometric multiset, allowing order changes |

The supported cylinder cross-section uses a density-spaced triangular lattice
clipped to the disk, retaining off-axis boundary samples for sub-pitch radii.
This avoids redundant concentric-ring sampling without changing the density,
radius or point limits.

G14 supports simple nonplanar terrain boundaries by ear clipping in the
best-fit plane, then filling each resulting triangle in its supplied 3D plane.
Canonical winding and a lexicographic starting vertex define the triangulation
for reversed or cyclically shifted boundaries. This preserves concavities and
terrain vertex heights; self-intersections still reject. All triangle outputs
count toward the original polygon's primitive limit and shared scene budget.

### Capacity and resource unit specification

The compatibility defaults are density 0.02 m, radius 0.05 m, 10,000,000 points
per primitive and 60,000,000 per scene. Tests lower limits to small values rather
than allocating those defaults. Budgets count returned points, including any
retained duplicate samples; they never count bytes or discarded candidates.
The replacement must document whether a failed multi-primitive call rolls back
previous successful reservations; tests verify that documented policy.

| ID | Stimulus | Required assertions / oracle |
| --- | --- | --- |
| B01 | Budget 10; reserve 0, then 4, then 6 | Remaining capacity 10,6,0; no geometry allocation required |
| B02 | Capacity overflow and negative/nonintegral reservation | Clear error; reservation itself is atomic; remaining never increases or becomes negative |
| B03 | Share a small budget across two primitives and two semantic layers | Sum of returned point counts equals capacity consumed; exact fit succeeds; next nonempty primitive rejects |
| B04 | Empty/degenerate primitive under exhausted budget | No capacity charge; no full-sized temporary array |
| B05 | Huge segment, huge polygon and adversarially small density | Refuse an over-limit request before memory grows in proportion to the requested output; isolate in a process with a small memory cap |
| B06 | 10,000-vertex small-radius planar boundary | Bounded completion and memory, finite points inside region; no quadratic-memory requirement imposed on implementers; record actual peak RSS |
| B07 | Polygon with a large bounding extent but narrow/concave valid area | Bound candidate workspace as well as accepted output; reject if the safe workspace policy cannot be met |
| B08 | Static geometry plus actors over repeated sweeps | Actor capacity includes resident static points; reusing a scene does not cumulatively consume new sweep budgets; failure leaves published scene state unchanged |

## Sensor and projection unit specification — CPU required

Use F05/F06 and a synthetic 128×1800 profile; do not copy a real sensor profile.
Run baseline and any optional sorted-ring configuration against the same
independent nearest-angle oracle.

| ID | Stimulus | Required assertions / oracle |
| --- | --- | --- |
| P01 | Valid direct model construction | Correct R/C shape, degrees conversion, leading-edge azimuths and 100,000 microsecond duration at 10 Hz; retained sensor name; input arrays unmodified |
| P02 | Empty/mismatched/nonfinite calibration, C<2, invalid frequency or option | ValueError; positive integer C, finite positive frequency; sorted lookup additionally requires distinct descending rings after the arithmetic precision used by that path |
| P03 | Nominal centre rays from F05 | `[R,C,3]`, finite unit norm, expected ring elevation and clockwise azimuth; forward near centre columns; inverse binning recovers every cell |
| P04 | F06 versus F05 ray grids | Nominal range-image rays unchanged by firing offsets; timing differences tested separately |
| P05 | Query on every ring, at midpoints and adjacent floating values | Minimum angular-distance row; exact ties choose lower original index; optional sorted path gives identical indices, including empty and >131072-element queries |
| P06 | Sensor points on +X, ±Y, -X and above/below ring span | Angles follow stated axes/units; validity uses the calibrated vertical span with 1e-6 rad boundary slack; zero-length direction is invalid with finite reported placeholders |
| P07 | Points at known nominal firing directions | Firing time in `[0,1]`; first/last nominal columns give 0/1; offsets move timing in the documented direction; range-image projection stays nominal |
| P08 | Rear seam using positive and negative signed zero in Y | Define both limiting sides explicitly: rear +0 goes to first column; rear -0 to last. No out-of-range index or collision between these two test returns |
| P09 | Every centre ray scaled to 12 m | Full range map equals 12 m; output FP32 on input device; intensity all zero when omitted; no lost cells, including azimuth ends |
| P10 | F08 occlusion with unique depths and tagged intensities | Nearest positive return supplies range and its associated intensity; permutation of uniquely nearest points does not change output |
| P11 | Equal-depth collision, different intensities | Range exact; intensity belongs to one nearest return; repeated same input deterministic. Publish a tie policy and add an exact test for it; never mix range from one depth with intensity from a farther depth |
| P12 | Empty input; zero range; returns at/above max_range | Empty/invalid cells exactly zero; positive range strictly below maximum is eligible; exact maximum and the next representable value are excluded |
| P13 | Elevation just inside/on/outside ring acceptance margin | Explicit bin/validity expectations; range-map boundary margin 0.1 degree; scalar angle-validity and range-map admission are tested as separate contracts |
| P14 | Wrong point/intensity shape or device, NaN/Inf values, invalid maximum range | Reject malformed inputs deterministically before scatter/output allocation; do not silently broadcast intensity or migrate devices |
| P15 | Interleaved valid/invalid points | Correct original-index correspondence and associated intensity; no accidental compacted-index interpretation |
| P16 | Noncontiguous float32/64 inputs | Same accepted bins and ranges as a contiguous independent copy; no input writes or dtype-dependent dropped cells away from boundaries |

## Rigid transform and moving-shutter unit specification

These tests constrain observable physical consistency without specifying a
quaternion library, numerical solver, iteration order or stopping algorithm.
The optional `quaternions`/`reuse_quaternions` arguments are compatibility hints;
all supported hints must preserve the declared mathematical result.

| ID | Stimulus | Required assertions / oracle |
| --- | --- | --- |
| T01 | F07 endpoints and times `[0,0.5,1]` | Endpoint poses exact within dtype tolerance; midpoint translation `(1,0,0)`; shortest-arc midpoint yaw +45°; rotations orthonormal with determinant +1 |
| T02 | Identical rotations, nearly identical rotations and opposite quaternion representations of one rotation | Finite interpolation; same physical orientation; no dependence on the sign representation of a quaternion |
| T03 | Rotation near 180°, empty times and batched time queries | No nonfinite values or reflection; time order preserved; correct empty shapes; declared shortest-arc/tie convention documented |
| T04 | Independently specified rotations/translations and world points | Inverse transform recovers known sensor coordinates; forward reconstruction gives original world points; correct translation direction |
| T05 | Invalid pose shape, reflection, nonfinite values or time outside `[0,1]` | ValueError; no silent repair, extrapolation or clamping unless an explicitly revised contract is approved |
| T06 | Stationary identity sweep with known valid directions | Surviving sensor coordinates equal input; times independently match firing columns; original indices correct |
| T07 | Stationary translated/rotated sweep | Output equals analytic static inverse pose; common world rigid transform applied to points and both poses leaves sensor output unchanged |
| T08 | Translation only; yaw only; forward/reverse combined motion | Output differs from instantaneous static projection where expected; every returned point transformed forward at its reported time reconstructs its original world point |
| T09 | Small F05 moving problem with selected fixed-point solutions | Enumerate candidate firing columns in an independent slow test oracle; accept a self-consistent column/pose solution within tolerance; document tie handling where more than one solution exists |
| T10 | Directions entering/leaving vertical field of view during motion | Correct retained original indices; no invented point or mismatched time; output reconstruction checked individually |
| T11 | Empty world cloud and fully out-of-view cloud | Correct empty result fields, index dtype/device and finite values; no division by empty-count error |
| T12 | Accuracy/iteration arguments including zero/negative/NaN | Validate invalid arguments. For valid bounded work, terminate within the requested iteration budget; report behavior if no self-consistent result is found rather than claiming convergence |
| T13 | `reuse_quaternions` off/on, supported precisions | Same original indices and masks; coordinates/time within shared tolerance; cached state never leaks across poses, sensors or scenes |
| T14 | `max_error` off/on and near plus very far distractors | Each mode meets its documented residual tolerance. For the per-point mode, adding/removing points incapable of returning must not alter retained near-point results |
| T15 | Endpoint changes between successive calls | No stale pose or firing schedule reuse; repeated first input reproduces its original result |

## Integration specification

Use pytest `ci_cpu` for metadata, analytic geometry and small scenes; `ci_gpu`
for genuine device tests; `manual` for model/checkpoint/real-data tests. Every
test has exactly one category. New functional tests belong next to the model,
not in generic app or core packages. Write independent fixtures in the new test
module; do not resurrect removed implementation-dependent tests.

| ID / category | Boundary and stimulus | Required assertions |
| --- | --- | --- |
| I01 / CPU | `prepared_lidar.load` with F09 and no numerical implementation | Metadata loads, ring/mount arrays and semantic schema intact; no ray generation, point sampling, checkpoint access or device allocation |
| I02 / CPU | Malformed F09 settings, coordinates, provenance, layers, rings, rigid mounts, actor IDs/frames/dimensions | Reject with ValueError; known file hash/path checks run before consuming malformed content; no model/GPU work |
| I03 / CPU | Load RGB-only v1 and v2 bundles in a fresh process with reference/training/tooling imports blocked | Seven RGB cameras and source clocks intact; v2 calibration metadata remains accessible; successful loading does not imply LiDAR rendering availability |
| I04 / CPU, current stub gate | Call every public operation, including empty input | NotImplementedError with actionable unavailable message; no arrays, dummy scans, mutations, fallback imports or hidden source loading |
| I05 / CPU, current stub gate | Renderer creation with baseline/optimized/cull/settle flags | Fail before mapping calls, geometry work, output allocation or upload; metadata-only operations remain usable |
| I06 / CPU, current stub gate | `make_live_lidar` on every rank role and with an existing cached renderer/codec | Same unsupported-feature failure before cached reuse, checkpoint access, rank assignment or collectives; no hanging rank |
| I07 / CPU, current stub gate | Backend prewarm/load with LiDAR requested, weights absent or resident | Fail before model/checkpoint loading; RGB mode with resident model remains usable. Invalid v1 scene retains its explicit missing-v2 error |
| I08 / CPU, current stub gate | Only one of geometry/projection becomes available | Preflight still rejects; do not advertise a half-implemented renderer |
| I09 / CPU, future functional | Empty F09 map/actors, valid synthetic profile | Renderer returns `[3,128,1800]` FP32 zeros; no implicit plane or fake returns |
| I10 / CPU, future functional | Add one independently placed primitive per map layer | Predicted hit cells have physically consistent range; units/axis conventions match rays; no dependency on dictionary traversal order |
| I11 / CPU, future functional | F10 actors at present/gap frames, then beyond scene end | Only actors present at selected source frame are included; use the held final source frame beyond scene end; no previous-frame ghost |
| I12 / CPU, future functional | Known static/actor rays at 2,5,20,99.999,100 m | Nearest-return occlusion occurs before training validity: a 2 m occluder prevents a 20 m return reappearing; 5 m valid, 100 m invalid; intensity control channel zero |
| I13 / CPU, future functional | Opening then subsequent sweeps at 30 RGB FPS/10 Hz | Opening instantaneous; later sweeps span their trailing three RGB frames; actor selection at endpoint; no look-ahead, off-by-one or mixed source-clock mapping |
| I14 / CPU, future functional | Select two different RGB anchor cameras with the same calibrated LiDAR mount | Derived world sensor pose unchanged; mount transforms applied once; changed rig origin does not move the physical sensor |
| I15 / CPU, future functional | Baseline/optimized/cull/settle flags with static, forward/reverse translation and yaw | Identical masks on analytic cases; range difference <=1e-3 m for moving cases; preserve near occluders and all points capable of valid returns |
| I16 / CPU, future functional | Scene reset, scene switch and repeat invocation | No old geometry/actors/poses/limits in new outputs; deterministic repeated scene; resource capacity does not accumulate across sweeps |
| I17 / CPU, future functional | Map/actor requests exceeding small configured limits | Reject before device upload; no partially published renderer/cache; previous valid scene remains usable or is explicitly invalidated |
| I18 / GPU, future functional | F05 and synthetic 128×1800 fixture on CUDA | CPU/GPU masks and indices exact away from defined boundary cases; ranges within tolerance; output remains on requested device; no host-only silent fallback |
| I19 / GPU, future functional | Repeated moving sweeps and alternating sensor profiles | No per-sweep memory growth after warmup; measure allocated/reserved peaks; no cross-profile caches; synchronize CUDA timings |
| I20a / CPU, future functional | Stand-in renderer and rank collectives | Each sweep rendered once; owner encodes/broadcasts in source order; compatible success/failure paths; unavailable feature fails before collective entry |
| I20b / GPU, future functional | Real distributed synthetic-scene smoke | Same ordered ownership/clock contract as I20a; no duplicate or missing sweep, deadlock or hidden per-rank fallback |
| I21 / CPU, future functional | Control tensor into approved codec adapter using a small fake codec | Correct `[1,3,T,128,1800]`, dtype, units, validity and chunk order; no proprietary geometry imports; distinguish renderer from tokenizer failures |
| I22 / manual, future functional | Approved separately supplied trained V1.2 codec, short static then moving synthetic schedule | Finite outputs, correct normalization/chunk clocks, independent cache reset; numerical/visual comparison on identical inputs/latents; do not use drifting joint generation as the geometry oracle |
| I23 / manual, future functional | Joint RGB/LiDAR run plus range/intensity/point-cloud inspection | Correct calibrated mounts, frame alignment, ordered continuity and occlusion; export labelled artifacts; report reconstruction/model errors separately from renderer errors |
| I24 / CPU | Source/package inspection | Both replacement files Apache-2.0; no removed numerical bodies, serialized/encoded copies, dynamic Git downloads, training imports or fallback implementation; package includes the intended source/spec notices |

## Execution, artifacts and completion gates

Current runnable coverage includes geometry, projection, prepared rendering,
metadata validation, runtime independence and preflight rejection. The cases
below are **specified requirements, not evidence of a passing run**. Use the
recorded case-to-test mapping and run results to assess coverage. Do not replace
failures with generic skips or xfails and call that functional acceptance.

Current test organization: `test_lidar_geometry.py` (G/B),
`test_lidar_projection.py` (P/T), `test_prepared_lidar.py` (I and supported-state
checks), and `test_lidar_projection_gpu.py` (device-specific cases). Include case
IDs in test names or parameter IDs and maintain a case-to-test mapping. Retain
input-validation, RGB compatibility, early-failure and reset coverage alongside
the functional assertions.

From the repository root with the integration's locked dependencies:

```bash
PYTHONPATH="$PWD/integrations_v2/cosmos_interactive:$PWD/flashdreams:$PWD/integrations_v2:$PWD/apps/interactive_drive:$PWD/integrations_v2/omnidreams/impl/ludus-renderer" \
python -m pytest -o addopts= --import-mode=importlib -q \
  integrations_v2/cosmos3_nano_sim_auto/tests/test_prepared_lidar.py \
  integrations_v2/cosmos3_nano_sim_auto/tests/test_prepared_scene.py -m ci_cpu
```

Every future result records commit, command, fixture/seed/hash, enabled options,
dtype, device and tolerances. GPU reports additionally record GPU/driver,
PyTorch/CUDA, cache status, warmup, synchronized median/p90 latency and peak
allocated/reserved memory. Store large arrays, videos and profiler traces outside
Git; retain compact case results and links. No speed target is inferred from
historical implementations or from the separate codec benchmark.

Reenable rendering only after: the independent authorship/handoff record is
reviewed; all required analytic and integration cases pass; optimizations satisfy
the same contract; package/import checks pass; and a representative GPU/manual
run is reviewed when GPU support is claimed. Report any untested hardware or
optional mode explicitly. Keep unrelated codec license/quality/performance
review separate from acceptance of these two new components.
