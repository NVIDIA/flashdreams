# GAME

[Options index](../OPTIONS.md)

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Gamepad Button Style:** | `game.gamepad_button_style` | Crazy Robotaxi | — | Labels shown for gamepad buttons: Xbox, PlayStation, or Nintendo Switch. It does not remap controls. |

## TAXI

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Seed:** | `game.taxi.seed` | Crazy Robotaxi | `--game-seed`, `--seed` | Seed for repeatable taxi gameplay; blank uses fresh randomness. This is independent of the model diffusion seed. |
| **High Scores Path:** | `game.taxi.high_scores_path` | Crazy Robotaxi | `--high-scores` | CSV file for the taxi leaderboard; blank uses the default high-score location. |

### RULES

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Waypoint Spacing M:** | `game.taxi.rules.waypoint_spacing_m` | Crazy Robotaxi | — | Distance between candidate points sampled along navigation routes. |
| **Pickup Grid Spacing M:** | `game.taxi.rules.pickup_grid_spacing_m` | Crazy Robotaxi | — | Spacing used to spread pickup locations across the map. |
| **Pickup Min Distance M:** | `game.taxi.rules.pickup_min_distance_m` | Crazy Robotaxi | — | Minimum straight-line distance from the taxi to a new pickup. |
| **Initial Pickup Max Distance M:** | `game.taxi.rules.initial_pickup_max_distance_m` | Crazy Robotaxi | — | Preferred maximum distance to the first, camera-visible pickup. |
| **Pickup Radius M:** | `game.taxi.rules.pickup_radius_m` | Crazy Robotaxi | — | Distance at which a passenger is collected. |
| **Dropoff Radius M:** | `game.taxi.rules.dropoff_radius_m` | Crazy Robotaxi | — | Distance at which a fare is completed. |
| **Fare Min Route Distance M:** | `game.taxi.rules.fare_min_route_distance_m` | Crazy Robotaxi | — | Preferred minimum route length from pickup to dropoff. |
| **Fare Max Route Distance M:** | `game.taxi.rules.fare_max_route_distance_m` | Crazy Robotaxi | — | Preferred maximum straight-line distance between fare endpoints. The minimum may not exceed this maximum. |
| **Target Speed Mps:** | `game.taxi.rules.target_speed_mps` | Crazy Robotaxi | — | Nominal speed used to calculate a fare's time limit. |
| **Grace S:** | `game.taxi.rules.grace_s` | Crazy Robotaxi | — | Extra time added to the distance-based fare limit. |
| **Min Time S:** | `game.taxi.rules.min_time_s` | Crazy Robotaxi | — | Lower bound for a fare's calculated time limit. |
| **Max Time S:** | `game.taxi.rules.max_time_s` | Crazy Robotaxi | — | Upper bound for a fare's calculated time limit; must be at least **Min Time S**. |
| **Trip Time Multiplier:** | `game.taxi.rules.trip_time_multiplier` | Crazy Robotaxi | — | Multiplies the fare limit after it is calculated and clamped. |
| **Base Fare Points:** | `game.taxi.rules.base_fare_points` | Crazy Robotaxi | — | Points awarded for a completed fare. |
| **Bonus Points Per Second:** | `game.taxi.rules.bonus_points_per_second` | Crazy Robotaxi | — | Additional points per whole second remaining on a completed fare. |
| **Event Banner S:** | `game.taxi.rules.event_banner_s` | Crazy Robotaxi | — | Duration of pickup, completion, and failure banners in simulation time. |
| **Global Time S:** | `game.taxi.rules.global_time_s` | Crazy Robotaxi | `--game-time-s` | Starting game clock; must be positive. |
| **Dropoff Time Bonus S:** | `game.taxi.rules.dropoff_time_bonus_s` | Crazy Robotaxi | — | Time added to the game clock after a successful dropoff. |
| **Ground Snap Max Absolute Rotation Deg:** | `game.taxi.rules.ground_snap_max_absolute_rotation_deg` | Crazy Robotaxi | — | Largest ground rotation accepted when aligning the taxi to the road surface. |
| **Ground Snap Settle Fraction:** | `game.taxi.rules.ground_snap_settle_fraction` | Crazy Robotaxi | — | Fraction of stale ground attitude removed after an invalid ground sample. |

### VEHICLE

These fields tune driving physics. Changing them can alter handling substantially.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Wheel Base M:** | `game.taxi.vehicle.wheel_base_m` | Game engine | — | Distance between front and rear axles in the vehicle model. |
| **Max Steer Rad:** | `game.taxi.vehicle.max_steer_rad` | Game engine | — | Maximum steering angle. |
| **Steer Rate Rad Per S:** | `game.taxi.vehicle.steer_rate_rad_per_s` | Game engine | — | Rate at which steering moves toward full lock. |
| **Steer Return Rate Rad Per S:** | `game.taxi.vehicle.steer_return_rate_rad_per_s` | Game engine | — | Rate at which steering recenters. |
| **Speed Limit Enabled:** | `game.taxi.vehicle.speed_limit_enabled` | Game engine | — | Applies map speed limits to the taxi. |
| **Max Speed Mps:** | `game.taxi.vehicle.max_speed_mps` | Game engine | — | Normal forward speed cap. |
| **Max Reverse Speed Mps:** | `game.taxi.vehicle.max_reverse_speed_mps` | Game engine | — | Reverse speed cap. |
| **Max Accel Mps2:** | `game.taxi.vehicle.max_accel_mps2` | Game engine | — | Forward acceleration cap. |
| **Reverse Accel Mps2:** | `game.taxi.vehicle.reverse_accel_mps2` | Crazy Robotaxi | — | Reverse acceleration cap. |
| **Max Brake Mps2:** | `game.taxi.vehicle.max_brake_mps2` | Game engine | — | Normal braking strength. |
| **Handbrake Decel Mps2:** | `game.taxi.vehicle.handbrake_decel_mps2` | Crazy Robotaxi | — | Deceleration while using the handbrake. |
| **Handbrake Yaw Gain:** | `game.taxi.vehicle.handbrake_yaw_gain` | Crazy Robotaxi | — | Additional rotation induced by the handbrake. |
| **Max Handbrake Yaw Rate Radps:** | `game.taxi.vehicle.max_handbrake_yaw_rate_radps` | Crazy Robotaxi | — | Cap on handbrake rotation speed. |
| **Max Lateral Accel Mps2:** | `game.taxi.vehicle.max_lateral_accel_mps2` | Game engine | — | Lateral acceleration ceiling for steering and body response. |
| **Drag Mps2:** | `game.taxi.vehicle.drag_mps2` | Game engine | — | Base speed loss from drag. |
| **Mass Kg:** | `game.taxi.vehicle.mass_kg` | Game engine | — | Vehicle mass used by the physical simulation. |
| **Tire Grip:** | `game.taxi.vehicle.tire_grip` | Game engine | — | Tire traction factor. |
| **Rolling Resistance:** | `game.taxi.vehicle.rolling_resistance` | Game engine | — | Resistance from rolling contact. |
| **Aero Drag Coefficient:** | `game.taxi.vehicle.aero_drag_coefficient` | Game engine | — | Aerodynamic drag factor. |
| **Collision Restitution:** | `game.taxi.vehicle.collision_restitution` | Game engine | — | General collision bounce. |
| **Collision Friction:** | `game.taxi.vehicle.collision_friction` | Game engine | — | Friction during collisions. |
| **Max Collision Yaw Rate Radps:** | `game.taxi.vehicle.max_collision_yaw_rate_radps` | Game engine | — | Cap on rotation caused by collisions. |
| **Suspension Stiffness:** | `game.taxi.vehicle.suspension_stiffness` | Game engine | — | Suspension spring strength. |
| **Suspension Damping:** | `game.taxi.vehicle.suspension_damping` | Game engine | — | How quickly suspension motion settles. |
| **Suspension Travel M:** | `game.taxi.vehicle.suspension_travel_m` | Game engine | — | Maximum suspension movement. |
| **Suspension Visual Gain:** | `game.taxi.vehicle.suspension_visual_gain` | Game engine | — | Amount of visible suspension response. |
| **Max Body Roll Rad:** | `game.taxi.vehicle.max_body_roll_rad` | Game engine | — | Limit on sideways body tilt. |
| **Max Body Pitch Rad:** | `game.taxi.vehicle.max_body_pitch_rad` | Game engine | — | Limit on forward/back body tilt. |
| **Actor Collision Enabled:** | `game.taxi.vehicle.actor_collision_enabled` | Game engine | — | Enables collisions with dynamic actors. |
| **Static Collision Enabled:** | `game.taxi.vehicle.static_collision_enabled` | Game engine | — | Enables collisions with static map barriers. |
| **Aabb Length M:** | `game.taxi.vehicle.aabb_length_m` | Game engine | — | Length of the taxi's axis-aligned collision box. |
| **Aabb Width M:** | `game.taxi.vehicle.aabb_width_m` | Game engine | — | Width of the taxi's axis-aligned collision box. |
| **Aabb Height M:** | `game.taxi.vehicle.aabb_height_m` | Game engine | — | Height of the taxi's axis-aligned collision box. |
| **Curb Collision Restitution:** | `game.taxi.vehicle.curb_collision_restitution` | Crazy Robotaxi | — | Bounce from curbs and other static barriers. |
| **Curb Forward Momentum Retention:** | `game.taxi.vehicle.curb_forward_momentum_retention` | Crazy Robotaxi | — | Minimum forward speed fraction retained after a glancing curb hit. |
| **Input Activation Threshold:** | `game.taxi.vehicle.input_activation_threshold` | Crazy Robotaxi | — | Smallest steering or pedal magnitude counted as active input. |
| **Direction Change Accel Multiplier:** | `game.taxi.vehicle.direction_change_accel_multiplier` | Crazy Robotaxi | — | Braking multiplier when switching travel direction. |
| **Speed Taper Knee Fraction:** | `game.taxi.vehicle.speed_taper_knee_fraction` | Crazy Robotaxi | — | Speed fraction where acceleration taper changes regime. |
| **Speed Taper Low Floor:** | `game.taxi.vehicle.speed_taper_low_floor` | Crazy Robotaxi | — | Minimum acceleration fraction below the speed-taper knee. |
| **Speed Taper High Floor:** | `game.taxi.vehicle.speed_taper_high_floor` | Crazy Robotaxi | — | Minimum acceleration fraction above the speed-taper knee. |
| **Speed Taper Exponent:** | `game.taxi.vehicle.speed_taper_exponent` | Crazy Robotaxi | — | How sharply acceleration falls above the knee. |
| **Manual Coast Decel Mps2:** | `game.taxi.vehicle.manual_coast_decel_mps2` | Crazy Robotaxi | — | Deceleration when neither pedal is pressed. |
| **Ragdoll Grip Rate:** | `game.taxi.vehicle.ragdoll_grip_rate` | Crazy Robotaxi | — | Lateral velocity damping during collision recovery. |
| **Ragdoll Yaw Response Rate:** | `game.taxi.vehicle.ragdoll_yaw_response_rate` | Crazy Robotaxi | — | Rotation response during collision recovery. |
| **Handbrake Yaw Response Rate:** | `game.taxi.vehicle.handbrake_yaw_response_rate` | Crazy Robotaxi | — | Rotation response while the handbrake is active. |
| **Handbrake Lateral Damping Rate:** | `game.taxi.vehicle.handbrake_lateral_damping_rate` | Crazy Robotaxi | — | Sideways velocity damping while the handbrake is active. |
| **Handbrake Lateral Accel Scale:** | `game.taxi.vehicle.handbrake_lateral_accel_scale` | Crazy Robotaxi | — | Body-roll acceleration scale with the handbrake. |

## RACE

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Times Path:** | `game.race.times_path` | Crazy Robotaxi | `--race-times` | File for race times; blank uses the default leaderboard location. |

## EFFECTS

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Visual Flare:** | `game.effects.visual_flare` | Crazy Robotaxi | `--visual-flare`, `--no-visual-flare` | Enables the game-directed visual flare effect. |
