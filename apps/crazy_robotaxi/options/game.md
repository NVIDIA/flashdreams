# GAME

[Options index](../OPTIONS.md)

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Gamepad Button Style:** | Crazy Robotaxi | — | Labels shown for gamepad buttons: Xbox, PlayStation, or Nintendo Switch. It does not remap controls. |

## TAXI

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Seed:** | Crazy Robotaxi | `--game-seed`, `--seed` | Seed for repeatable taxi gameplay; blank uses fresh randomness. This is independent of the model diffusion seed. |
| **High Scores Path:** | Crazy Robotaxi | `--high-scores` | CSV file for the taxi leaderboard; blank uses the default high-score location. |

### RULES

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Waypoint Spacing M:** | Crazy Robotaxi | — | Distance between candidate points sampled along navigation routes. |
| **Pickup Grid Spacing M:** | Crazy Robotaxi | — | Spacing used to spread pickup locations across the map. |
| **Pickup Min Distance M:** | Crazy Robotaxi | — | Minimum straight-line distance from the taxi to a new pickup. |
| **Initial Pickup Max Distance M:** | Crazy Robotaxi | — | Preferred maximum distance to the first, camera-visible pickup. |
| **Pickup Radius M:** | Crazy Robotaxi | — | Distance at which a passenger is collected. |
| **Dropoff Radius M:** | Crazy Robotaxi | — | Distance at which a fare is completed. |
| **Fare Min Route Distance M:** | Crazy Robotaxi | — | Preferred minimum route length from pickup to dropoff. |
| **Fare Max Route Distance M:** | Crazy Robotaxi | — | Preferred maximum straight-line distance between fare endpoints. The minimum may not exceed this maximum. |
| **Target Speed Mps:** | Crazy Robotaxi | — | Nominal speed used to calculate a fare's time limit. |
| **Grace S:** | Crazy Robotaxi | — | Extra time added to the distance-based fare limit. |
| **Min Time S:** | Crazy Robotaxi | — | Lower bound for a fare's calculated time limit. |
| **Max Time S:** | Crazy Robotaxi | — | Upper bound for a fare's calculated time limit; must be at least **Min Time S**. |
| **Trip Time Multiplier:** | Crazy Robotaxi | — | Multiplies the fare limit after it is calculated and clamped. |
| **Base Fare Points:** | Crazy Robotaxi | — | Points awarded for a completed fare. |
| **Bonus Points Per Second:** | Crazy Robotaxi | — | Additional points per whole second remaining on a completed fare. |
| **Event Banner S:** | Crazy Robotaxi | — | Duration of pickup, completion, and failure banners in simulation time. |
| **Global Time S:** | Crazy Robotaxi | `--game-time-s` | Starting game clock; must be positive. |
| **Dropoff Time Bonus S:** | Crazy Robotaxi | — | Time added to the game clock after a successful dropoff. |
| **Ground Snap Max Absolute Rotation Deg:** | Crazy Robotaxi | — | Largest ground rotation accepted when aligning the taxi to the road surface. |
| **Ground Snap Settle Fraction:** | Crazy Robotaxi | — | Fraction of stale ground attitude removed after an invalid ground sample. |

### VEHICLE

These fields tune driving physics. Changing them can alter handling substantially.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Wheel Base M:** | Game engine | — | Distance between front and rear axles in the vehicle model. |
| **Max Steer Rad:** | Game engine | — | Maximum steering angle. |
| **Steer Rate Rad Per S:** | Game engine | — | Rate at which steering moves toward full lock. |
| **Steer Return Rate Rad Per S:** | Game engine | — | Rate at which steering recenters. |
| **Speed Limit Enabled:** | Game engine | — | Applies map speed limits to the taxi. |
| **Max Speed Mps:** | Game engine | — | Normal forward speed cap. |
| **Max Reverse Speed Mps:** | Game engine | — | Reverse speed cap. |
| **Max Accel Mps2:** | Game engine | — | Forward acceleration cap. |
| **Reverse Accel Mps2:** | Crazy Robotaxi | — | Reverse acceleration cap. |
| **Max Brake Mps2:** | Game engine | — | Normal braking strength. |
| **Handbrake Decel Mps2:** | Crazy Robotaxi | — | Deceleration while using the handbrake. |
| **Handbrake Yaw Gain:** | Crazy Robotaxi | — | Additional rotation induced by the handbrake. |
| **Max Handbrake Yaw Rate Radps:** | Crazy Robotaxi | — | Cap on handbrake rotation speed. |
| **Max Lateral Accel Mps2:** | Game engine | — | Lateral acceleration ceiling for steering and body response. |
| **Drag Mps2:** | Game engine | — | Base speed loss from drag. |
| **Mass Kg:** | Game engine | — | Vehicle mass used by the physical simulation. |
| **Tire Grip:** | Game engine | — | Tire traction factor. |
| **Rolling Resistance:** | Game engine | — | Resistance from rolling contact. |
| **Aero Drag Coefficient:** | Game engine | — | Aerodynamic drag factor. |
| **Collision Restitution:** | Game engine | — | General collision bounce. |
| **Collision Friction:** | Game engine | — | Friction during collisions. |
| **Max Collision Yaw Rate Radps:** | Game engine | — | Cap on rotation caused by collisions. |
| **Suspension Stiffness:** | Game engine | — | Suspension spring strength. |
| **Suspension Damping:** | Game engine | — | How quickly suspension motion settles. |
| **Suspension Travel M:** | Game engine | — | Maximum suspension movement. |
| **Suspension Visual Gain:** | Game engine | — | Amount of visible suspension response. |
| **Max Body Roll Rad:** | Game engine | — | Limit on sideways body tilt. |
| **Max Body Pitch Rad:** | Game engine | — | Limit on forward/back body tilt. |
| **Actor Collision Enabled:** | Game engine | — | Enables collisions with dynamic actors. |
| **Static Collision Enabled:** | Game engine | — | Enables collisions with static map barriers. |
| **Aabb Length M:** | Game engine | — | Length of the taxi's axis-aligned collision box. |
| **Aabb Width M:** | Game engine | — | Width of the taxi's axis-aligned collision box. |
| **Aabb Height M:** | Game engine | — | Height of the taxi's axis-aligned collision box. |
| **Curb Collision Restitution:** | Crazy Robotaxi | — | Bounce from curbs and other static barriers. |
| **Curb Forward Momentum Retention:** | Crazy Robotaxi | — | Minimum forward speed fraction retained after a glancing curb hit. |
| **Input Activation Threshold:** | Crazy Robotaxi | — | Smallest steering or pedal magnitude counted as active input. |
| **Direction Change Accel Multiplier:** | Crazy Robotaxi | — | Braking multiplier when switching travel direction. |
| **Speed Taper Knee Fraction:** | Crazy Robotaxi | — | Speed fraction where acceleration taper changes regime. |
| **Speed Taper Low Floor:** | Crazy Robotaxi | — | Minimum acceleration fraction below the speed-taper knee. |
| **Speed Taper High Floor:** | Crazy Robotaxi | — | Minimum acceleration fraction above the speed-taper knee. |
| **Speed Taper Exponent:** | Crazy Robotaxi | — | How sharply acceleration falls above the knee. |
| **Manual Coast Decel Mps2:** | Crazy Robotaxi | — | Deceleration when neither pedal is pressed. |
| **Ragdoll Grip Rate:** | Crazy Robotaxi | — | Lateral velocity damping during collision recovery. |
| **Ragdoll Yaw Response Rate:** | Crazy Robotaxi | — | Rotation response during collision recovery. |
| **Handbrake Yaw Response Rate:** | Crazy Robotaxi | — | Rotation response while the handbrake is active. |
| **Handbrake Lateral Damping Rate:** | Crazy Robotaxi | — | Sideways velocity damping while the handbrake is active. |
| **Handbrake Lateral Accel Scale:** | Crazy Robotaxi | — | Body-roll acceleration scale with the handbrake. |

## RACE

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Times Path:** | Crazy Robotaxi | `--race-times` | File for race times; blank uses the default leaderboard location. |

## EFFECTS

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Visual Flare:** | Crazy Robotaxi | `--visual-flare`, `--no-visual-flare` | Enables the game-directed visual flare effect. |
