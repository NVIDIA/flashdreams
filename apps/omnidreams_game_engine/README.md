# OmniDreams Game Engine

This package owns the reusable scene, simulation, input, conditioning,
presentation, and legacy inference runtime used by standalone OmniDreams games.
It is intentionally independent of `omnidreams.interactive_drive` so game
development can diverge without changing the enterprise demo.

Games inject an application policy into `InteractiveDriveApp`; the historical
class name is retained temporarily to keep the proven runtime behavior stable.
