# OmniDreams Game Engine

Reusable model-thread simulation, authored-map, physics, and conditioning
components for FlashDreams V2 applications. The engine does not own a runtime
loop, presentation backend, or worker thread.

`engine_settings.py` provides strict partial YAML overlays for reusable map,
rendering, presentation, wheel, world-model, and runtime settings. Applications
compose those settings with their own typed defaults and explicit CLI options;
relative paths resolve beside the YAML document.
