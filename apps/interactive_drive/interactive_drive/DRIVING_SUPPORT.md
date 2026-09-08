# Interactive Drive support

This package contains model-neutral scene loading, vehicle simulation,
conditioning rasterization, presentation, and input handling used by the
`interactive-drive` application.

World-model construction is injected as a backend factory. Pipeline configs,
checkpoints, scene download policy, and session implementations belong in
`integrations_v2/<model>/`; this package does not select a model.

Use the registered application entry points rather than importing an
integration directly:

```bash
uv run flashdreams-run-v2 interactive-drive-omnidreams --mode webrtc -- \
    --scene scene.usdz
```
