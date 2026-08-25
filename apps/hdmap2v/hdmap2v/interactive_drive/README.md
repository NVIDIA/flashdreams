# HDMap2V driving support

This package contains model-neutral scene loading, vehicle simulation,
conditioning rasterization, presentation, and input handling used by the
`hdmap2v` and `interactive-drive` applications.

World-model construction is injected through application hooks. Model
manifests, checkpoints, scene download policy, and session implementations
belong in `integrations_v2/<model>/`; this package does not select a model.

Use the registered application entry points rather than importing an
integration directly:

```bash
uv run flashdreams-run-v2 hdmap2v --mode local-window -- --scene scene.usdz
uv run flashdreams-run-v2 interactive-drive --mode local-window
```
