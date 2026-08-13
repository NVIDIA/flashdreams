# Triangle application

`triangle_app` is a reusable FlashDreams application contract. It owns:

- triangle geometry and color input schemas;
- keyboard-to-color behavior;
- model input preparation;
- model discovery;
- MP4, null, WebRTC, and local-window orchestration.

It does not provide inference. A model integration subclasses `TriangleApp`,
implements `create_runtime()`, and registers itself with the application.

```bash
uv run --package flashdreams-triangle-model \
  flashdreams-run triangle-app --model triangle-model
```

See `integrations/triangle_model` for the reference model.
