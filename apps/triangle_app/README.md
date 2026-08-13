# Triangle application

`triangle_app` is a reusable FlashDreams application contract. It owns:

- triangle geometry and color input schemas;
- keyboard-to-color behavior;
- model input preparation;
- supported input and output modes.

It does not provide inference or register a runnable command. A model
integration subclasses `TriangleApp`, implements `create_runtime()`, and
registers the concrete package with `flashdreams-run`.

See `integrations/triangle_model` for the reference implementation.
