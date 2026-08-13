# Triangle application

`TriangleApp` is a reusable FlashDreams application prototype. It owns the
triangle scenario, keyboard-to-color behavior, model input preparation, and
supported input/output modes.

Concrete model packages subclass `TriangleApp`, implement `create_runtime()`,
and expose a `create_app(args)` entry point. FlashDreams owns package discovery,
output backend creation, and session driving.

See `integrations/triangle_model` for the reference implementation.
