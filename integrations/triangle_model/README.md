# Triangle model

The integration implements `TriangleModel(TriangleApp)`, its runtime/session,
and the package application factory:

```python
def create_app(args):
    return TriangleModel(...)
```

Run it directly through FlashDreams:

```bash
uv run --package flashdreams-triangle-model \
  flashdreams-run triangle-model
```

Modes are `local-window`, `webrtc`, `mp4`, and `null`. Press `R`, `G`, or `B`
to select a color; Space cycles colors.
