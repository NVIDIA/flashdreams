# Triangle model

The integration implements the checkpoint and inference hooks on
`TriangleModel(TriangleApp)`. `TriangleApp` supplies the runtime/session
template, and the integration supplies the package application factory:

```python
def create_app(args):
    return TriangleModel(...)
```

Run it directly through FlashDreams:

```bash
uv run --package flashdreams-triangle-model --extra demo \
  flashdreams-run triangle-model
```

Built-in IO handlers are `local-window`, `webrtc`, `mp4`, and `null`. Press
`R`, `G`, or `B` to select a color; Space cycles colors.
