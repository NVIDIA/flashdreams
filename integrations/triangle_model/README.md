# Triangle model

`TriangleModel` is a concrete implementation of the reusable
`triangle_app.TriangleApp` contract:

```python
class TriangleModel(TriangleApp):
    def create_runtime(self, config):
        return TriangleRuntime()
```

Run the concrete package:

```bash
uv run --package flashdreams-triangle-model \
  flashdreams-run triangle-model
```

Available modes are `local-window`, `webrtc`, `mp4`, and `null`. Press `R`,
`G`, or `B` to select the triangle color; Space cycles colors.
