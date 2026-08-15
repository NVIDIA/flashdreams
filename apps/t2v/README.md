# FlashDreams T2V applications

The shared T2V package provides the application/session protocol; each model
integration owns its small `t2v/app.py` factory. A non-empty `--prompt` is
required.

For a targeted workspace environment, select the integration distribution in
the `uv run` command. This syncs the integration, `flashdreams-t2v`, and the
local-window/serving dependencies without installing unrelated integrations:

- `t2v-causal-forcing` → `--package flashdreams-causal-forcing`
- `t2v-cosmos-predict2` → `--package flashdreams-cosmos-predict2`
- `t2v-self-forcing` → `--package flashdreams-self-forcing`

Native SlangPy window (default):

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --prompt "A robot walking through a forest."
```

The same demo can be launched from Python with the application runner used by
`flashdreams-run t2v-causal-forcing`:

```python
from flashdreams.demo import run_application

run_application(
    "t2v-causal-forcing",
    ["--prompt", "A robot walking through a forest."],
)
```

WebRTC browser backend:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --output webrtc --host 0.0.0.0 --port 8080 \
  --prompt "A robot walking through a forest."
```

Then open `http://localhost:8080/request_session`.

MP4 artifact:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --output mp4 --output-path artifacts/output.mp4 \
  --prompt "A robot walking through a forest."
```

Available slugs are `t2v-cosmos-predict2`, `t2v-causal-forcing`, and
`t2v-self-forcing`. All backends receive the same transport-neutral
`InputHandler` and `OutputSink` API. Input handlers publish named,
time-windowed `CanonicalInputWindow` values matching each application's
`CanonicalInputSchema`.
