# FlashDreams T2V applications

The shared T2V package provides the application/session protocol; each model
integration owns its small `t2v/app.py` factory. A non-empty `--prompt` is
required.

For a targeted workspace environment, select the integration distribution in
the `uv run` command. This syncs the integration, `flashdreams-t2v`, and the
local-window/serving dependencies without installing unrelated integrations:

- `t2v-causal-forcing` → `--package flashdreams-causal-forcing`
- `t2v-cosmos-predict2` → `--package flashdreams-cosmos-predict2`
- `t2v-fastvideo-causal-wan22` → `--package flashdreams-fastvideo-causal-wan22`
- `t2v-self-forcing` → `--package flashdreams-self-forcing`
- `t2v-wan21` → `--package flashdreams-wan21`
- `ti2v-wan22` → `--package flashdreams-wan22`

The applications use one of two rollout modes:

- `t2v-cosmos-predict2`, `t2v-wan21`, and `ti2v-wan22` are
  **bidirectional**. They generate the complete clip in one rollout and require
  exactly one block (`--total-blocks 1`, which is the default).
- `t2v-causal-forcing`, `t2v-fastvideo-causal-wan22`, and
  `t2v-self-forcing` are causal, streaming applications that can generate
  multiple blocks.

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

Available slugs are `t2v-cosmos-predict2`, `t2v-causal-forcing`,
`t2v-fastvideo-causal-wan22`, `t2v-self-forcing`, `t2v-wan21`, and
`ti2v-wan22`. Wan 2.2 is first-frame conditioned and additionally requires
`--image-path`; see its [integration README](../../integrations/wan22/README.md).
All backends receive the same transport-neutral `InputHandler` and
`OutputSink` API. Input handlers publish named, time-windowed
`CanonicalInputWindow` values matching each application's
`CanonicalInputSchema`.
