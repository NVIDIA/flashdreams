# FlashDreams text-to-video demo

This app uses the shared `flashdreams.runtime.demo` replay/session lifecycle.
The selected integration owns its pipeline and checkpoint configuration; the
app owns only prompt input, output selection, and the thin runtime adapter.

Run a saved replay:

```bash
uv run python -m apps.t2v_demo.app replay --backend causal-forcing --output outputs/t2v.mp4
```

Serve streamed WebRTC output:

```bash
uv run python -m apps.t2v_demo.app webrtc --backend self-forcing
```

`--backend` accepts `causal-forcing`, `cosmos-predict2`, and `self-forcing`.
Use `--preset-id` to select an integration-specific T2V runner preset. The
browser UI accepts a prompt before opening a generation session, plays emitted
chunks as they arrive, and records the received WebRTC stream for download.
