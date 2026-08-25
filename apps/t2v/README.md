# FlashDreams T2V application

`apps/t2v` owns the reusable text-to-video application, session, model loop,
SlangPy UI, and test helpers. Model integrations do not implement demos. Their
packages under `integrations_v2` only load a pipeline config and pass any
model-shape differences to `T2VIntegrationHooks`.

The installed adapter selects the model while every slug runs the same app:

- `t2v-causal-forcing`
- `t2v-cosmos-predict2`
- `t2v-fastvideo-causal-wan22`
- `t2v-self-forcing`
- `t2v-wan21`
- `ti2v-wan22`

For example:

```bash
uv sync --package flashdreams-causal-forcing
uv run flashdreams-run-v2 t2v-causal-forcing --mode webrtc -- \
  --prompt "A robot walking through a forest."
```

Use `--mode mp4 --output-path artifacts/output.mp4` with the same slug for file
output. Wan 2.2 TI2V additionally requires `--image-path`.

## Ownership boundary

- `apps/t2v`: application behavior, UI, sessions, transport-neutral model loop,
  shared argument handling, and reusable tests.
- `integrations_v2/<model>`: the model implementation, root `config.py`, runner
  configs, and narrowly scoped hooks used by nested app adapters.
- `integrations_v2/<model>/<app>/adapter.py`: bare-minimum binding from a model
  config to this shared application.

An integration adapter must not define an `IApplication`, `ISession`, model
loop, UI loop, or model-specific application subclass.

## Tests

```bash
uv run --no-sync pytest apps/t2v -m ci_cpu -v
```
