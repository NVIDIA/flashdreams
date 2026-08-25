# Integrations v2

This directory contains one package per model. Every integration has the same
top-level shape:

```text
integrations_v2/<integration>/
├── config.py
├── impl/
├── apps/<slug>/
└── tests/
```

`config.py` is the only Python module at the integration root. Model code lives
under `impl/`; app-specific bindings live under `apps/<slug>/`; and all
integration-owned tests live under `tests/`. Packaging files such as
`pyproject.toml` and `README.md` remain beside those directories. A rare,
integration-wide operational script may also remain at the root when it is an
actual user or build entry point; all other source, data, and vendored trees go
under `impl/`.

Adapters read model settings only from the root config and do not own demos, UI
loops, sessions, or model-specific application subclasses.

Runnable applications live under [`apps`](../apps/README.md). For example, the
T2V adapters under `apps/t2v/` pass root-config defaults and hooks to the shared
implementation in [`apps/t2v`](../apps/t2v/README.md).
