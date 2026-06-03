---
name: integrate-a-model
description: End-to-end workflow for porting an external video diffusion model into a flashdreams integration — scope the architecture, scaffold a workspace-member plugin, reuse an existing recipe, write the checkpoint key-remap, layer model-specific conditioners, wire the runner, and verify with checkpoint weight-equality + upstream parity + a GPU rollout. Use when integrating a new model (e.g. a HuggingFace/research release) into flashdreams or a downstream repo, porting upstream weights, or reproducing an existing integration. Pairs with the `flashdreams-integrations` skill (architecture map) — this skill is the ordered procedure; that one is the contract reference.
---

# Integrate a model into flashdreams

The ordered procedure for binding an external video model to the flashdreams
framework. Read the **`flashdreams-integrations`** skill first for the architecture
(layers, contracts, the cache tree) — this skill is the *route*, that one is the *map*.

**Worked example throughout:** `integrations/hy_worldplay/` (HY-WorldPlay WAN-5B I2V),
which reuses the `integrations/wan22/` Wan 2.2 TI2V-5B recipe. It is the most complete
reference integration; read it side-by-side. Match `python-docstring-style`.

## The core bet: reuse, don't re-implement

Most modern video models are DiT-family. Before writing anything, find the closest
existing flashdreams recipe (`integrations/wan22`, `wan21`, `self_forcing`, …) and
**subclass it**. HY-WorldPlay is a Wan 2.2 TI2V-5B with three conditioner deltas — it
adds ~3 small subclasses, not a from-scratch network. If your model maps onto an
existing backbone, the job is *config + checkpoint remap + deltas + verify*, which is
days–weeks. If it needs a novel network/attention/inference loop, it is much longer —
say so up front.

---

## Phase 0 — Scope (½–2 days; do this before promising a timeline)

Answer these from the upstream repo + model card, and write the answers down:

1. **Backbone family.** Is it a Wan/DiT variant? Diffusion-transformer? → which existing
   recipe is the closest base. (Decisive for the estimate.)
2. **Checkpoints.** What does upstream publish — native `.pth`/safetensors, a diffusers
   port, sharded or single-file? Note the HF repo ids. (Drives the remap; see Phase 3.)
3. **Inference shape.** Steps (distilled? e.g. HY = 4-step Euler), scheduler, guidance,
   resolution, AR/streaming vs one-shot, KV cache.
4. **Conditioners / deltas.** What does it add beyond the base backbone (camera, action,
   memory, control)? Each is a subclass + (usually) extra checkpoint keys.
5. **Reference for parity.** Can you run upstream to get a ground-truth output to diff
   against? (You need this for Phase 6.)

Output: a one-paragraph scope note + the "closest base recipe" decision. If the answer to
(1) is "novel architecture", flag it — the rest of this playbook still applies but Phase
2/4 grow a lot.

## Phase 1 — Scaffold the plugin

Integrations ship as **uv workspace-member plugins** under `integrations/<name>/`
(mirror `integrations/self_forcing/` / `integrations/hy_worldplay/`):

```
integrations/<name>/
├── pyproject.toml          # name flashdreams-<name>, version matches flashdreams, dep on flashdreams
├── <name>/
│   ├── __init__.py
│   ├── config.py           # static pipeline + runner config literals
│   ├── runner.py           # RunnerConfig + Runner.run()
│   └── _*.py               # model-specific subclasses (encoder/transformer/network)
└── tests/
    ├── test_smoke.py       # ci_cpu: import + static-config assertions
    └── parity_check/       # GPU parity harness (gitignored heavy deps)
```

- The repo-root `integrations/*` glob auto-adds it to the uv workspace.
- Register the runner under the `flashdreams.runner_configs` entry point (so
  `flashdreams-run <slug>` finds it) — copy the `[project.entry-points]` block from a
  sibling.
- `version` must match `flashdreams._version.__version__`; the `sync-version`
  pre-commit hook enforces it (CI fails otherwise).

## Phase 2 — Recipe config (subclass the base, ship a static literal)

In `config.py`, `copy.deepcopy` the closest base pipeline and swap the pieces that
differ — encoder / transformer.network / scheduler — into model-specific subclasses.
Ship **one module-level literal** `PIPELINE_<NAME>` (no `build_*` factories) + a
`RUNNER_<NAME>` literal + a `<NAME>_CONFIGS` dict keyed by `name`. See
`hy_worldplay/config.py::_build_hy_worldplay_pipeline`.

- Subclass `Wan21TransformerConfig` / the network / encoder configs; copy field-by-field
  so a future base-class field addition surfaces loudly instead of silently dropping.
- Set the standard transformer knobs (`len_t`, `window_size_t`, `guidance_scale`,
  `stamp_image_latent`, …) — see `flashdreams-integrations` §"Standard transformer knobs".
- Distilled models: swap the scheduler (HY → 4-step `FlowMatchEulerDiscreteScheduler`).

## Phase 3 — Checkpoint loading + key remap (the highest-leverage phase)

Upstream weights almost never match flashdreams key names. You write a
`state_dict_transform` (regex rename) consumed by the transformer/VAE config.

**Prefer the native checkpoint over a diffusers port when both exist.** flashdreams'
networks are typically ported from the *native* model, so native keys often match
1:1 (HY-WorldPlay DiT: `Wan-AI/Wan2.2-TI2V-5B` native keys = `WanDiTNetwork` keys
exactly → **zero** remap; the diffusers port needs ~25 rules). The native VAE `.pth`
needed only 4 rules vs the diffusers ~50.

**Verify the remap is a key/shape bijection on CPU — no GPU needed.** This is the
single most valuable check. Build the model on `meta` and diff against the checkpoint
keys; any model key the transform doesn't supply stays on `meta` and `.to(device)`
later raises "Cannot copy out of meta tensor":

```python
import torch
with torch.device("meta"):
    net = MyNetworkConfig().setup()
model_keys = set(net.state_dict())
remapped = set(my_state_dict_transform(load_keys_only(ckpt)))   # names+shapes
assert not (model_keys - remapped), "would stay on meta"        # missing
assert not (remapped - model_keys), "unexpected keys"            # extra
# shapes: assert remapped[k].shape == model_state[k].shape for all k
```

Codify it as a `ci_cpu` test (`test_*_remap_is_full_bijection`) + spot-checks against
real key strings (`test_*_remap_spot_checks_real_keys`).

**Before flipping a default checkpoint source, prove weight-equality.** If you switch
the production config to a different checkpoint (e.g. native `.pth` instead of diffusers),
load *both*, apply each transform, and assert every tensor matches
(`max |Δ| == 0`). Identical weights ⇒ identical output, no decode smoke needed. This is
how the VAE/DiT defaults were flipped safely (`test_*_weights_identical`, marked
`manual` since it downloads checkpoints).

**Pitfall — "missing params" is usually a naming mismatch, not absent weights.** If a
load fails with missing keys, diff the *names* first; the weights are almost always
present under a different convention.

## Phase 4 — Model-specific conditioners / deltas

Each delta = a subclass + (usually) extra checkpoint keys. HY-WorldPlay adds action
AdaLN (`action_embedding`), PRoPE dual-branch camera attention (`o_prope`), and
reconstituted-context memory. Conventions that make these parity-safe:

- **Zero-init new residual heads** so the conditioner is a strict identity until trained
  weights load (`nn.init.zeros_(head.weight)`). The un-conditioned pipeline then matches
  the base model exactly.
- **Tolerate the extra zero-init keys when loading a base checkpoint** that lacks them.
  Override `load_state_dict` on the network to allow *exactly* those keys missing (keep
  it strict for everything else) — see
  `HyWorldPlayWanDiTNetwork.load_state_dict`. Without this, a base/un-distilled load
  raises `Missing key(s)`.
- Keep model deltas in the integration — never branch `core/` or `infra/`; expose a
  config slot or override hook instead.

## Phase 5 — Runner + CLI

`runner.py` ships a `RunnerConfig` subclass (I/O fields: image/prompt/output, ckpt
override, knobs) + a `Runner` whose `run()` drives `initialize_cache` → per-AR-step
`generate`/`finalize` → decode → write mp4. Mirror `hy_worldplay/runner.py`. Thread an
optional `--ckpt-path` through `derive_config` to swap the checkpoint + transform at
construction time. Add example-data download helpers if useful for demos.

## Phase 6 — Verify (CPU first, then GPU)

In order of cost:

1. **`ci_cpu` smoke** (`test_smoke.py`): imports, the static config is fully swapped,
   runner slug == pipeline name, entry point registered, remap bijection tests.
   Run: `uv run --extra dev pytest integrations/<name>/tests/test_smoke.py`.
2. **Checkpoint weight-equality** (Phase 3) — proves the load is correct without a GPU.
3. **GPU rollout smoke** — `flashdreams-run <slug> --ckpt-path <distilled> --num-chunk 1`
   produces a valid mp4. (Use `--ckpt-path`; a base/un-distilled run gives identity-only
   output. Keep `num_chunk` small to dodge OOM and short-rollout edge cases.)
4. **Upstream parity** — run upstream on the same input/seed, diff decoded frames,
   report **mean `|Δ|` / 255**. HY-WorldPlay's bar: `≤ 20/255` (landed at 15.65). The
   residual is bf16 FP noise; don't chase bit-exactness across two kernel stacks.

## Phase 7 — Perf + model card (the visible deliverable)

- Bench native vs upstream, **stack-matched** (both cuDNN SDPA + `torch.compile`), at the
  largest `num_chunk` the GPU allows, discarding warmup chunks. Scope = **DiT + VAE
  enc/dec**, per-stage medians post-warmup. Harnesses: `tests/parity_check/bench.sh`
  (matched) / `bench_batch.sh` (native-only sample loop).
- Author a model-card page mirroring `docs/source/models/lingbot_world.rst` (hero +
  gallery videos, perf table, methodology); register it in `docs/source/models/index.rst`.

## Gotchas (hard-won)

- **CI-pinned ruff is the source of truth** — `uvx ruff` defaults to a newer version that
  sorts imports differently and touches unrelated files. Use the pinned version
  (`uvx ruff@<pinned> …`; check `.pre-commit-config.yaml`).
- **`ty` needs the real deps** — a torch-less env can't catch signature/None errors; CI's
  `cpu` job (full deps) is the real type check. Fix diagnostics, don't `# ty: ignore` what
  is fixable; remove `ty: ignore` once unneeded (CI flags unused ones).
- **`uv sync`/`uv run` builds `block-sparse-attn`** (CUDA ext) → needs `CUDA_HOME`. On a
  GPU box, use a synced venv; on CPU, run modules with `PYTHONPATH` against a venv that
  already has torch.
- **`expandable_segments:True` breaks CUDA graphs** — scope it to non-graph legs only.
- **First AR chunk's `diffuse` time is cold `torch.compile` autotune**, not steady-state
  — that's why bench discards warmup chunks.
- **Diffusers single-file URLs may 404** if the repo is actually sharded — point at the
  `.safetensors.index.json`; `load_checkpoint` resolves shards from it.
- **Keep heavy/scratch out of git** — checkpoints, vendor trees, bench outputs,
  handoff notes (gitignore them).

## Done criteria

- [ ] `ci_cpu` smoke + remap-bijection tests pass.
- [ ] Checkpoint weight-equality proven (or remap bijection + a GPU decode smoke).
- [ ] GPU rollout produces a valid mp4.
- [ ] Upstream parity `mean |Δ|` under the agreed bar.
- [ ] Runner registered; `flashdreams-run <slug> --help` works.
- [ ] Perf numbers + model-card page (if in scope).
- [ ] lint/`ty` green under the CI-pinned tools.

## Evaluating this skill

To test the skill, point a fresh agent at the repo state **before** an integration
landed (e.g. a branch reverting `integrations/hy_worldplay` + `integrations/wan22`) and
have it reproduce the integration following this skill. Score against the merged result
(the integration PR + its follow-ups) — key set / shapes, parity `|Δ|`, test coverage,
and how many of the gotchas it hits unaided. Feed the gaps back into this file.
