---
name: integrate-a-model
description: End-to-end workflow for porting an external video diffusion model into a flashdreams integration — scope the architecture, scaffold a workspace-member plugin, reuse an existing recipe, write the checkpoint key-remap, layer model-specific conditioners, wire the runner, and verify with checkpoint weight-equality + upstream parity + a GPU rollout. Use when integrating a new model (e.g. a HuggingFace/research release) into flashdreams or a downstream repo, porting upstream weights, or reproducing an existing integration. Pairs with the `flashdreams-integrations` skill (architecture map) — this skill is the ordered procedure; that one is the contract reference.
---

# Integrate a model into flashdreams

The ordered procedure for binding an external video model to the flashdreams
framework. Read the **`flashdreams-integrations`** skill first for the architecture
(layers, contracts, the cache tree) — this skill is the *route*, that one is the *map*.
(That skill's `name:` is `flashdreams-integrations` but it lives in the directory
`agentic/skills/flashdreams-recipes/` — mind the dir/name mismatch.)

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
exactly → **zero** remap, the transform is `lambda sd: sd`; the diffusers port needs
~25 rules). The native VAE needed only 4 rules vs the diffusers ~50. Note the native
checkpoint can be **either** a single-file `.pth` **or** sharded safetensors + a
`.safetensors.index.json` (the Wan native DiT is the latter, at the repo root; its VAE
is a nested `.pth`) — `load_checkpoint` resolves both. Fast pre-check before any
set-diff: do the key *counts* even match? (825 == 825 → you likely picked the right
source.)

**If you must remap (the diffusers port), the renames cluster into a few families.**
From the Wan diffusers→native mapping, expect: `attn1.*`→`self_attn.*`,
`attn2.*`→`cross_attn.*`, `to_q/to_k/to_v`→`q/k/v`, `to_out.0`→`o`,
`condition_embedder.{text,time}_embedder.linear_{1,2}`→`{text,time}_embedding.{0,2}`,
`condition_embedder.time_proj`→`time_projection.1`, `ffn.net.0.proj`/`ffn.net.2`→
`ffn.0`/`ffn.2`, `norm2`→`norm3`, `scale_shift_table`→`modulation` (per-block) /
`head.modulation` (top), `proj_out`→`head.head`. Write them as ordered regex rules and
let unmatched keys fall through (they show up as `unexpected_keys`, which the bijection
check below catches).

**Verify the remap is a key/shape bijection on CPU — no GPU needed.** This is the
single most valuable check. Build the model on `meta` and diff against the checkpoint
keys; any model key the transform doesn't supply stays on `meta` and `.to(device)`
later raises "Cannot copy out of meta tensor". Read names+shapes from the checkpoint
*without* loading weights via `safetensors.safe_open` (per-shard, walking the
`.index.json` `weight_map`):

```python
import json, glob, torch
from safetensors import safe_open

with torch.device("meta"):
    net = MyNetworkConfig().setup()
model = {k: tuple(v.shape) for k, v in net.state_dict().items()}

ckpt = {}                                                   # name -> shape, no weights
index = json.load(open(f"{ckpt_dir}/diffusion_pytorch_model.safetensors.index.json"))
for shard in set(index["weight_map"].values()):
    with safe_open(f"{ckpt_dir}/{shard}", framework="pt") as f:
        for k in f.keys():
            ckpt[k] = tuple(f.get_slice(k).get_shape())
ckpt = {k: ckpt[k] for k in my_state_dict_transform_keys(ckpt)}  # apply your rename

missing  = set(model) - set(ckpt)        # would stay on meta — must be empty
extra    = set(ckpt) - set(model)         # unexpected keys — must be empty
shapemm  = [k for k in model if k in ckpt and model[k] != ckpt[k]]
assert not missing and not extra and not shapemm, (missing, extra, shapemm)
```

Codify it as a `ci_cpu` test (`test_*_remap_is_full_bijection`) + spot-checks against
real key strings (`test_*_remap_spot_checks_real_keys`). (For a single-file `.pth`,
`torch.load(..., map_location="meta")` gives the same name→shape dict.)

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

To test the skill, point a fresh agent (no prior context) at the repo state **before**
an integration landed — a branch that **removes the integration plugins but keeps this
skill and the core network/recipe scaffolding** (e.g. `git rm -r integrations/wan22
integrations/hy_worldplay` off a branch that already has this skill). Have it reproduce
the integration following this skill; score against the merged result (the integration
PR + its follow-ups) — key set / shapes, parity `|Δ|`, test coverage, and how many
gotchas it hits unaided. Feed the gaps back into this file.

Eval-harness must-haves (learned the hard way):
- The eval branch / worktree must actually contain **both** this skill **and** the
  target config (`WanDiTNetworkTI2V5BConfig` etc.). Confirm with `ls` before launching —
  a stale worktree off the wrong base wastes the run.
- Give the agent a **torch-capable interpreter path** + `PYTHONPATH` (CPU is enough for
  the remap/bijection slice) and tell it not to read git history or the removed
  reference integration (no peeking at the answer).
- Scope the first run to the highest-signal, GPU-free slice — the **checkpoint remap +
  bijection** (Phase 3) — before attempting the full conditioner/runner port.

First run (Wan 2.2 DiT remap slice): a fresh agent correctly picked the native
checkpoint, found the zero-remap identity, and verified the 825↔825 bijection in
~20 min. Gaps it surfaced (now folded in above): the bijection snippet was pseudocode
(made runnable w/ `safetensors`), the native-checkpoint framing over-assumed `.pth`
(now notes sharded-safetensors), no diffusers-remap guidance (added the rename
families), and the `flashdreams-integrations` dir/name mismatch (now flagged).
