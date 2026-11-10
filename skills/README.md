# Agent Skills

FlashDreams maintains two opt-in skill collections:

| Location | Audience | Publication |
| --- | --- | --- |
| `skills/<name>/` | Developers using or extending FlashDreams | Candidates for individual NVIDIA catalog registration. |
| `contributor-skills/<name>/` | Contributors maintaining this repository | Repository-only; exclude these paths from catalog sync. |

Read each skill's `SKILL.md` frontmatter (`name` and `description`) to find the
workflow that matches your task, then load that skill's full instructions.

## Opt in locally

No client is assumed to load skills automatically. From the repository root,
choose the discovery directory your client supports. For product skills only:

```bash
# General agents (use .claude or .cursor in place of .agents as appropriate).
mkdir -p .agents
ln -s ../skills .agents/skills
```

For a selective mix, use a **real directory** at `.agents/skills` and individual
links instead:

```bash
mkdir -p .agents/skills
ln -s ../../skills/flashdreams-integrate-model .agents/skills/flashdreams-integrate-model
ln -s ../../contributor-skills/python-docstring-style .agents/skills/python-docstring-style
```

These are alternative setups. If `.agents/skills` is already a whole-directory
symlink, replace that local symlink with a real directory before adding individual
links; otherwise the links would be written into the canonical `skills/` tree.
Update any individual links to renamed or moved skills. `.agents/`, `.claude/`,
and `.cursor/` are gitignored, so discovery preferences remain local.

## Authoring and maintenance

Follow the [Agent Skills specification](https://agentskills.io/specification).
Keep a skill's directory name equal to its frontmatter `name`, use a precise
`description`, and keep `SKILL.md` concise (under 500 lines). Public skills use
`flashdreams-` names and the repository's Apache-2.0 license. Author metadata uses
the team contact recorded in `REUSE.toml`.

Include the essential workflow inside the skill. Product source paths refer to a
user's FlashDreams checkout, not the installed skill's parent directory. Public
skills must not depend on unregistered contributor skills. Link to current docs
for details instead of copying entire manuals or mutable model catalogs.

Refresh commands and contracts on product releases, rerun evaluations, and
re-sign changed published content.
Catalog registration must list each public skill separately; do not register the
repository root or `contributor-skills/`.

## Local checks

The repository checks names/descriptions for both collections and public dataset
integrity without importing models or downloading weights:

```bash
# Uses an isolated pytest tool environment instead of syncing the GPU workspace.
PYTHONPATH=flashdreams uv run --no-project --with pytest python -m pytest tests/test_agent_skills.py -q
```

With [SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) and its Tier 3
extra installed, run:

```bash
skillevaluator validate skills/ --external --tiers 1 --output-dir /tmp/flashdreams-skill-validation
for skill in skills/*/SKILL.md; do
  skillevaluator tier3 validate "${skill%/SKILL.md}" --strict
done
```

Tier 1 requires its scanner dependencies. Inspect detailed reports for incomplete
scans, including unresolved source or generated-output references; an incomplete
scan is not a pass. `tier3 validate` checks datasets without executing agents.
Neither command proves task completion or skill uplift.

## Behavioral evaluation and publication

Each public skill ships six cases in `evals/evals.json`: four positive tasks,
one neighboring-workflow negative, and one unrelated negative. These are
self-contained CPU smoke scenarios for command construction and code/design
review. They need no product checkout, checkpoint, GPU, or network inside the
agent task; model/provider access is still needed by the evaluation harness.
Keep prompts unforced so trigger selection can be measured. `expected_skill: null`
means this skill should not activate; another applicable skill may activate.

Run both agent harnesses with and without the skill in isolated workspaces using
the same model/settings and cases. With the Tier 3 extra, provider credentials,
and a working sandbox configured, for example:

```bash
skillevaluator tier3 evaluate skills/flashdreams-run-model \
  --agents codex,claude-code --env-mode docker \
  --results-dir /tmp/flashdreams-skill-evals
```

Repeat for all four skills and record the evaluator revision, product commit,
agent/model versions, trigger correctness, task/behavior results, token use and
elapsed time. Inspect the verdict: a completed evaluation command alone does not
mean it passed. The smoke suite does **not** prove end-to-end model integration,
GUI behavior, numerical parity, throughput or offline portability.

Before publication, add team-run evidence for a real model launch, a model/adapter
integration with CPU and GPU parity checks, a demo tested with a stub and real
adapter, and a complete offline bundle on a clean network-disabled target. Use
team GPU infrastructure/BYOE for work beyond the publishing CI's capacity.

Do not invent a passing `BENCHMARK.md`, skill card, or signature. Those artifacts
must reflect completed evaluation and the signing pipeline. These skills remain
publication candidates until evaluation, onboarding, signing and catalog review
are complete.
