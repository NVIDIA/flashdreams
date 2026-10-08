# Edit-SFT (Tier-2b) — style restyles; see PLAN.md

Teach the realtime distilled student to apply a **global visual style**
("arcade racing game world", "comic ink", ...) mid-stream on a plain prompt
swap, imitating offline JoyAI restyles. Training is teacher-forced flow
matching on edit-timestamped clips: history = the model's own source rollout,
targets after the swap chunk = the JoyAI-styled video encoded into the
model's latent space. The checkpoint format is the guidance-distillation
trainer's, so it deploys through the live-edit deploy hook (`TextEditLoRA`,
PR #431) unchanged. The training helpers (`_host.py`, `_lora.py`,
`_train_attn.py`) are vendored from the Clean Forcing training infra
(PR #398) so this directory is self-contained on `main`; consolidate when
that lands.

Run from the repo root, in order:

1. `.venv/bin/python integrations/omnidreams/edit_sft/generate_sources.py`
   — roll the source corpus (videos + per-chunk latents + manifest,
   `outputs/sources/`). *Done: 10 clips x 28 chunks.*
2. JoyAI-Video-Edit batch (separate venv, see the project notes) over the
   style instruction bank -> `outputs/style_pairs/<uuid>__<style>.mp4`.
   *Done: 62 pairs across 11 slugs.*
3. `PAIRS_DIR=integrations/omnidreams/edit_sft/outputs/style_pairs \
   JOBS=style_jobs.jsonl,audition2_jobs.jsonl MODE=style \
   .venv/bin/python integrations/omnidreams/edit_sft/filter_pairs.py`
   — style-mode VLM gate (edit applied + persistent + road-layout at an
   early and a late frame) -> `style_pairs/filter_report.json`. `passed`
   pairs train anywhere; `early_window_ok` pairs train only in the first
   ~4 s (before streaming layout drift). *Running.*
4. `.venv/bin/python integrations/omnidreams/edit_sft/precompute_style.py`
   — one shot, ~20 GB VRAM: style + clip prompt embeddings, first-frame
   latents, per-chunk HDMap latents, and per-chunk styled-target latents
   (pipeline Wan-VAE encoder, AR chunk schedule 5->2 / 8->2 latent frames,
   patchified). Resumable; rerun if a decode assert fires.
5. `STEPS=1000 .venv/bin/python integrations/omnidreams/edit_sft/train_style_sft.py`
   — the trainer; **~30 GB VRAM eager** (2B DiT + one per-episode KV cache
   + immediate per-term backwards). Per step: replay 3 source chunks (KV
   window) at base weights, plain prompt swap at `k ~ U[4, 20]`, then
   `PRE_CHUNKS=1` no-op chunk (source target, original prompt) +
   `SPAN=4` styled chunks (styled targets, style prompt), committing the
   teacher-forced latents into the KV as it proceeds. Checkpoints
   (`{"lora": {i: tensor}}`, loadable by `TextEditLoRA`) land in
   `outputs/lora_style_stepN.pt` every 100 steps.

Knobs: `STEPS`, `LR` (2e-4), `RANK` (64), `SEED`, `HOLDOUT` (2),
`NOOP_PROB` (0.1), `PRE_CHUNKS` (1), `SPAN` (4), `SWAP_MIN`/`SWAP_MAX`
(4/20), `EARLY_CHUNKS` (15), `REPLAY_CHUNKS` (3), `SKIP_STYLES`
(`cyberpunk_neon,watercolor`).

## Style filtering (`MODE=style`) and the heavy-style early-window policy

`filter_pairs.py MODE=style` adapts the VLM gate to global restyles, where
the object-mode `scene_preserved` axis is meaningless (everything changes
by design). It keeps `edit_applied` / `persistent` and adds a
style-agnostic **road-layout** check — the judge compares a source frame
against the styled frame and scores only WHERE the road, lane lines, and
vehicles sit (describe-then-score prompting; a bare cross-style JSON score
collapses to 0) — at an early (frame 60) and a late (frame 190) point:

- **`passed`** (`edit_applied`, `persistent`, and `layout_late` all >= 3):
  the restyle holds and the layout survives the whole clip — the pair
  trains at any swap position.
- **`early_window_ok`** (`edit_applied` and `layout_early` >= 3, but the
  late check fails): the observed heavy-style failure mode — streaming
  error accumulation progressively replaces the road layout — sets in
  only after a few seconds. Instead of dropping these pairs,
  `train_style_sft.py` *demotes* them: they train only within the first
  `EARLY_CHUNKS` (15) chunks (~4 s), the pre-drift window the filter
  certified. This keeps heavy styles (cartoon cel, anime, pixel art) in
  the training set without teaching layout drift.
- Everything else is dropped. `SKIP_STYLES` (default
  `cyberpunk_neon,watercolor` — scene re-imagined / washed out at the
  audition) are excluded regardless of their filter verdict.

## Eval plan (gate before any deploy claim)

Reuse the guidance-distillation eval protocol (its `eval_guidance.py`, not
yet upstreamed) on the 2 held-out clips x the filter-passing styles,
RNG-matched against a shared no-edit control rollout:

- `lora_plain`: LoRA gated to `[SWAP_AT, SWAP_AT + SPAN)` + plain swap to
  the style prompt (deployment semantics; `RANK` must match the ckpt).
- `base_plain`: base weights + plain swap (floor).
- Reference divergence: the JoyAI styled clip vs the source clip over the
  same window (the offline target this SFT imitates — there is no "guided
  teacher" arm for styles).

Gate: styled-swap divergence ratio
`sum(gap_lora) / sum(gap_styled_target) >= 0.8` over the window, averaged
across combos, with `pre_swap_max_gap ~ 0` (the LoRA is inactive before the
swap, so any pre-swap divergence is a bug). Then the two checks divergence
cannot prove: (a) the style-mode VLM judge (`filter_pairs.py MODE=style`)
on (control, lora rollout) — `edit_applied` and `layout` scores, catching
"diverged but not styled" and layout damage; (b) a human visual pass on
side-by-sides (project lesson: divergence alone proves nothing). For
early-window styles, evaluate with `SWAP_AT <= 10` so the window ends
inside the certified range.
