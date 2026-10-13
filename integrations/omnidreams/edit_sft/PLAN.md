# Instruction-SFT for grounded edits (Tier-2b of the live-edit hack)

**Goal:** push mid-stream editing beyond what the base model's priors give us
training-free. Three targets, ranked by demo value / feasibility:

1. **Object-by-prompt** — "a red sports car parked ahead", "boxes on the road":
   prompt-driven object addition that actually materializes (today only weather/
   lighting/atmosphere edits are reliable).
2. **Grounded props** — "Other"-class bboxes (cones, barriers) render as visible
   obstacles (today they under-render even oversized and grounded).
3. **Trajectory-locked moving actors** — a commanded constant-gap lead vehicle stays a
   lead vehicle (today the residential scene prior overrides commanded box motion and
   paints an oncoming pass instead). Hardest; data-dependent; may slip.

## Data recipes

- **A. JoyAI edit pairs (targets 1, partially 2).** Source videos = plain rollouts of
  our own model over the 32 local HF sample clips (`generate_sources.py`). Push each
  through JoyAI-Video-Edit (Apache-2.0, 30 FPS streaming editor, released weights)
  with a driving-edit instruction bank (add parked vehicles / obstacles / cones,
  appearance changes) → temporally aligned (source, instruction, edited) triplets.
  Filter: VLM edit-correctness + preservation + temporal consistency (JoyAI's own
  criteria); recaption with the achieved edit.
- **B. Cone/prop self-training (target 2).** Our model + construction-zone prompt
  assist demonstrably paints cones. Generate rollouts with oversized "Other" boxes +
  prompt assist; VLM-filter frames where a prop actually appears near the box
  projection; the surviving clips are (HDMap-with-box → video-with-prop) pairs that
  teach the box→prop mapping *without* the prompt crutch.
- **C. Teacher-regenerated continuations (targets 1, 3; optional).** The verified
  bidirectional teacher (HF `single_view/teacher/...`, same latent space) regenerates
  post-edit continuations at 35 steps + CFG under edit prompts / modified HDMaps —
  higher-compliance targets than the student can produce for itself. Use where JoyAI's
  weak spot (local add) shows.

## Training recipe

Teacher-forced flow-matching SFT on **edit-timestamped clips** (prompt A for chunks
[0, k), prompt/HDMap B after), LoRA on the same 8-projection target set as the
guidance-distillation trainer (its infra — functional attention, per-block
checkpointing, premerge deploy — is vendored into this directory). Two phases:

1. **Context build:** replay chunks [0, k) from the SOURCE latents through
   `finalize_kv_cache` (the vendored `_host.py:replay_history` machinery from the
   Clean Forcing training infra, PR #398) — history is real source content.
2. **Edit supervision:** for chunks [k, k+m), flow-matching MSE on the EDITED latents
   under the new conditioning, per-chunk independent timesteps; commit edited latents
   into the KV as teacher forcing proceeds.

Composes with the Tier-2a LoRA (train on top of, or merge and re-gate — decide after
first results). Eval: the InterBench-style VLM judge (Trigger/Align/Consistency) on a
held-out instruction bank + the existing divergence harness.

## Milestones

- M1: source corpus generated; JoyAI running on one clip end-to-end.
- M2: ≥500 filtered pairs across ~10 instruction types; recipe B pilot (~200 clips).
- M3: SFT v1 trained; object-by-prompt trigger rate measured vs base.
- M4: props from boxes without prompt assist; decision point on target 3.
