# Guidance self-distillation (Tier-2a of the live-edit hack)

**Goal:** bake the two-prompt text-edit guidance (`TextEditGuidance`, s≈3) into a LoRA so
a *plain* mid-stream prompt swap responds like a *guided* one — recovering the ~2x edit
strength at **zero inference cost** (guidance doubles the DiT forwards while active).

**Why it should work:** the teacher and student are the same network; the target is the
network's own guided output on RNG-matched on-policy states. This is standard
CFG-distillation, except the "CFG" here is the old-prompt/new-prompt axis and it only
matters for a few chunks after a swap. No external data or models needed.

## Recipe (on-policy, mirrors `drift_correction/train_v2.py`)

Per training step:

1. **Sample** a clip (32 local HF samples, `drift_correction/build_pairs._sample_files`),
   a swap chunk `k ~ U[4, 20]`, and an edit prompt from the bank.
2. **Roll the student** (LoRA active, plain swap at `k`) with the KV cache to a random
   chunk `j >= k` — self-forcing-style on-policy states. History replay machinery:
   `drift_correction/_host.py` (`reset_history`, `replay_history`, bracket helpers).
3. **At chunk `j`, per denoise step** (timesteps 1000, 450):
   - Teacher flow = frozen base (LoRA scale 0) with the guidance combine
     (`kv_old`/`kv_new` loads + `flow_old + s*(flow_new - flow_old)`) — i.e. exactly
     `CosmosTransformer._predict_with_text_edit_guidance` on unwrapped weights.
   - Student flow = LoRA'd network, single branch, new-prompt KV only.
   - Loss = MSE(student, teacher) in v-space; optionally also the context forward
     (t=128) so committed history matches.
4. **Backprop** through the student's step only (history detached — the KV buffer write
   severs grads anyway; use `_train_attn.py` functional dual-branch attention +
   per-block `torch.utils.checkpoint`, both proven on this host).

**LoRA config:** start from the drift-corrector recipe — r16 on
`blocks.*.self_attn.{q,k,v,output}_proj` — and add `cross_attn.{q,k,v,output}_proj`
(the edit signal enters through cross-attn; likely where the capacity is needed).
`_lora.py:apply_lora` handles both via substring match.

**Prompt bank (v1):** the weather/lighting set from `scripts/sweep_text_edit.py`
(incl. scene-native snow/rain phrasings) + per-clip base prompts as "no-op edits"
(swap to the same prompt → teacher == plain flow → regularizes against drift).
Precompute all text embeddings once (`pipeline.precompute_embeddings` pattern) so the
14 GB text encoder is not resident during training.

## Deployment: gate the LoRA like the guidance countdown

Enable the LoRA **only for the N chunks after a swap** — the exact window
`TextEditGuidance.chunks_remaining` covers today — via the drift corrector's per-chunk
gating + premerge pattern (`_drift_corrector.py`; premerged weight swaps cost ~0 ms).
Outside the window the base weights run untouched, so non-edit behavior carries zero
regression risk by construction.

## Eval / kill gate

- Reuse `scripts/sweep_text_edit.py`: (LoRA + plain swap) vs (base + guided) divergence
  curves on held-out clips x prompts; eyeball grids.
- Pass: LoRA plain-swap reaches >=80% of guided divergence at matched chunks, with
  no MUSIQ drop on no-swap rollouts (drift eval harness `eval_rollouts.py`).
- Budget: ~1k steps eager w/ checkpointing; hours on the shared GB300 (fits the
  ~65 GB share; full card is comfortable).

## Open choices

- Distill a *fixed* s (3.0) vs conditioning on s (start fixed; the wrapper default
  becomes "swap = guided-strength swap").
- Whether to include ReCache in the teacher rollout (probably yes — it is on by
  default in serving).
- Later (Tier-2b): extend the same loop with object/appearance edit pairs from
  JoyAI-Video-Edit to push beyond what guidance alone can reach.
