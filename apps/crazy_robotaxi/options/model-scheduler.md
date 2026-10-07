# MODEL — SCHEDULER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **SCHEDULER** within **PIPELINE**, under **DIFFUSION MODEL**.

The shipped OmniDreams presets use a flow-matching scheduler.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Num Inference Steps:** | FlashDreams | — | Number of denoising steps; must equal len(denoising_timesteps). |
| **Denoising Timesteps:** | FlashDreams | — | Per-step diffusion timesteps in [0, num_train_timesteps]. |
| **Shift:** | FlashDreams | — | Schedule warp factor. |
| **Warp Denoising Step:** | FlashDreams | — | Map denoising_timesteps through the warped sigma schedule. |
| **Num Train Timesteps:** | FlashDreams | — | Length of the training sigma table. |
| **Sigma Max:** | FlashDreams | — | Top of the linspace before warping; 1.0 matches DiffSynth, upstream Wan / Lingbot ships 0.999. |
| **Sigma Min:** | FlashDreams | — | Bottom of the linspace before warping. Reserved for upstream parity; only 0.0 is exercised. |
| **Extra One Step:** | FlashDreams | — | If True, build the schedule from linspace(sigma_max, sigma_min, N+1)[:-1] (matches DiffSynth / upstream Wan); False uses N points and is kept for non-Wan recipes. |
| **Timestep Dtype:** | FlashDreams | — | Dtype of denoising_step_list. |
| **Enable Tqdm:** | FlashDreams | — | Whether to enable tqdm progress bar. |
