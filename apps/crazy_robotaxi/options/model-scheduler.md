# MODEL — SCHEDULER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **SCHEDULER** within **PIPELINE**, under **DIFFUSION MODEL**.

The shipped OmniDreams presets use a flow-matching scheduler.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Num Inference Steps:** | FlashDreams | — | Number of denoising steps; must match the number of **Denoising Timesteps**. |
| **Denoising Timesteps:** | FlashDreams | — | Ordered model timesteps used for those steps. |
| **Shift:** | FlashDreams | — | Warping factor for the noise schedule. |
| **Warp Denoising Step:** | FlashDreams | — | Applies that warped schedule to the listed timesteps. |
| **Num Train Timesteps:** | FlashDreams | — | Length of the training timestep scale. |
| **Sigma Max:** | FlashDreams | — | Upper end of the noise schedule before warping. |
| **Sigma Min:** | FlashDreams | — | Lower end of the noise schedule before warping. |
| **Extra One Step:** | FlashDreams | — | Builds the schedule using one extra sample, then drops the last. |
| **Timestep Dtype:** | FlashDreams | — | Numeric type passed to the model time embedding. |
| **Enable Tqdm:** | FlashDreams | — | Shows scheduler progress output. |
