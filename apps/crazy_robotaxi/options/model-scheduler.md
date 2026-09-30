# MODEL — SCHEDULER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **SCHEDULER** within **PIPELINE**, under **DIFFUSION MODEL**.

The shipped OmniDreams presets use a flow-matching scheduler.

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Num Inference Steps:** | `model.pipeline.diffusion_model.scheduler.num_inference_steps` | FlashDreams | — | Number of denoising steps; must match the number of **Denoising Timesteps**. |
| **Denoising Timesteps:** | `model.pipeline.diffusion_model.scheduler.denoising_timesteps` | FlashDreams | — | Ordered model timesteps used for those steps. |
| **Shift:** | `model.pipeline.diffusion_model.scheduler.shift` | FlashDreams | — | Warping factor for the noise schedule. |
| **Warp Denoising Step:** | `model.pipeline.diffusion_model.scheduler.warp_denoising_step` | FlashDreams | — | Applies that warped schedule to the listed timesteps. |
| **Num Train Timesteps:** | `model.pipeline.diffusion_model.scheduler.num_train_timesteps` | FlashDreams | — | Length of the training timestep scale. |
| **Sigma Max:** | `model.pipeline.diffusion_model.scheduler.sigma_max` | FlashDreams | — | Upper end of the noise schedule before warping. |
| **Sigma Min:** | `model.pipeline.diffusion_model.scheduler.sigma_min` | FlashDreams | — | Lower end of the noise schedule before warping. |
| **Extra One Step:** | `model.pipeline.diffusion_model.scheduler.extra_one_step` | FlashDreams | — | Builds the schedule using one extra sample, then drops the last. |
| **Timestep Dtype:** | `model.pipeline.diffusion_model.scheduler.timestep_dtype` | FlashDreams | — | Numeric type passed to the model time embedding. |
| **Enable Tqdm:** | `model.pipeline.diffusion_model.scheduler.enable_tqdm` | FlashDreams | — | Shows scheduler progress output. |
