"""Runner for the LingBot-VA Robotwin I2AV integration.

Uses the native compiled LingBot-VA transformer wrapper for inference
with ``torch.compile`` acceleration.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from PIL import Image
from tqdm import tqdm

from flashdreams.infra.runner import Runner, RunnerConfig
from lingbot_va.action import LingbotVAActionProcessor
from lingbot_va.constants import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_INPUT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPT,
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ACTION_GUIDANCE_SCALE,
    ROBOTWIN_ACTION_INFERENCE_STEPS,
    ROBOTWIN_ACTION_PER_FRAME,
    ROBOTWIN_ACTION_SNR_SHIFT,
    ROBOTWIN_ATTENTION_WINDOW,
    ROBOTWIN_FRAME_CHUNK_SIZE,
    ROBOTWIN_GUIDANCE_SCALE,
    ROBOTWIN_HEIGHT,
    ROBOTWIN_LATENT_CHANNELS,
    ROBOTWIN_OBS_CAM_KEYS,
    ROBOTWIN_PATCH_SIZE,
    ROBOTWIN_SINK_SIZE,
    ROBOTWIN_SNR_SHIFT,
    ROBOTWIN_USED_ACTION_CHANNEL_IDS,
    ROBOTWIN_VIDEO_INFERENCE_STEPS,
    ROBOTWIN_WIDTH,
)
from lingbot_va.pipeline import LingbotVAInferencePipeline
from lingbot_va.transformer import LingbotVATransformerConfig
from lingbot_va.utils import data_seq_to_patch, get_mesh_id, resolve_prompt


# Stub flash_attn before any upstream imports
import importlib.util as _ilu
import types as _types

def _stub_module(name: str) -> None:
    if name in sys.modules:
        return
    mod = _types.ModuleType(name)
    mod.__spec__ = _ilu.spec_from_loader(name, loader=None)
    mod.flash_attn_func = None  # type: ignore[attr-defined]
    sys.modules[name] = mod

_stub_module("flash_attn")
_stub_module("flash_attn_interface")


@dataclass(kw_only=True)
class LingbotVARobotwinRunnerConfig(RunnerConfig):
    """User-facing config for the LingBot-VA Robotwin I2AV runner."""

    _target: type["LingbotVARobotwinRunner"] = field(
        default_factory=lambda: LingbotVARobotwinRunner
    )

    checkpoint_root: str | Path = DEFAULT_CHECKPOINT_ROOT
    input_image_dir: Path = DEFAULT_INPUT_IMAGE_DIR
    prompt: str | Path = DEFAULT_PROMPT
    num_chunks: int = 10
    save_video: bool = True
    save_actions: bool = True
    metadata_only: bool = False
    seed: int = 42
    enable_offload: bool = False
    compile_network: bool = True
    benchmark: bool = False

    pixel_height: int = ROBOTWIN_HEIGHT
    pixel_width: int = ROBOTWIN_WIDTH
    frame_chunk_size: int = ROBOTWIN_FRAME_CHUNK_SIZE
    action_dim: int = ROBOTWIN_ACTION_DIM
    action_per_frame: int = ROBOTWIN_ACTION_PER_FRAME
    attn_window: int = ROBOTWIN_ATTENTION_WINDOW
    sink_size: int = ROBOTWIN_SINK_SIZE
    guidance_scale: float = ROBOTWIN_GUIDANCE_SCALE
    action_guidance_scale: float = ROBOTWIN_ACTION_GUIDANCE_SCALE
    num_inference_steps: int = ROBOTWIN_VIDEO_INFERENCE_STEPS
    action_num_inference_steps: int = ROBOTWIN_ACTION_INFERENCE_STEPS
    snr_shift: float = ROBOTWIN_SNR_SHIFT
    action_snr_shift: float = ROBOTWIN_ACTION_SNR_SHIFT
    patch_size: tuple[int, int, int] = ROBOTWIN_PATCH_SIZE


class LingbotVARobotwinRunner(
    Runner[LingbotVARobotwinRunnerConfig, LingbotVAInferencePipeline]
):
    """Robotwin I2AV driver with full GPU inference."""

    config: LingbotVARobotwinRunnerConfig

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _required_image_paths(self) -> dict[str, Path]:
        return {
            key: self.config.input_image_dir / f"{key}.png"
            for key in ROBOTWIN_OBS_CAM_KEYS
        }

    def _validate_inputs(self) -> dict[str, Path]:
        image_paths = self._required_image_paths()
        missing = [p for p in image_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing camera PNGs: " + ", ".join(str(p) for p in missing)
            )
        return image_paths

    def _write_metadata_manifest(self) -> Path:
        cfg = self.config
        prompt = resolve_prompt(cfg.prompt)
        image_paths = self._validate_inputs()
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cfg.output_dir / "runtime.json"
        manifest = {
            "runner_name": cfg.runner_name,
            "checkpoint_root": str(cfg.checkpoint_root),
            "input_image_dir": str(cfg.input_image_dir),
            "image_paths": {k: str(v) for k, v in image_paths.items()},
            "prompt": prompt,
            "num_chunks": cfg.num_chunks,
            "seed": cfg.seed,
            "height": cfg.pixel_height,
            "width": cfg.pixel_width,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------

    def _load_models(self, device: torch.device, dtype: torch.dtype):
        """Load transformer, VAE(s), text encoder, tokenizer."""
        from lingbot_va._loaders import (
            WanVAEStreamingWrapper,
            load_text_encoder,
            load_tokenizer,
            load_vae,
        )

        cfg = self.config
        ckpt = str(cfg.checkpoint_root)
        enable_offload = cfg.enable_offload
        vae_device = "cpu" if enable_offload else device

        self._vae = load_vae(os.path.join(ckpt, "vae"), dtype, vae_device)
        self._streaming_vae = WanVAEStreamingWrapper(self._vae)

        self._streaming_vae_half = None
        vae_half = load_vae(os.path.join(ckpt, "vae"), dtype, vae_device)
        self._streaming_vae_half = WanVAEStreamingWrapper(vae_half)

        self._tokenizer = load_tokenizer(os.path.join(ckpt, "tokenizer"))
        self._text_encoder = load_text_encoder(
            os.path.join(ckpt, "text_encoder"), dtype,
            "cpu" if enable_offload else device,
        )

        lh = (cfg.pixel_height // 16 * 3) // 2
        lw = cfg.pixel_width // 16

        from lingbot_va.transformer.impl.network import WanVADiTNetworkConfig
        transformer_cfg = LingbotVATransformerConfig(
            network=WanVADiTNetworkConfig(
                action_dim=cfg.action_dim,
                action_per_frame=cfg.action_per_frame,
                patch_size=cfg.patch_size,
            ),
            checkpoint_root=ckpt,
            dtype=dtype,
            compile_network=cfg.compile_network,
            guidance_scale=cfg.guidance_scale,
            action_guidance_scale=cfg.action_guidance_scale,
            latent_height=lh,
            latent_width=lw,
            frame_chunk_size=cfg.frame_chunk_size,
            action_per_frame=cfg.action_per_frame,
            attn_window=cfg.attn_window,
        )
        self._transformer = transformer_cfg.setup()
        self._transformer.load_model(device)

    # ------------------------------------------------------------------
    # prompt encoding (from upstream _get_t5_prompt_embeds / encode_prompt)
    # ------------------------------------------------------------------

    def _encode_prompt(self, prompt: str, device: torch.device, dtype: torch.dtype):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        prompt = prompt_clean(prompt)
        text_inputs = self._tokenizer(
            [prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_len = mask.gt(0).sum(dim=1).long()

        enc_device = next(self._text_encoder.parameters()).device
        embeds = self._text_encoder(ids.to(enc_device), mask.to(enc_device)).last_hidden_state
        embeds = embeds.to(dtype=dtype, device=device)
        embeds = [u[:v] for u, v in zip(embeds, seq_len)]
        embeds = torch.stack([
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in embeds
        ], dim=0)
        return embeds.to(device)

    # ------------------------------------------------------------------
    # observation encoding (from upstream _encode_obs)
    # ------------------------------------------------------------------

    def _encode_obs(self, obs_images: dict[str, np.ndarray],
                    device: torch.device, dtype: torch.dtype):
        cfg = self.config
        videos = []
        for k_i, k in enumerate(ROBOTWIN_OBS_CAM_KEYS):
            if k_i == 0:
                h_i, w_i = cfg.pixel_height, cfg.pixel_width
            else:
                h_i, w_i = cfg.pixel_height // 2, cfg.pixel_width // 2
            img = torch.from_numpy(obs_images[k]).float().permute(2, 0, 1).unsqueeze(1)
            img = F.interpolate(img, size=(h_i, w_i), mode="bilinear", align_corners=False)
            videos.append(img.unsqueeze(0))

        videos_high = videos[0] / 255.0 * 2.0 - 1.0
        videos_wrist = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0

        vae_device = next(self._streaming_vae.vae.parameters()).device
        enc_high = self._streaming_vae.encode_chunk(videos_high.to(vae_device).to(dtype))
        enc_wrist = self._streaming_vae_half.encode_chunk(videos_wrist.to(vae_device).to(dtype))

        enc_out = torch.cat([
            torch.cat(enc_wrist.split(1, dim=0), dim=-1),
            enc_high,
        ], dim=-2)

        mu, _ = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self._vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self._vae.config.latents_std).to(mu.device)
        mu_norm = self._normalize_latents(mu, latents_mean, 1.0 / latents_std)
        return mu_norm.to(device)

    @staticmethod
    def _normalize_latents(latents, mean, std):
        mean = mean.view(1, -1, 1, 1, 1).to(latents.device)
        std = std.view(1, -1, 1, 1, 1).to(latents.device)
        return ((latents.float() - mean) * std).to(latents)

    # ------------------------------------------------------------------
    # single chunk inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _infer_chunk(self, init_latent, frame_st_id, device, dtype,
                     cache, action_mask, scheduler, action_scheduler):
        cfg = self.config
        fcs = cfg.frame_chunk_size
        ps = cfg.patch_size

        lh = (cfg.pixel_height // 16 * 3) // 2
        lw = cfg.pixel_width // 16

        latents = torch.randn(1, ROBOTWIN_LATENT_CHANNELS, fcs, lh, lw, device=device, dtype=dtype)
        actions = torch.randn(1, cfg.action_dim, fcs, cfg.action_per_frame, 1, device=device, dtype=dtype)

        scheduler.set_timesteps(cfg.num_inference_steps)
        action_scheduler.set_timesteps(cfg.action_num_inference_steps)
        timesteps = F.pad(scheduler.timesteps, (0, 1), value=0)
        action_timesteps = F.pad(action_scheduler.timesteps, (0, 1), value=0)

        # Precompute grid_ids for RoPE
        video_grid_id = get_mesh_id(
            fcs // ps[0], lh // ps[1], lw // ps[2], 0, 1, frame_st_id,
        ).to(device)
        action_grid_id = get_mesh_id(
            fcs, cfg.action_per_frame, 1, 1, 1, frame_st_id, action=True,
        ).to(device)

        cache.start(frame_st_id)

        # --- video denoise ---
        for i, t in enumerate(tqdm(timesteps, desc="video denoise")):
            last_step = i == len(timesteps) - 1
            latent_cond = init_latent[:, :, 0:1].to(dtype) if frame_st_id == 0 else None

            # Patchify: [B, C, F, H, W] -> [B, L, C*p1*p2*p3]
            noisy = latents.clone()
            if latent_cond is not None:
                noisy[:, :, 0:1] = latent_cond
            x = rearrange(
                noisy,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=ps[0], p2=ps[1], p3=ps[2],
            )

            # Per-token timesteps
            t_val = float(t)
            n_tokens = x.shape[1]
            ts = torch.full((1, n_tokens), t_val, dtype=torch.float32, device=device)
            if latent_cond is not None and frame_st_id == 0:
                cond_tokens = (lh // ps[1]) * (lw // ps[2])
                ts[:, :cond_tokens] = 0.0

            pred = self._transformer.predict_flow(
                x, ts, cache,
                input={"grid_id": video_grid_id},
                persist=last_step,
            )

            if not last_step:
                pred = rearrange(
                    pred,
                    'b (f h w) (c kt kh kw) -> b c (f kt) (h kh) (w kw)',
                    f=fcs // ps[0], h=lh // ps[1], w=lw // ps[2],
                    kt=ps[0], kh=ps[1], kw=ps[2],
                )
                latents = scheduler.step(pred, t, latents)

            latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

        # --- action denoise ---
        for i, t in enumerate(tqdm(action_timesteps, desc="action denoise")):
            last_step = i == len(action_timesteps) - 1
            action_cond = (
                torch.zeros(1, cfg.action_dim, 1, cfg.action_per_frame, 1, device=device, dtype=dtype)
                if frame_st_id == 0 else None
            )

            noisy_a = actions.clone()
            if action_cond is not None:
                noisy_a[:, :, 0:1] = action_cond
            noisy_a[:, ~action_mask] *= 0
            x_a = rearrange(noisy_a, 'b c f h w -> b (f h w) c')

            t_val = float(t)
            n_tokens_a = x_a.shape[1]
            ts_a = torch.full((1, n_tokens_a), t_val, dtype=torch.float32, device=device)
            if action_cond is not None and frame_st_id == 0:
                ts_a[:, :cfg.action_per_frame] = 0.0

            pred = self._transformer.predict_action_flow(
                x_a, ts_a, cache,
                input={"grid_id": action_grid_id},
                persist=last_step,
            )

            if not last_step:
                pred = rearrange(pred, "b (f n) c -> b c f n 1", f=fcs)
                actions = action_scheduler.step(pred, t, actions)

            actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        # Commit both video and action KV to the cache
        self._transformer.commit_cache_slot(cache)
        cache.finalize(frame_st_id)

        actions[:, ~action_mask] *= 0
        return actions, latents

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        cfg = self.config

        if cfg.metadata_only:
            mp = self._write_metadata_manifest()
            logger.info("Wrote manifest -> {}", mp.resolve())
            return

        prompt = resolve_prompt(cfg.prompt)
        image_paths = self._validate_inputs()

        device = torch.device(f"cuda:{self.local_rank}")
        dtype = torch.bfloat16

        # seed
        torch.manual_seed(cfg.seed)

        # load models
        logger.info("Loading models from {}", cfg.checkpoint_root)
        self._load_models(device, dtype)

        from lingbot_va._loaders import FlowMatchScheduler

        scheduler = FlowMatchScheduler(shift=cfg.snr_shift, sigma_min=0.0, extra_one_step=True)
        action_scheduler = FlowMatchScheduler(shift=cfg.action_snr_shift, sigma_min=0.0, extra_one_step=True)

        # encode prompt
        if cfg.benchmark:
            torch.cuda.synchronize()
            t_pipeline_start = time.perf_counter()
        logger.info("Encoding prompt")
        use_cfg = (cfg.guidance_scale > 1) or (cfg.action_guidance_scale > 1)
        if cfg.enable_offload:
            self._text_encoder = self._text_encoder.to(device)
        prompt_embeds = self._encode_prompt(prompt, device, dtype)
        neg_embeds = self._encode_prompt("", device, dtype) if use_cfg else prompt_embeds
        if cfg.enable_offload:
            self._text_encoder = self._text_encoder.cpu()
            torch.cuda.empty_cache()

        # load observation images
        obs_images = {
            k: np.array(Image.open(str(p)).convert("RGB"))
            for k, p in image_paths.items()
        }

        # encode observation
        logger.info("Encoding observation")
        if cfg.enable_offload:
            self._vae.to(device)
        init_latent = self._encode_obs(obs_images, device, dtype)
        if cfg.enable_offload:
            self._vae.cpu()
            torch.cuda.empty_cache()

        # initialize native KV cache
        logger.info("Initializing AR cache")
        ar_cache = self._transformer.initialize_autoregressive_cache(
            text_embeddings=prompt_embeds,
            negative_text_embeddings=neg_embeds if use_cfg else None,
            batch_size=1,
        )

        # action mask
        action_mask = torch.zeros(cfg.action_dim, dtype=torch.bool, device=device)
        action_mask[list(ROBOTWIN_USED_ACTION_CHANNEL_IDS)] = True

        # AR loop
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        action_processor = LingbotVAActionProcessor()
        pred_latents = []
        pred_actions = []

        if cfg.benchmark:
            torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            chunk_times = []

        for chunk_id in range(cfg.num_chunks):
            frame_st_id = chunk_id * cfg.frame_chunk_size
            logger.info("Generating chunk {}/{} (frame_st_id={})", chunk_id + 1, cfg.num_chunks, frame_st_id)
            if cfg.benchmark:
                torch.cuda.synchronize()
                t_chunk_start = time.perf_counter()
            raw_actions, latent_chunk = self._infer_chunk(
                init_latent, frame_st_id, device, dtype,
                ar_cache, action_mask,
                scheduler, action_scheduler,
            )
            if cfg.benchmark:
                torch.cuda.synchronize()
                chunk_times.append(time.perf_counter() - t_chunk_start)
            post_actions = action_processor.postprocess(raw_actions)
            pred_latents.append(latent_chunk.detach().cpu())
            pred_actions.append(post_actions)
            del raw_actions, latent_chunk
            torch.cuda.empty_cache()

        if cfg.benchmark:
            torch.cuda.synchronize()
            t_infer_end = time.perf_counter()
            infer_elapsed = t_infer_end - t_infer_start
            pipeline_elapsed = t_infer_end - t_pipeline_start

            timing_results = {
                "method": "flashdreams-lingbot-va",
                "num_chunks": cfg.num_chunks,
                "total_pipeline_sec": round(pipeline_elapsed, 4),
                "total_inference_sec": round(infer_elapsed, 4),
                "per_chunk_sec": [round(t, 4) for t in chunk_times],
                "avg_chunk_sec": round(sum(chunk_times) / len(chunk_times), 4),
                "note": "total_pipeline = prompt_enc + obs_enc + inference; total_inference = AR loop only; excludes model loading and video decoding",
            }
            timing_path = cfg.output_dir / "timing_flashdreams.json"
            timing_path.write_text(json.dumps(timing_results, indent=2))
            logger.info("Timing saved to {}", timing_path)
            logger.info("[FlashDreams] Pipeline: {:.4f}s, AR loop: {:.4f}s, avg chunk: {:.4f}s",
                         pipeline_elapsed, infer_elapsed, timing_results["avg_chunk_sec"])

        # save
        all_latents = torch.cat(pred_latents, dim=2)
        all_actions = torch.cat(pred_actions, dim=1).flatten(1).numpy()

        if cfg.save_actions:
            np.save(str(cfg.output_dir / "actions.npy"), all_actions)
            logger.info("Saved actions.npy")

        torch.save(all_latents, str(cfg.output_dir / "latents.pt"))
        logger.info("Saved latents.pt")

        # optional video decode
        if cfg.save_video:
            logger.info("Decoding video — freeing inference models")
            logger.info("VRAM before cleanup: {:.2f} GiB allocated",
                        torch.cuda.memory_allocated() / 1024**3)
            # Free KV caches (self-attn + cross-attn)
            for nc in [ar_cache.network_cache, ar_cache.network_cache_uncond]:
                if nc is None:
                    continue
                for bc in nc.block_caches:
                    bc.self_attn.reset()
                    bc.cross_attn.text.k = torch.empty(0)
                    bc.cross_attn.text.v = torch.empty(0)
            del ar_cache
            # Free streaming VAE caches (but keep self._vae on GPU for decode)
            self._streaming_vae.clear_cache()
            if self._streaming_vae_half:
                self._streaming_vae_half.clear_cache()
                self._streaming_vae_half.vae.to("cpu")
            # Move all models except VAE to CPU
            self._transformer.network.to("cpu")
            if self._streaming_vae_half:
                self._streaming_vae_half.vae.to("cpu")
            if hasattr(self, '_text_encoder') and self._text_encoder is not None:
                self._text_encoder.to("cpu")
            del self._transformer
            del self._streaming_vae
            del self._streaming_vae_half
            del self._text_encoder
            del prompt_embeds, neg_embeds, init_latent, action_mask
            del pred_latents, pred_actions
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("VRAM after cleanup: {:.2f} GiB allocated",
                        torch.cuda.memory_allocated() / 1024**3)

            self._vae = self._vae.to(device).to(dtype)

            from diffusers.utils import export_to_video
            from diffusers.video_processor import VideoProcessor

            vp = VideoProcessor(vae_scale_factor=1)

            # Denormalize latents
            lat = all_latents.to(device).to(self._vae.dtype)
            del all_latents
            lat_mean = (
                torch.tensor(self._vae.config.latents_mean)
                .view(1, self._vae.config.z_dim, 1, 1, 1)
                .to(lat.device, lat.dtype)
            )
            lat_std = (
                1.0 / torch.tensor(self._vae.config.latents_std)
                .view(1, self._vae.config.z_dim, 1, 1, 1)
                .to(lat.device, lat.dtype)
            )
            lat = lat / lat_std + lat_mean

            # Memory-efficient decode: process frame-by-frame using the VAE's
            # causal conv3d streaming interface but offload decoded frames to CPU
            # immediately to avoid accumulating the full pixel-space video on GPU.
            vae = self._vae
            num_latent_frames = lat.shape[2]
            vae.clear_cache()
            x = vae.post_quant_conv(lat)
            del lat
            decoded_frames = []
            for i in range(num_latent_frames):
                vae._conv_idx = [0]
                if i == 0:
                    frame = vae.decoder(
                        x[:, :, i:i+1, :, :],
                        feat_cache=vae._feat_map,
                        feat_idx=vae._conv_idx,
                        first_chunk=True,
                    )
                else:
                    frame = vae.decoder(
                        x[:, :, i:i+1, :, :],
                        feat_cache=vae._feat_map,
                        feat_idx=vae._conv_idx,
                    )
                decoded_frames.append(frame.detach().cpu())
                del frame
            del x
            vae.clear_cache()
            torch.cuda.empty_cache()

            out = torch.cat(decoded_frames, dim=2)
            del decoded_frames
            if vae.config.patch_size is not None:
                from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify
                out = unpatchify(out, patch_size=vae.config.patch_size)
            video = torch.clamp(out, min=-1.0, max=1.0)
            del out

            video = vp.postprocess_video(video, output_type="np")[0]
            export_to_video(video, str(cfg.output_dir / "demo.mp4"), fps=10)
            logger.info("Saved demo.mp4")

        # runtime manifest
        manifest = {
            "runner_name": cfg.runner_name,
            "checkpoint_root": str(cfg.checkpoint_root),
            "prompt": prompt,
            "num_chunks": cfg.num_chunks,
            "seed": cfg.seed,
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        (cfg.output_dir / "runtime.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Done. Outputs in {}", cfg.output_dir.resolve())
