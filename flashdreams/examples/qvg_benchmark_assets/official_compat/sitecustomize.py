"""Compatibility shims for running official QVG in the FlashDreams env."""

from __future__ import annotations


def _install_torchvision_write_video() -> None:
    try:
        import torchvision.io as tv_io
    except Exception:
        return

    def write_video(filename, video_array, fps, video_codec=None, options=None, **kwargs):
        import numpy as np
        import torch

        if isinstance(video_array, torch.Tensor):
            array = video_array.detach().cpu().numpy()
        else:
            array = np.asarray(video_array)
        array = np.clip(array, 0, 255).astype(np.uint8)
        try:
            import mediapy as media

            media.write_video(filename, array, fps=fps)
            return
        except Exception:
            pass

        import cv2

        height, width = array.shape[1:3]
        writer = cv2.VideoWriter(
            str(filename),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for frame in array:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    tv_io.write_video = write_video


_install_torchvision_write_video()


def _install_official_qvg_flash_attention_fallback() -> None:
    try:
        import torch
        import torch.nn.functional as F
        import wan.modules.attention as official_attention
    except Exception:
        return

    if (
        getattr(official_attention, "FLASH_ATTN_2_AVAILABLE", False)
        or getattr(official_attention, "FLASH_ATTN_3_AVAILABLE", False)
    ):
        return

    def flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        out_dtype = q.dtype
        bsz, lq, nq, _ = q.shape
        nk = k.shape[2]
        out = q.new_zeros((bsz, lq, nq, v.shape[-1]), dtype=out_dtype)

        for batch_idx in range(bsz):
            q_len = int(q_lens[batch_idx].item()) if q_lens is not None else lq
            k_len = int(k_lens[batch_idx].item()) if k_lens is not None else k.shape[1]
            qi = q[batch_idx : batch_idx + 1, :q_len].transpose(1, 2).to(dtype)
            ki = k[batch_idx : batch_idx + 1, :k_len].transpose(1, 2).to(dtype)
            vi = v[batch_idx : batch_idx + 1, :k_len].transpose(1, 2).to(dtype)

            if q_scale is not None:
                qi = qi * q_scale
            if nq != nk:
                assert nq % nk == 0, f"Expected query heads divisible by KV heads, got {nq=} {nk=}"
                repeat = nq // nk
                ki = ki.repeat_interleave(repeat, dim=1)
                vi = vi.repeat_interleave(repeat, dim=1)

            attn_mask = None
            if window_size != (-1, -1):
                left, right = window_size
                q_pos = torch.arange(q_len, device=q.device)[:, None]
                k_pos = torch.arange(k_len, device=q.device)[None, :]
                keep = torch.ones((q_len, k_len), device=q.device, dtype=torch.bool)
                if left >= 0:
                    keep &= k_pos >= q_pos - left
                if right >= 0:
                    keep &= k_pos <= q_pos + right
                attn_mask = keep[None, None, :, :]

            oi = F.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
            out[batch_idx : batch_idx + 1, :q_len] = oi.transpose(1, 2).to(out_dtype)

        return out

    official_attention.flash_attention = flash_attention


_install_official_qvg_flash_attention_fallback()
