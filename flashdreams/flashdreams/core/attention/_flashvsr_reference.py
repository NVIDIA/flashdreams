import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange

from block_sparse_attn import block_sparse_attn_func

from projects.cosmos.sil.causal_wan.fast_infer.v2.tests.templated_benchmark import torch_cudnn_attention

try:
    from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb
except ImportError:
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

from projects.cosmos.sil.causal_wan.fast_infer.v2.network_cosmos2 import SinsoidalRotaryFrequency

@torch.no_grad()
def build_local_block_mask_shifted_vec_normal_slide(block_h: int,
                                                   block_w: int,
                                                   win_h: int = 6,
                                                   win_w: int = 6,
                                                   include_self: bool = True,
                                                   device=None) -> torch.Tensor:
    device = device or torch.device("cpu")
    H, W = block_h, block_w
    r = torch.arange(H, device=device)
    c = torch.arange(W, device=device)
    YY, XX = torch.meshgrid(r, c, indexing="ij")
    r_all = YY.reshape(-1)
    c_all = XX.reshape(-1)
    r_half = win_h // 2
    c_half = win_w // 2
    start_r = r_all - r_half
    end_r   = start_r + win_h - 1
    start_c = c_all - c_half
    end_c   = start_c + win_w - 1
    in_row = (r_all[None, :] >= start_r[:, None]) & (r_all[None, :] <= end_r[:, None])
    in_col = (c_all[None, :] >= start_c[:, None]) & (c_all[None, :] <= end_c[:, None])
    mask = in_row & in_col
    if not include_self:
        mask.fill_diagonal_(False)
    return mask


class WindowPartition3D:
    """Partition / reverse-partition helpers for 5-D tensors (B,F,H,W,C)."""
    @staticmethod
    def partition(x: torch.Tensor, win: Tuple[int, int, int]):
        B, F, H, W, C = x.shape
        wf, wh, ww = win
        assert F % wf == 0 and H % wh == 0 and W % ww == 0, "Dims must divide by window size."
        x = x.view(B, F // wf, wf, H // wh, wh, W // ww, ww, C)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        return x.view(-1, wf * wh * ww, C)

    @staticmethod
    def reverse(windows: torch.Tensor, win: Tuple[int, int, int], orig: Tuple[int, int, int]):
        F, H, W = orig
        wf, wh, ww = win
        nf, nh, nw = F // wf, H // wh, W // ww
        B = windows.size(0) // (nf * nh * nw)
        x = windows.view(B, nf, nh, nw, wf, wh, ww, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        return x.view(B, F, H, W, -1)


@torch.no_grad()
def generate_draft_block_mask(batch_size, nheads, seqlen,
                              q_w, k_w, topk=10, local_attn_mask=None):
    assert batch_size == 1, "Only batch_size=1 supported for now"
    assert local_attn_mask is not None, "local_attn_mask must be provided"
    avgpool_q = torch.mean(q_w, dim=1) 
    avgpool_k = torch.mean(k_w, dim=1)
    avgpool_q = rearrange(avgpool_q, 's (h d) -> s h d', h=nheads)
    avgpool_k = rearrange(avgpool_k, 's (h d) -> s h d', h=nheads)
    q_heads = avgpool_q.permute(1, 0, 2)
    k_heads = avgpool_k.permute(1, 0, 2)
    D = avgpool_q.shape[-1]
    scores = torch.einsum("hld,hmd->hlm", q_heads, k_heads) / math.sqrt(D)

    repeat_head = scores.shape[0]
    repeat_len = scores.shape[1] // local_attn_mask.shape[0]
    repeat_num = scores.shape[2] // local_attn_mask.shape[1]
    local_attn_mask = local_attn_mask.unsqueeze(1).unsqueeze(0).repeat(repeat_len, 1, repeat_num, 1)
    local_attn_mask = rearrange(local_attn_mask, 'x a y b -> (x a) (y b)')
    local_attn_mask = local_attn_mask.unsqueeze(0).repeat(repeat_head, 1, 1)
    local_attn_mask = local_attn_mask.to(torch.float32)
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == False, -float('inf'))
    local_attn_mask = local_attn_mask.masked_fill(local_attn_mask == True, 0)
    scores = scores + local_attn_mask

    attn_map = torch.softmax(scores, dim=-1)
    attn_map = rearrange(attn_map, 'h (it s1) s2 -> (h it) s1 s2', it=seqlen)
    loop_num, s1, s2 = attn_map.shape
    flat = attn_map.reshape(loop_num, -1)
    n = flat.shape[1]
    apply_topk = min(flat.shape[1]-1, topk)
    thresholds = torch.topk(flat, k=apply_topk + 1, dim=1, largest=True).values[:, -1]
    thresholds = thresholds.unsqueeze(1)
    mask_new = (flat > thresholds).reshape(loop_num, s1, s2)
    mask_new = rearrange(mask_new, '(h it) s1 s2 -> h (it s1) s2', it=seqlen)  # keep shape note
    # 修正：上行变量名统一
    # mask_new = rearrange(attn_map, 'h (it s1) s2 -> h (it s1) s2', it=seqlen) * 0 + mask_new
    mask = mask_new.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return mask



# ----------------------------
# Attention kernels
# ----------------------------
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask=None):
    if attention_mask is not None:
        # print(f"Using block sparse attention")
        seqlen = q.shape[1]
        seqlen_kv = k.shape[1]
        q = rearrange(q, "b s (n d) -> (b s) n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> (b s) n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> (b s) n d", n=num_heads)
        cu_seqlens_q = torch.tensor([0, seqlen], device=q.device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_kv], device=q.device, dtype=torch.int32)
        head_mask_type = torch.tensor([1]*num_heads, device=q.device, dtype=torch.int32)
        streaming_info = None
        base_blockmask = attention_mask
        max_seqlen_q_ = seqlen
        max_seqlen_k_ = seqlen_kv
        p_dropout = 0.0
        x = block_sparse_attn_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            head_mask_type,
            streaming_info,
            base_blockmask,
            max_seqlen_q_, max_seqlen_k_,
            p_dropout,
            deterministic=False,
            softmax_scale=None,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        ).unsqueeze(0)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    else:
        # print(f"Using cudnn attention")
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = torch_cudnn_attention(q, k, v, return_lse=False)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def rope_apply(x, freqs, num_heads):
    assert x.ndim == 3 and freqs.ndim == 4
    b, s, D = x.shape
    x = x.reshape(b, s, num_heads, D // num_heads)
    x_out = apply_rotary_pos_emb(x, freqs, tensor_format="bshd", fused=True, interleaved=True)
    return x_out.reshape(b, s, D)


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, attention_mask=attention_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = torch.nn.RMSNorm(dim, eps=eps)
        self.norm_k = torch.nn.RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)
        self.local_attn_mask = None

    def forward(self, x, freqs, f=None, h=None, w=None, topk=None,
                kv_len=None,
                is_stream=False, pre_cache_k=None, pre_cache_v=None, local_range = 9):

        B, L, D = x.shape
        assert L == f * h * w, "Sequence length mismatch with provided (f,h,w)."

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        win = (2, 8, 8)
        q = q.view(B, f, h, w, D)
        k = k.view(B, f, h, w, D)
        v = v.view(B, f, h, w, D)

        q_w = WindowPartition3D.partition(q, win)
        k_w = WindowPartition3D.partition(k, win)
        v_w = WindowPartition3D.partition(v, win)

        seqlen = f//win[0]
        one_len = k_w.shape[0] // B // seqlen
        if pre_cache_k is not None and pre_cache_v is not None:
            k_w = torch.cat([pre_cache_k, k_w], dim=0)
            v_w = torch.cat([pre_cache_v, v_w], dim=0)

        block_n = q_w.shape[0] // B
        block_s = q_w.shape[1]
        block_n_kv = k_w.shape[0] // B

        reorder_q = rearrange(q_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n, block_s=block_s)
        reorder_k = rearrange(k_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)
        reorder_v = rearrange(v_w, '(b block_n) (block_s) d -> b (block_n block_s) d', block_n=block_n_kv, block_s=block_s)

        if self.local_attn_mask is None or self.local_attn_mask_h!=h//8 or self.local_attn_mask_w!=w//8 or self.local_range!=local_range:
            self.local_attn_mask = build_local_block_mask_shifted_vec_normal_slide(h//8, w//8, local_range, local_range, include_self=True, device=k_w.device)
            self.local_attn_mask_h = h//8
            self.local_attn_mask_w = w//8
            self.local_range = local_range
        attention_mask = generate_draft_block_mask(B, self.num_heads, seqlen, q_w, k_w, topk=topk, local_attn_mask=self.local_attn_mask)

        x = self.attn(reorder_q, reorder_k, reorder_v, attention_mask)

        cur_block_n, cur_block_s, _ = k_w.shape
        cache_num = cur_block_n // one_len
        if cache_num > kv_len:
            cache_k = k_w[one_len:, :, :]
            cache_v = v_w[one_len:, :, :]
        else:
            cache_k = k_w
            cache_v = v_w

        x = rearrange(x, 'b (block_n block_s) d -> (b block_n) (block_s) d', block_n=block_n, block_s=block_s)
        x = WindowPartition3D.reverse(x, win, (f, h, w))
        x = x.view(B, f*h*w, D)

        if is_stream:
            return self.o(x), cache_k, cache_v
        return self.o(x)

