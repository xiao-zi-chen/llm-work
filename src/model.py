from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from utils import PAD_ID


@dataclass
class ModelConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_kv_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 96
    attn_type: str = "mha"
    use_moe: bool = False
    num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_weight: float = 0.01


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiQueryAttention(nn.Module):
    """MHA/MQA/GQA through a configurable number of key-value heads."""

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = query.shape
        k_len = key.size(1)

        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), x.new_zeros(())


class MoEFeedForward(nn.Module):
    """Top-k token-level MoE FFN with a small load-balancing auxiliary loss."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        router_logits = self.router(flat)
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_idx == expert_id
            if not mask.any():
                continue
            token_pos, route_pos = mask.nonzero(as_tuple=True)
            expert_out, _ = expert(flat[token_pos])
            output[token_pos] += expert_out * top_probs[token_pos, route_pos].unsqueeze(-1)

        importance = router_probs.mean(dim=0)
        load = F.one_hot(top_idx, num_classes=self.num_experts).float().mean(dim=(0, 1))
        aux_loss = self.num_experts * torch.sum(importance * load)
        return output.view(original_shape), aux_loss


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiQueryAttention(cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout)
        self.ffn = MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout, cfg.num_experts, cfg.moe_top_k) if cfg.use_moe else FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.dropout(attn_out)
        ffn_out, aux = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x, aux


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiQueryAttention(cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout)
        self.cross_attn = MultiQueryAttention(cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout)
        self.ffn = MoEFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout, cfg.num_experts, cfg.moe_top_k) if cfg.use_moe else FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_x = self.norm1(x)
        x = x + self.dropout(self.self_attn(norm_x, norm_x, norm_x, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, memory_mask))
        ffn_out, aux = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_out)
        return x, aux


class TransformerMT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.generator = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src != PAD_ID).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tgt.shape
        pad_mask = (tgt != PAD_ID).unsqueeze(1).unsqueeze(2)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device))
        causal = causal.unsqueeze(0).unsqueeze(0)
        return pad_mask & causal

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)
        x = self.pos(self.src_embed(src) * math.sqrt(self.cfg.d_model))
        aux = x.new_zeros(())
        for layer in self.encoder:
            x, layer_aux = layer(x, src_mask)
            aux = aux + layer_aux
        return self.enc_norm(x), aux

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tgt_mask = self.make_tgt_mask(tgt)
        memory_mask = self.make_src_mask(src)
        x = self.pos(self.tgt_embed(tgt) * math.sqrt(self.cfg.d_model))
        aux = x.new_zeros(())
        for layer in self.decoder:
            x, layer_aux = layer(x, memory, tgt_mask, memory_mask)
            aux = aux + layer_aux
        return self.dec_norm(x), aux

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory, enc_aux = self.encode(src)
        dec, dec_aux = self.decode(tgt_in, memory, src)
        return self.generator(dec), enc_aux + dec_aux
