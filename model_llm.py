import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ==================== ПАРАМЕТРЫ ====================
MODEL_DIM = 1536
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = MODEL_DIM // N_HEADS
FFN_DIM = 4000
DROPOUT = 0.05
ROPE_THETA = 10000.0
MAX_SEQ_LEN = int(1024*2.5)
N_LAYERS = 20
# ====================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** (-0.5))
        return x / (rms + self.eps) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = ROPE_THETA, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (L, dim/2)

        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1,1,L,dim/2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer('cos_cached', cos.repeat_interleave(2, dim=-1))  # (1,1,L,dim)
        self.register_buffer('sin_cached', sin.repeat_interleave(2, dim=-1))

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        rot_x = torch.cat((-x2, x1), dim=-1)

        return x * cos + rot_x * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.repeat = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, n_kv_heads * self.head_dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.dropout = dropout
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, L, head_dim)

        kv = self.kv_proj(x).view(B, L, self.n_kv_heads, 2, self.head_dim)  # (B, L, n_kv_heads, 2, head_dim)
        kv = kv.permute(0, 2, 3, 1, 4)  # (B, n_kv_heads, 2, L, head_dim)

        k = kv[:, :, 0, :, :]  # (B, n_kv_heads, L, head_dim)
        v = kv[:, :, 1, :, :]  # (B, n_kv_heads, L, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, L)
        k = self.rope(k, L)

        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)  # (B, n_heads, L, head_dim)
            v = v.repeat_interleave(self.repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scale
        )

        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, n_kv_heads, dropout)

        self.ffn_gate = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        gate = F.silu(self.ffn_gate(self.norm2(x)))
        up = self.ffn_up(self.norm2(x))
        ffn_out = self.ffn_down(gate * up)
        x = x + self.dropout(ffn_out)
        return x


class DenseTransformer(nn.Module):
    def __init__(self, vocab_size: int,
                 dim: int = MODEL_DIM,
                 n_heads: int = N_HEADS,
                 n_kv_heads: int = N_KV_HEADS,
                 ffn_dim: int = FFN_DIM,
                 n_layers: int = N_LAYERS,
                 dropout: float = DROPOUT):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight  # tie weights

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module in [layer.attn.out_proj for layer in self.layers] + \
                         [layer.ffn_down for layer in self.layers]:
                torch.nn.init.zeros_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                class_weights: Optional[torch.Tensor] = None,
                ignore_index: int = -1) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T = tokens.shape

        x = self.dropout(self.tok_emb(tokens))

        for layer in self.layers:
            x = layer(x, mask=None)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.reshape(-1),
            weight=class_weights,
            ignore_index=ignore_index
        )

        return logits, {'loss': loss, 'main_loss': loss}

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    from testoBPE import BPE
    tok = BPE()
    model = DenseTransformer(vocab_size=tok.len)
    print(f"Параметров: {model.get_num_params() / 1e6:.2f}M")