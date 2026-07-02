import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from typing import Optional

# ==================== ТОКЕНИЗАТОР ====================
from testoBPE import BPE
tok = BPE()
VOCAB_SIZE = tok.len if hasattr(tok, 'len') else len(tok)
# ====================================================

# ==================== ПАРАМЕТРЫ ====================
MODEL_DIM = 1536
N_HEADS = 16
N_KV_HEADS = 4
HEAD_DIM = MODEL_DIM // N_HEADS
FFN_DIM = 4096
DROPOUT = 0.1
ROPE_THETA = 50000.0
MAX_SEQ_LEN = 20480
N_LAYERS = 10
OUTPUT_DIM = 768
BOS_TOKEN_ID = 3
PAD_TOKEN_ID = 0
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
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        self.register_buffer('cos_cached', cos.repeat_interleave(2, dim=-1))
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
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, L, self.n_kv_heads, 2, self.head_dim).permute(0, 2, 3, 1, 4)
        k, v = kv[:, :, 0, :, :], kv[:, :, 1, :, :]

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q, L)
        k = self.rope(k, L)

        if self.repeat > 1:
            k = k.repeat_interleave(self.repeat, dim=1)
            v = v.repeat_interleave(self.repeat, dim=1)

        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask.bool().unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = mask.bool()

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
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
        ffn_in = self.norm2(x)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(ffn_in)) * self.ffn_up(ffn_in))
        return x + self.dropout(ffn_out)


class RAGEncoder(nn.Module):
    def __init__(self, dim: int = MODEL_DIM, n_heads: int = N_HEADS,
                 n_kv_heads: int = N_KV_HEADS, ffn_dim: int = FFN_DIM,
                 n_layers: int = N_LAYERS, dropout: float = DROPOUT,
                 output_dim: int = OUTPUT_DIM, max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.tok_emb = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD_TOKEN_ID)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, output_dim, bias=False)

        self._use_checkpointing = False

    def load_embeddings(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        ckpt = torch.load(path, map_location="cpu")
        
        # Распаковка распространенных форматов чекпоинтов (model + optimizer, lightning и т.д.)
        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ['model', 'model_state_dict', 'state_dict', 'net', 'module']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    break
                    
        emb_weight = None
        target_keys = ['tok_emb.weight', 'module.tok_emb.weight', 'embedding.weight', 'embeddings.weight', 'model.tok_emb.weight']
        
        if isinstance(state_dict, dict):
            for k in target_keys:
                if k in state_dict:
                    emb_weight = state_dict[k]
                    break
            # Фоллбэк: поиск по ожидаемой форме тензора
            if emb_weight is None:
                for k, v in state_dict.items():
                    if isinstance(v, torch.Tensor) and v.shape == (VOCAB_SIZE, self.dim):
                        emb_weight = v
                        print(f"[i] Fallback: found embedding tensor under key '{k}'")
                        break

        if emb_weight is None:
            if isinstance(state_dict, dict):
                keys = list(state_dict.keys())[:15]
                raise ValueError(f"Embedding weights not found. Available keys: {keys}...")
            else:
                raise ValueError("Checkpoint structure not recognized.")

        self.tok_emb.weight.data.copy_(emb_weight)
        self.tok_emb.weight.requires_grad_(False)
        print(f"[✓] Embeddings loaded from {path} | Shape: {self.tok_emb.weight.shape} | Frozen: True")

    def enable_gradient_checkpointing(self, enable: bool = True):
        self._use_checkpointing = enable

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L = tokens.shape
        x = self.dropout(self.tok_emb(tokens))

        for layer in self.layers:
            if self._use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)

        x = self.norm(x)
        bos_hidden = x[:, 0, :]  # [BOS] pooling
        
        out = self.proj(bos_hidden)
        out = F.normalize(out, p=2, dim=-1)
        return out

    def get_num_params(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)


if __name__ == '__main__':
    print("="*50)
    print("🧪 ТЕСТ ИНИЦИАЛИЗАЦИИ И ЭМБЕДДИНГОВ")
    print("="*50)

    model = RAGEncoder()
    print(f"📦 Размер словаря: {VOCAB_SIZE}")
    print(f"📐 Конфигурация: dim={MODEL_DIM}, layers={N_LAYERS}, heads={N_HEADS}/{N_KV_HEADS}")
    
    total_params = model.get_num_params(trainable_only=False)
    trainable_params = model.get_num_params(trainable_only=True)
    print(f"📊 Всего параметров: {total_params / 1e6:.2f}M")
    print(f"📊 Обучаемых параметров (до загрузки эмбеддингов): {trainable_params / 1e6:.2f}M")

    ckpt_path = "OLD/logs/checkpoint.pth"
    if os.path.exists(ckpt_path):
        try:
            model.load_embeddings(ckpt_path)
            trainable_after = model.get_num_params(trainable_only=True)
            print(f"📊 Обучаемых параметров (после заморозки): {trainable_after / 1e6:.2f}M")
        except Exception as e:
            print(f"❌ Ошибка загрузки эмбеддингов: {e}")
    else:
        print(f"⚠️ Файл {ckpt_path} не найден. Эмбеддинги не загружены.")

    print("\n🔍 Проверка forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy_tokens = torch.tensor([[3, 15, 22, 10, 4, 0, 0, 0]], device=device)
    dummy_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], device=device)

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.float16):
            embeddings = model(dummy_tokens, attention_mask=dummy_mask)
            
    print(f"✅ Вход тензор: {dummy_tokens.shape}")
    print(f"✅ Выходной вектор: {embeddings.shape}")
    print(f"✅ L2 норма (должна быть ~1.0): {embeddings.norm(dim=-1).item():.6f}")
    print(f"✅ Устройство: {embeddings.device}")
    print("\n🚀 Модель готова к обучению.")