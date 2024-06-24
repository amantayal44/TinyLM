from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
import time

@dataclass
class LlamaConfig:
    vocab_size: int = 1024
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 8
    n_dim: int = 256
    multiple_of: int = 256
    dropout: float = 0.0
    norm_eps: float = 1e-5


# RMSNorm normalization
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(self.weight.dtype)

# FFN layer using SwiGLU
class FFN(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        n_dim = config.n_dim
        multiple_of = config.multiple_of
        dropout = config.dropout

        hidden_dim = config.n_dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(n_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_dim, bias=False)
        self.w3 = nn.Linear(n_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return x


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    dim_freqs = 1.0 / (theta ** (torch.arange(0, dim, 2) / dim))
    # freq for each position is m*theta
    freqs = torch.outer(torch.arange(seq_len, device=dim_freqs.device), dim_freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert to 2d vectors
    q_reshaped = q.reshape(q.shape[:-1] + (-1, 2)) # (B, T, nh_q, C/2, 2)
    k_reshaped = k.reshape(k.shape[:-1] + (-1, 2)) # (B, T, nh_k, C/2, 2)

    q_r, q_i = q_reshaped[..., 0], q_reshaped[..., 1] # (B, T, nh_q, C/2), (B, T, nh_q, C/2)
    k_r, k_i = k_reshaped[..., 0], k_reshaped[..., 1] # (B, T, nh_k, C/2), (B, T, nh_k, C/2)

    # Reshape freqs
    cos_freqs = cos_freqs.unsqueeze(0).unsqueeze(2) # (1, T, 1, C/2)
    sin_freqs = sin_freqs.unsqueeze(0).unsqueeze(2) # (1, T, 1, C/2)

    # Apply rotations in-place
    q_out_r = q_r * cos_freqs - q_i * sin_freqs
    q_out_i = q_r * sin_freqs + q_i * cos_freqs
    k_out_r = k_r * cos_freqs - k_i * sin_freqs
    k_out_i = k_r * sin_freqs + k_i * cos_freqs

    q_out = torch.stack((q_out_r, q_out_i), dim=-1).reshape_as(q) # (B, T, nh_q, C)
    k_out = torch.stack((k_out_r, k_out_i), dim=-1).reshape_as(k) # (B, T, nh_k, C)

    return q_out, k_out

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# MHA layer using GQA and RoPE
class MHA(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        assert config.n_dim % config.n_head == 0, "n_dim must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_dim = config.n_dim
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.group = config.n_head // config.n_kv_head
        self.n_head_dim = config.n_dim // config.n_head
        self.dropout = config.dropout

        self.wq = nn.Linear(self.n_dim, self.n_head * self.n_head_dim, bias=False)
        self.wk = nn.Linear(self.n_dim, self.n_kv_head * self.n_head_dim, bias=False)
        self.wv = nn.Linear(self.n_dim, self.n_kv_head * self.n_head_dim, bias=False)
        self.proj = nn.Linear(self.n_head * self.n_head_dim, self.n_dim, bias=False)
        self.proj_dropout = nn.Dropout(self.dropout)


    def forward(self, x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor):
        B,T,C = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(B, T, self.n_head, self.n_head_dim) # (B, T, n_head, n_head_dim)
        k = k.view(B, T, self.n_kv_head, self.n_head_dim) # (B, T, n_kv_head, n_head_dim)
        v = v.view(B, T, self.n_kv_head, self.n_head_dim) # (B, T, n_kv_head, n_head_dim)

        # RoPE embeddings
        q, k = apply_rope(q, k, cos_freqs, sin_freqs)

        # Repeate kv heads for Grouped Query Attention
        k = repeat_kv(k, self.group)
        v = repeat_kv(v, self.group)

        # Transposing head dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.norm1 = RMSNorm(config.n_dim, eps=config.norm_eps)
        self.attn = MHA(config)
        self.norm2 = RMSNorm(config.n_dim, eps=config.norm_eps)
        self.ffn = FFN(config)

    def forward(self, x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor):
        x = x + self.attn(self.norm1(x), cos_freqs, sin_freqs)
        x = x + self.ffn(self.norm2(x))
        return x
    
        
# Adam - b1 = 0.9, b2 = 0.95, e = 1e-5, cosine lr - min_lr = 0.1*max_lr and grad clipping of 1.0, warmup_steps=2000, batch_size=4M tokens and total_token=2T -> steps = 500,000, lr = 3e-4 (for small) and 1.5e-4 (for large)
class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.n_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_dim, eps=config.norm_eps)
        self.out = nn.Linear(config.n_dim, config.vocab_size, bias=False)

        # Weight sharing between token embedding and output layer
        self.tok_embed.weight = self.out.weight

        cos_freqs, sin_freqs = precompute_rope_freqs(config.n_dim // config.n_head, config.block_size, theta=10000.0)
        self.register_buffer('cos_freqs', cos_freqs, persistent=False)
        self.register_buffer('sin_freqs', sin_freqs, persistent=False)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}" 
        
        x = self.tok_embed(idx)
        x = self.dropout(x)

        start = time.time()
        cos_freqs, sin_freqs = self.cos_freqs[:T], self.sin_freqs[:T]
        for layer in self.layers:
            x = layer(x, cos_freqs, sin_freqs)

        x = self.norm(x)
        x = self.out(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, self.config.vocab_size), targets.view(-1), ignore_index=-1)

        return x, loss

    def configure_optimizers(self, lr: float, weight_decay: float = 0):
        decay_params = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        non_decay_params = [p for _, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in non_decay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(non_decay_params)}, with {num_non_decay_params:,} parameters")

        # For LLama, b1 = 0.9, b2 = 0.95 and eps = 1e-5
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-5)
        return optimizer

if __name__ == '__main__':
    device = utils.get_device()

    config = LlamaConfig(n_kv_head=4)
    print(config)

    model = Llama(config).to(device)
    print(model)

    x = torch.randint(0, config.vocab_size, (10, config.block_size)).to(device)
    y, loss = model(x, x)
    
    print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    approx_loss = torch.log(torch.tensor(config.vocab_size))
    print(f'Approx loss with random selection: {approx_loss:.4f}, Loss: {loss:.4f}')

    model_params = utils.count_parameters(model)
    print(f'Model has {model_params:,} parameters.')

    optimizer = model.configure_optimizers(lr=3e-4, weight_decay=0.1)

