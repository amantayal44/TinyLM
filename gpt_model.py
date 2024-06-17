from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_dim: int = 256


class FFN(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.ff = nn.Linear(config.n_dim, 4 * config.n_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.proj = nn.Linear(4 * config.n_dim, config.n_dim)
        self.proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.ff(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x


class MHA(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.qkv = nn.Linear(config.n_dim, 3 * config.n_dim)
        self.proj = nn.Linear(config.n_dim, config.n_dim)
        self.proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)

        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)  # (B, nh, T, nd)
        k = k.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, C // self.config.n_head).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_dim)
        self.mha = MHA(config)
        self.ln2 = nn.LayerNorm(config.n_dim)
        self.ffn = FFN(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_embed = nn.Embedding(config.vocab_size, config.n_dim)
        self.pos_embed = nn.Embedding(config.block_size, config.n_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln = nn.LayerNorm(config.n_dim)
        self.out = nn.Linear(config.n_dim, config.vocab_size, bias=False)

        # Weight sharing between token embedding and output layer
        self.tok_embed.weight = self.out.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # proj layer has std divided by n_layers
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        tok_embed = self.tok_embed(idx)
        pos = torch.arange(T, device=idx.device)
        pos_embed = self.pos_embed(pos)
        x = tok_embed + pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
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

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


if __name__ == '__main__':
    config = GPTConfig()
    print(config)

    model = GPT(config)
    print(model)

    x = torch.randint(0, config.vocab_size, (10, config.block_size))
    y, loss = model(x, x)

    print(f'Input shape: {x.shape}, Output shape: {y.shape}')

    approx_loss = torch.log(torch.tensor(config.vocab_size))
    print(f'Approx loss with random selection: {approx_loss:.4f}, Loss: {loss:.4f}')

    model_params = sum(p.numel() for p in model.parameters())
    print(f'Model has {model_params:,} parameters.')
