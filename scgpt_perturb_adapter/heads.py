import torch
from torch import nn

class PerturbConditioner(nn.Module):
    """Adds a learned perturbation embedding to every token (FiLM-lite)."""
    def __init__(self, num_perts: int, hidden_size: int):
        super().__init__()
        self.emb = nn.Embedding(num_perts, hidden_size)

    def forward(self, x, pert_ids):
        cond = self.emb(pert_ids).unsqueeze(1)
        return x + cond

class GeneRegressorHead(nn.Module):
    """Predict a scalar (log1p expression) per token."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, token_emb):
        y = self.proj(token_emb)
        return y.squeeze(-1)
