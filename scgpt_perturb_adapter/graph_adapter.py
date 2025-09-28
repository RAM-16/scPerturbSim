import torch
from torch import nn
import torch.nn.functional as F

class GraphAdapter(nn.Module):
    """Lightweight gene–gene message passing on top of frozen token embeddings."""
    def __init__(self, hidden_size=512, bottleneck=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.gate = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, neighbor_idx):
        B, T, D = x.shape
        out = x.clone()
        for t in range(1, T):
            nbrs = neighbor_idx[t]
            if not nbrs: continue
            agg = x[:, nbrs, :].mean(dim=1)
            z = self.up(F.relu(self.down(agg)))
            g = torch.sigmoid(self.gate(torch.cat([x[:, t, :], agg], dim=-1)))
            out[:, t, :] = x[:, t, :] + g * z
        return out
