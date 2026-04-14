import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        d_model : dim. to normalize over
        eps     : numerical stability constant
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # learned scale
        self.beta = nn.Parameter(torch.zeros(d_model))   # learned shift
        self.eps = eps

    def forward(self, x):
        # Normalize within each token across d_model dim.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        norm_x = (x - mean) / (std + self.eps)

        return self.gamma * norm_x + self.beta