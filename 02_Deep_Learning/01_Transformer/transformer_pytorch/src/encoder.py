import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForward
from layer_norm import LayerNorm

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        One encoder layer:
            x -> Multi-Head Self Attention -> Add & Norm 
              -> Feed Forward -> Add & Norm
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # 1. Multi-Head Self-Attention + Add & Norm
        attn_output, _ = self.attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Feed Forward + Add & Norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x