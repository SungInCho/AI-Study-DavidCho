import numpy as np
from attention import MultiHeadAttention
from feedforward import FeedForward
from layer_norm import LayerNorm

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        """
        x -> Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
        """
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # 1. Multi-Head Self-Attention + Residual + LayerNorm
        attn_output, _ = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)   # Add & Norm

        # 2. Feed Forward + Residual + LayerNorm
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)     # Add & Norm

        return x
    
class Encoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        """
        num_layers: number of Encoder Bolcks
        """
        self.layers = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer.forward(x, mask)
        return x