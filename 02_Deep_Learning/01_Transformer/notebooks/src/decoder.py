import numpy as np
from attention import MultiHeadAttention
from feedforward import FeedForward
from layer_norm import LayerNorm

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        """
        x -> Masked Self-Attention -> Add & Norm
          -> Cross-Attention       -> Add & Norm
          -> Feed Forward          -> Add & Norm
        """
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def make_causal_mask(self, seq_len):
        """
        Generate causal mask to prevent attending to future tokens
        ex ) seq_len=4:
        [[1, 0, 0, 0],
         [1, 1, 0 ,0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
        """
        return np.tril(np.ones((seq_len, seq_len)))
    
    def forward(self, x, encoder_output, mask=None):
        seq_len = x.shape[0]

        # 1. Masked Multi-Head Self-Attention + Residual + LayerNorm
        causal_mask = self.make_causal_mask(seq_len)
        attn_output, _ = self.masked_attention.forward(x, x, x, causal_mask)
        x = self.norm1.forward(x + attn_output)

        # 2. Cross-Attention + Residual + LayerNorm
        # Q: Decoder, K/V: Encoder
        cross_output, _ = self.cross_attention.forward(x, encoder_output, encoder_output, mask)
        x = self.norm2.forward(x + cross_output)

        # 3. Feed-Forward + Residual + LayerNorm
        ff_output = self.ff.forward(x)
        x = self.norm3.forward(x + ff_output)

        return x
    
class Decoder:
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        """
        num_layers: Number of Decoder Block
        """
        self.layers = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, x, encoder_output, mask=None):
        for layer in self.layers:
            x = layer.forward(x, encoder_output, mask)
        return x