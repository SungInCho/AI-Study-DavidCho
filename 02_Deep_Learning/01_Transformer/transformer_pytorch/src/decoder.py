import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import FeedForward
from layer_norm import LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        One Decoder Layer:
            x -> Masked Multi-Head Self-Attention -> Add & Norm
              -> Multi-Head Cross-Attention -> Add & Norm
              -> Feed Forward -> Add & Norm
        """
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def make_causal_mask(self, seq_len, device):
        # Lower triangular matrix to prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0) # adds dim. - (1, 1, seq_len, seq_len)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        seq_len = x.size(1)
        causal_mask = self.make_causal_mask(seq_len, x.device)

        # Combine causal mask with target padding mask if provided
        if tgt_mask is not None:
            causal_mask = causal_mask & tgt_mask

        # 1. Masked Multi-Head Self-Attention + Add & Norm
        attn_output, _ = self.masked_attention(x, x, x, causal_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Cross-Attention + Add & Norm
        cross_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))

        # 3. Feed Forward + Add & Norm
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
    

class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x
