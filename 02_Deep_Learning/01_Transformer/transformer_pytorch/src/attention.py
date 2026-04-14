import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch, heads, seq_len, d_k)
        K: (batch, heads, seq_len, d_k)
        V: (batch, heads, seq_len, d_k)
        mask: padding mask - (batch, 1, 1, seq_len) or causal mask - (batch, 1, seq_len, seq_len)
        """
        d_k = Q.size(-1)

        # 1. Q x K.t / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. Apply mask (replace masked positions with -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # 3. Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted sum of V
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
    
    def split_heads(self, x):
        """
        (batch, seq_len, d_model)  ->  (batch, heads, seq_len, d_k)
        """
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q, K ,V, mask = None):
        # 1. Linear projection
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2. Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Scaled dot-product attention per head
        output, attn_weights = self.attention.forward(Q, K, V, mask)

        # 4. Concatenate heads
        batch, _, seq_len, _ = output.size()
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.d_model)

        # 5. Output linear projection
        output = self.W_O(output)

        return output, attn_weights