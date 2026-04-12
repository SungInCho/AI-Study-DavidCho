import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    mask: (seq_len, seq_len) - For Decoder Masking (optional)
    """
    d_k = Q.shape[-1]

    # 1. Q x K.T
    scores = np.matmul(Q, K.T) # (seq_len, seq_len)

    # 2. Scale
    scores = scores / np.sqrt(d_k)  # prevent gradient vanishing 

    # 3. Mask
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # 3. Softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Score x V
    output = np.matmul(attn_weights, V) # (seq_len, d_v)

    return output, attn_weights


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        """
        d_model: Embedding dimension
        num_heads: number of heads
        d_k = d_model / num_heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Scaled Initialization
        scale = np.sqrt(1.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def split_heads(self, x):
        """
        (seq_len, d_model) -> (num_heads, seq_len, d_k)
        """
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)
    
    def forward(self, Q, K, V, mask=None):
        # 1. Linear projection
        Q = np.matmul(Q, self.W_Q)
        K = np.matmul(K, self.W_K)
        V = np.matmul(V, self.W_V)

        # 2. Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 3. Scaled dot-product attention per head
        all_outputs = []
        all_weights = []
        for i in range(self.num_heads):
            out, weights = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            all_outputs.append(out)
            all_weights.append(weights)

        # 4. Concatenate heads
        concat = np.concatenate(all_outputs, axis = -1)

        # 5. Final linear projection
        output = np.matmul(concat, self.W_O)

        return output, np.array(all_weights)
