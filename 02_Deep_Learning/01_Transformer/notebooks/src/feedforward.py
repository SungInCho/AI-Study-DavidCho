import numpy as np

class FeedForward:
    def __init__(self, d_model, d_ff):
        """
        d_model: dimension of input and output
        d_ff: internal hidden dimension

        FFN(x) = max(0, xW1 + b1)W2 + b2 <- ReLU
        """
        scale = np.sqrt(1.0 / d_model)

        # Linear Layer 1: d_model -> d_ff
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)

        # Linear Layer 2: d_ff -> d_model
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, x):
        # 1. Linear Layer 1 + ReLU
        x = self.relu(np.matmul(x, self.W1) + self.b1)   # (seq_len, d_ff)

        # 2. Linear Layer 2
        x = np.matmul(x, self.W2) + self.b2              #(seq_len, d_model)

        return x