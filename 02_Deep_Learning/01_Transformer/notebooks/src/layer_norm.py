import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        """
        d_model: dimension of the model
        eps: to prevent denom being 0
        gamma, beta: hyper-parameter (scale, shift)
        """

        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)    # (seq_len, 1)
        std = np.std(x, axis=-1, keepdims=True)      # (seq_len, 1)

        # Normalize witin each Token across d_model dimensions
        x_norm = (x - mean) / (std + self.eps)         # (seq_len, d_model)

        return self.gamma * x_norm + self.beta