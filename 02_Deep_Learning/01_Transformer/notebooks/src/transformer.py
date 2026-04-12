import numpy as np
from encoder import Encoder
from decoder import Decoder
from positional_encoding import positional_encoding

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, num_layers, vocab_size):
        """
        d_model: embedding dimension
        num_heads: number of attention heads
        d_ff: feed-forward hidden dimension
        num_layers: number of encoder/decoder layers
        vocab_size: output vocabulary size
        """
        self.d_model = d_model
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)

        # Output linear layer: d_model → vocab_size
        scale = np.sqrt(1.0 / d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * scale

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # numerical stability
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, src, tgt):
        """
        src: source sequence (src_len, d_model)
        tgt: target sequence (tgt_len, d_model)
        """
        # 1. Add Positional Encoding to src and tgt
        src = src * np.sqrt(self.d_model) + positional_encoding(src.shape[0], self.d_model)
        tgt = tgt * np.sqrt(self.d_model) + positional_encoding(tgt.shape[0], self.d_model)

        # 2. Encoder forward pass
        encoder_output = self.encoder.forward(src)

        # 3. Decoder forward pass
        decoder_output = self.decoder.forward(tgt, encoder_output)

        # 4. Output Linear + Softmax + probability distribution over vocab
        logits = np.matmul(decoder_output, self.W_out)
        probs = self.softmax(logits)

        return probs