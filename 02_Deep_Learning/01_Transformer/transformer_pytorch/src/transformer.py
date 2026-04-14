import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_heads=8, d_ff=2048,
                 num_layers=6, dropout=0.1, max_seq_len=512):
        """
        src_vocab_size: source vocabulary size
        tgt_vocab_size: target vocabulary size
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.register_buffer('pe', self._positional_encoding(max_seq_len, d_model))

        # Encoder & Decoder
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)

        # Output Projection
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        
    def _positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float() # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * - (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # even dim.
        pe[:, 1::2] = torch.cos(position * div_term) # odd dim.
        return pe.unsqueeze(0) # (1, max_seq_len, d_model)
    
    def make_src_mask(self, src, pad_idx):
        # (batch, 1, 1, src_seq_len)
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt, pad_idx):
        # (batch, 1, 1, tgt_seq_len)
        return (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pe[:, :x.size(1)])
        return self.encoder(x, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(x + self.pe[:, :x.size(1)])
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_linear(decoder_output)