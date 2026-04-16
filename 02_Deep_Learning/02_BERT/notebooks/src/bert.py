import torch
import torch.nn as nn
import sys
sys.path.append("../../01_Transformer/transformer_pytorch/src")
from encoder import Encoder

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=512, dropout=0.1):
        """
        BERT Input Embedding:
            Token Embedding + Positional Embedding + Segment Embedding

        Args:
            vocab_size  : vocabulary size
            d_model     : embedding dimension (BERT-base: 768)
            max_seq_len : maximum sequence length (BERT: 512)
            dropout     : dropout rate
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.segment_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, segment_ids):
        """
        Args:
            x           : token indices (batch, seq_len)
            segment_ids : sentence A/B indicators (batch, seq_len)
        Returns:
            embedding   : (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # (1, seq_len)

        # Sum three embeddings
        embedding = (self.token_embedding(x) 
                     + self.pos_embedding(positions)
                     + self.segment_embedding(segment_ids))
        
        return self.dropout(self.norm(embedding))
    
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12,
                 d_ff=3072, num_layers=12, dropout=0.1, max_seq_len=512):
        """
        BERT-base config.:
            vocab_size   :  vocabulary size
            d_model      :  768
            num_heads    :  12
            d_ff         :  3072 (d_model x 4)
            num_layers   :  12
            max_seq_len  :  512
        """
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len, dropout)

        # Reuse Transformer Encoder - no causal mask -> Birectional
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers, dropout)

        # MLM head: predict original token at [MASK] position
        # Linear -> GELU -> LayerNorm -> Linear(vocab_size)
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

        # NSP head: predict if sentence B follows sentence A
        # Uses only [CLS] token at position 0 which aggregates full sequence meaning
        self.nsp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 2)  # 0: not next sentence, 1: is next sentence
        )

    def forward(self, x, segment_ids, mask=None):
        """
        Args:
            x            :  token indices (batch, seq_len)
            segment_ids  :  sentence A/B indicators (batch, seq_len)
            mask         :  padding mask (batch, 1, 1, seq_len)
        """
        # Embedding: Token + Positional + Segment
        x = self.embedding(x, segment_ids) # (batch, seq_len, d_model)

        # Bidirectional Encoder (no causal mask)
        x = self.encoder(x, mask)

        # MLM: predict for all token positions
        mlm_output = self.mlm_head(x) # (batch, seq_len, vocab_size)

        # NSP: [CLS] token at position 0 aggregates full sequence meaning
        nsp_output = self.nsp_head(x[:, 0, :]) # (batch, 2)

        return mlm_output, nsp_output