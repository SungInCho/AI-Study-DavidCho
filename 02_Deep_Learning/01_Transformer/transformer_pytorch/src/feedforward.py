import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        d_model : input/output dim.
        d_ff    : hidden dim. 
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. Linear + ReLu
        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        # 2. Linear
        x = self.linear2(x)
        return x
    
    