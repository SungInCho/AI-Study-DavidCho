import numpy as np

def positional_encoding(seq_len, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model // 2):
            denom = 10000 ** (2 * i / d_model)
            PE[pos, 2*i] = np.sin(pos / denom)
            PE[pos, 2*i + 1] = np.cos(pos / denom)
    
    return PE