import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_seq_len=128):
        """
        data           :  HuggingFace dataset split (train/val/test)
        src_tokenizer  :  Korean tokenizer
        tgt_tokenizer  :  Engilsh tokenizer
        max_seq_len    :  maximum sequence length (truncate if longer)
        """
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text = self.data[idx]['translation']['ko']
        tgt_text = self.data[idx]['translation']['en']

        # Tokenize (includes BOS/EOS)
        src_ids = self.src_tokenizer.encode(src_text)[:self.max_seq_len]
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)[:self.max_seq_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
def collate_fn(batch, src_pad_idx=0, tgt_pad_idx=0):
    """
    Collate function for DataLoader:
        - Pads sequences to the same length within a batch
        - Generates padding masks
    """
    src_batch, tgt_batch = zip(*batch)

    # Pad sequences to max length in batch
    src_padded = pad_sequence(src_batch, src_pad_idx) # (batch, src_seq_len)
    tgt_padded = pad_sequence(tgt_batch, tgt_pad_idx) # (batch, tgt_seq_len)

    # Padding masks
    # (batch, 1, 1, seq_len) - True where NOT padding
    src_mask = (src_padded != src_pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt_padded != tgt_pad_idx).unsqueeze(1).unsqueeze(2)

    return src_padded, tgt_padded, src_mask, tgt_mask

def pad_sequence(sequences, pad_idx):
    """
    Pad a list of tensors to the same length.
    sequences  :  list of 1D tensors (each is one sentence)
    pad_idx    :  index used for padding ([PAD] token)
    """
    max_len = max(seq.size(0) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq
    return padded

def get_dataloader(data, src_tokenizer, tgt_tokenizer,
                   batch_size=64, max_seq_len=17, shuffle=True):
    """
    Returns a DataLoader for the given dataset split
    """
    dataset = TranslationDataset(data, src_tokenizer, tgt_tokenizer, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn = lambda b: collate_fn(b, src_tokenizer.PAD_IDX, tgt_tokenizer.PAD_IDX)
    )