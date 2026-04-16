import torch
import random
from torch.utils.data import Dataset, DataLoader

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=512):
        """
        Dataset for BERT pre-training (MLM + NSP)

        Args:
            data         :  HuggingFace dataset split
            tokenizer    :  trained SentencePiece tokenizer
            max_seq_len  :  maximum sequence length (BERT: 512)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get sentence A
        sentence_a = self.data[idx]['translation']['en']

        # NSP: 50% real next sentence, 50% random sentence
        tokens, segment_ids, nsp_label = self.create_nsp_sample(sentence_a, idx)

        # MLM: randomly mask 15% of tokens
        tokens, mlm_labels = self.create_mlm_sample(tokens)

        return (torch.tensor(tokens),
                torch.tensor(segment_ids),
                torch.tensor(mlm_labels),
                torch.tensor(nsp_label)
                )
    
    def create_nsp_sample(self, sentence_a, idx):
        """
        Create NSP sample:
            50% -> real next sentence (label = 1)
            50% -> random next sentence (label = 0)

        Returns:
            tokens       :  [CLS] sentence_A [SEP] sentence_B [SEP]
            segment_ids  :  0 for sentence_A, 1 for sentence_B
            nsp_label    :  1 if real next, 0 if random
        """
        # Encode sentence A (BOS=[CLS], EOS=[SEP] auto added)
        ids_a = self.tokenizer.encode(sentence_a)[:self.max_seq_len // 2]

        if random.random() > 0.5:
            # Real next sentence
            next_idx = (idx + 1) % len(self.data)
            sentence_b = self.data[next_idx]['translation']['en']
            nsp_label = 1
        else:
            # Random sentence
            rand_idx = random.randint(0, len(self.date) - 1)
            sentence_b = self.data[rand_idx]['translation']['en']
            nsp_label = 0
        
        # Encode sentence B - remove leading [CLS] to avoid duplicate
        ids_b = self.tokenizer.encode(sentence_b)[:self.max_seq_len // 2]
        ids_b = ids_b[1:] # remove [CLS]

        # Combine: [CLS] sentence_A [SEP] sentence_B [SEP]
        tokens = ids_a + ids_b

        # Truncate to max_seq_len
        tokens = tokens[:self.max_seq_len]

        # Segment ids: 0 for sentence_A, 1 for sentence_B
        segment_ids = (
            [0] * len(ids_a) +
            [1] * len(ids_b)
        )[:self.max_seq_len]

        return tokens, segment_ids, nsp_label
    
    def create_mlm_sample(self, tokens):
        """
        Create MLM sample:
        - Randomly select 15% of tokens
        - 80% -> replace with [MASK]
        - 10% -> replace with random token
        - 10% -> keep original token
        - Never mask [CLS], [SEP], [PAD]

        Returns:
            tokens      :  masked token sequence
            mlm_labels  :  original tokens at masked positions, -100 elsewhere
        """
        tokens = tokens.copy()
        mlm_labels = [-100] * len(tokens) # -100 means ignore in loss

        # Special tokens that should not be masked
        special_tokens = {
            self.tokenizer.PAD_IDX,
            self.tokenizer.BOS_IDX, # [CLS]
            self.tokenizer.EOS_IDX  # [SEP]
        }

        # Candidate positions for masking (exclude special tokens)
        candidates = [i for i, t in enumerate(tokens)
                      if t not in special_tokens]
        
        # Select 15% of candidates
        num_to_mask = max(1, int(len(candidates) * 0.15))
        masked_positions = random.sample(candidates, num_to_mask)

        for pos in masked_positions:
            original_token = tokens[pos]
            mlm_labels[pos] = original_token # save original for loss

            prob = random.random()
            # 80%: replace with [MASK]
            if prob < 0.8:
                tokens[pos] = self.tokenizer.MASK_IDX
            # 10%: replace with random token
            elif prob < 0.9:
                tokens[pos] = random.randint(5, self.tokenizer.vocab_size() - 1)
            # 10% keep original token

        return tokens, mlm_labels
    
    def collate_fn(batch, pad_idx=0):
        """
        Collate function:
        - Pad sequences to max length in batch
        - Generate padding mask
        - MLM labels padded with -100 (ignored in loss)
        """
        tokens_batch, segment_batch, mlm_batch, nsp_batch = zip(*batch)

        # Pad to max length in batch
        max_len = max(t.size(0) for t in tokens_batch)

        tokens_padded = torch.full((len(tokens_batch), max_len), pad_idx, dtype=torch.long)
        segment_padded = torch.full((len(tokens_batch), max_len), 0, dtype=torch.long)
        mlm_padded = torch.full((len(tokens_batch), max_len), -100, dtype=torch.long)

        for i, (tokens, segment, mlm) in enumerate(zip(tokens_batch, segment_batch, mlm_batch)):
            tokens_padded[i, :tokens.size(0)] = tokens
            segment_padded[i, :segment.size(0)] = segment
            mlm_padded[i, :mlm.size(0)] = mlm

        # Padding mask: (batch, 1, 1, seq_len) - True where NOT padding
        mask = (tokens_padded != pad_idx).unsqueeze(1).unsqueeze(2)

        nsp_labels = torch.tensor(nsp_batch, dtype=torch.long)

        return tokens_padded, segment_padded, mlm_padded, nsp_labels, mask
    
    def get_bert_dataloader(data, tokenizer, batch_size=32,
                            max_seq_len=512, shuffle=True):
        """
        Returns DataLoader for BERT pre-training
        """
        dataset = BERTDataset(data, tokenizer, max_seq_len)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer.PAD_IDX)
        )
