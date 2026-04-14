import sentencepiece as spm
import os

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()

        # Special tokens
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.BOS_IDX = 2  # Beginning of sentence [START]
        self.EOS_IDX = 3  # End of sentence [END]

    def train(self, input_file, model_prefix, vocab_size=16000):
        """
        Train SentencePiece BPE model on corpus
        input_file    : path to text file (one sentence per line)
        model_prefix  : prefix for saved model files
        vocab_size    : vocabulary size
        """
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.9995, # higher for Korean
            model_type='bpe',
            pad_id=self.PAD_IDX,
            unk_id=self.UNK_IDX,
            bos_id=self.BOS_IDX,
            eos_id=self.EOS_IDX
        )
        self.sp.load(f'{model_prefix}.model')
        print(f"Tokenizer trained. Vocab size: {self.vocab_size()}")

    def laod(self, model_path):
        """Load pre-trained tokenizer model"""
        self.sp.load(model_path)
        print(f"Tokenizer loadeed. Vocab size: {self.vocab_size}")

    def encode(self, text, add_special_tokens=True):
        """
        Text -> token indices
        ex) "나는 학교" -> [2, 23, 145, 3]
        """
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.BOS_IDX] + ids + [self.EOS_IDX]
        return ids
    
    def decode(self, ins):
        """
        Token indices -> text
        ex) [2, 23, 145, 3] -> "나는 학교"
        """
        # Remove special tokens
        ids = [i for i in ids if i not in
               [self.PAD_IDX, self.BOX_IDX, self.EOS_IDX]]
        return self.sp.decode(ids)
    
    def vocab_size(self):
        return self.sp.get_piece_size()