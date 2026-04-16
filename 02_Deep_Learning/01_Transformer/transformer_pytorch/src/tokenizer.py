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
        self.MASK_IDX = 4 #   [MASK] for BERT

    def train(self, input_file, model_prefix, vocab_size=16000):
        """
        Train SentencePiece BPE (Byte Pair Encoding) model on corpus
        input_file    : path to text file (one sentence per line)
        model_prefix  : prefix for saved model files
        vocab_size    : vocabulary size
        """
        print(f"--- Starting SentencePiece training for {model_prefix} ---")
        
        spm.SentencePieceTrainer.train(
            input=input_file,
            # sentence_iterator=iter(sentences),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=0.999, # higher for Korean
            model_type='bpe',
            pad_id=self.PAD_IDX,
            unk_id=self.UNK_IDX,
            bos_id=self.BOS_IDX,
            eos_id=self.EOS_IDX,
            user_defined_symbols=['[MASK]'],
            input_sentence_size=1000000,
            shuffle_input_sentence=True, 
            train_extremely_large_corpus=False,
            split_by_whitespace=True,
            byte_fallback=True,
            num_threads=4
        )
        self.sp.load(f'{model_prefix}.model')
        print(f"Tokenizer trained. Vocab size: {self.vocab_size()}")

    def load(self, model_path):
        """Load pre-trained tokenizer model"""
        self.sp.load(model_path)
        print(f"Tokenizer loaded. Vocab size: {self.vocab_size()}")

    def encode(self, text, add_special_tokens=True):
        """
        Text -> token indices
        ex) "나는 학교" -> [2, 23, 145, 3]
        """
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.BOS_IDX] + ids + [self.EOS_IDX]
        return ids
    
    def decode(self, ids):
        """
        Token indices -> text
        ex) [2, 23, 145, 3] -> "나는 학교"
        """
        # Remove special tokens
        ids = [i for i in ids if i not in
               [self.PAD_IDX, self.BOS_IDX, self.EOS_IDX]]
        return self.sp.decode(ids)
    
    def vocab_size(self):
        return self.sp.get_piece_size()