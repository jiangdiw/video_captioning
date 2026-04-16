# src/data/vocabulary.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from collections import Counter

class Vocabulary:
    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq=2):
        self.min_freq  = min_freq
        self.word2idx  = {}
        self.idx2word  = {}
        for token in [self.PAD_TOKEN, self.SOS_TOKEN,
                      self.EOS_TOKEN, self.UNK_TOKEN]:
            self._add_word(token)

    def _add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    @staticmethod
    def tokenize(text):
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build(self, captions_dict):
        """captions_dict: {video_id: [cap1, cap2, ...]}"""
        counter = Counter()
        for caps in captions_dict.values():
            for cap in caps:
                counter.update(self.tokenize(cap))
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self._add_word(word)
        print(f"Vocabulary size: {len(self.word2idx)} "
              f"(min_freq={self.min_freq})")

    def encode(self, caption):
        tokens = self.tokenize(caption)
        return (
            [self.word2idx[self.SOS_TOKEN]] +
            [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN])
             for t in tokens] +
            [self.word2idx[self.EOS_TOKEN]]
        )

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(int(idx), self.UNK_TOKEN)
            if word == self.EOS_TOKEN:
                break
            if word not in [self.PAD_TOKEN, self.SOS_TOKEN]:
                words.append(word)
        return " ".join(words)

    @property
    def pad_idx(self): return self.word2idx[self.PAD_TOKEN]
    @property
    def sos_idx(self): return self.word2idx[self.SOS_TOKEN]
    @property
    def eos_idx(self): return self.word2idx[self.EOS_TOKEN]
    def __len__(self):  return len(self.word2idx)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "word2idx" : self.word2idx,
                "idx2word" : {str(k): v for k, v in self.idx2word.items()},
                "min_freq" : self.min_freq
            }, f, indent=2)
        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        print(f"Vocabulary loaded: {len(vocab)} words")
        return vocab