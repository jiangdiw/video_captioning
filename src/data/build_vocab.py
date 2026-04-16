# src/data/build_vocab.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.data.vocabulary import Vocabulary

DATA_ROOT  = Path("data")
TRAIN_CAPS = DATA_ROOT / "processed/captions/train_captions.json"
VOCAB_PATH = DATA_ROOT / "processed/captions/vocabulary.json"

def main():
    # Build vocab from training captions only
    with open(TRAIN_CAPS) as f:
        train_captions = json.load(f)

    vocab = Vocabulary(min_freq=2)
    vocab.build(train_captions)
    vocab.save(VOCAB_PATH)

    # Quick check
    test_cap = "a man is playing guitar outside"
    encoded  = vocab.encode(test_cap)
    decoded  = vocab.decode(encoded)
    print(f"\nTest encode → decode:")
    print(f"  Original : {test_cap}")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : {decoded}")

if __name__ == "__main__":
    main()