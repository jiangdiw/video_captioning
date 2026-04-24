import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.msrvtt import get_processed_layout, normalize_dataset_mode
from data.vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Build a vocabulary from MSR-VTT training captions.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--min-freq", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    train_caps = layout.captions_root / "train_captions.json"
    if not train_caps.exists():
        raise FileNotFoundError(
            f"Missing training captions for {dataset_mode}: {train_caps}. "
            f"Run data/extract_captions.py --dataset-mode {dataset_mode} first."
        )

    train_captions = json.loads(train_caps.read_text())
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build(train_captions)
    layout.captions_root.mkdir(parents=True, exist_ok=True)
    vocab.save(layout.vocab_path)

    test_cap = "a man is playing guitar outside"
    encoded = vocab.encode(test_cap)
    decoded = vocab.decode(encoded)
    print(f"\nDataset mode : {dataset_mode}")
    print("Test encode → decode:")
    print(f"  Original : {test_cap}")
    print(f"  Encoded  : {encoded}")
    print(f"  Decoded  : {decoded}")


if __name__ == "__main__":
    main()
