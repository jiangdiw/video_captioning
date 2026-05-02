"""
Generate captions for a few val videos and print them alongside ground truth.

Usage:
    python generate_bart.py
    python generate_bart.py --n 20 --num-beams 4
"""
import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import BartTokenizer

from data.msrvtt import get_processed_layout, get_split_video_ids, normalize_dataset_mode
from models.bart_captioning_model import BartCaptioningModel


def load_video(vid, clip_dir, dino_dir, audio_dir, device):
    clip  = torch.tensor(np.load(clip_dir  / f"{vid}.npy"), dtype=torch.float32).unsqueeze(0).to(device)
    dino  = torch.tensor(np.load(dino_dir  / f"{vid}.npy"), dtype=torch.float32).unsqueeze(0).to(device)
    ap = audio_dir / f"{vid}.npy"
    audio = torch.tensor(np.load(ap), dtype=torch.float32).unsqueeze(0).to(device) if ap.exists() else torch.zeros(1, 1, 128, device=device)
    return clip, dino, audio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/bart_checkpoints/best.pt")
    p.add_argument("--audio-dir",  default="vggish_embeddings")
    p.add_argument("--n",          type=int, default=10)
    p.add_argument("--num-beams",  type=int, default=4)
    p.add_argument("--dataset-mode", default="subset")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartCaptioningModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})\n")

    layout = get_processed_layout(args.dataset_mode)
    clip_dir  = layout.raw_clip_root / "val"
    dino_dir  = layout.raw_dino_root / "val"
    audio_dir = Path(args.audio_dir) / "val"

    with open(layout.captions_root / "val_captions.json") as f:
        captions = json.load(f)

    val_ids = get_split_video_ids(args.dataset_mode)["val"]
    available = [v for v in val_ids if (clip_dir / f"{v}.npy").exists() and v in captions]
    sample = random.sample(available, min(args.n, len(available)))

    for vid in sample:
        clip, dino, audio = load_video(vid, clip_dir, dino_dir, audio_dir, device)
        with torch.no_grad():
            token_ids = model.generate(clip, dino, audio, max_new_tokens=40, num_beams=args.num_beams)
        generated = tokenizer.decode(token_ids[0], skip_special_tokens=True)
        gt = captions[vid][:3]

        print(f"[{vid}]")
        print(f"  Generated : {generated}")
        for i, c in enumerate(gt):
            print(f"  GT {i+1}      : {c}")
        print()


if __name__ == "__main__":
    main()
