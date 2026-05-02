"""
BART captioning without audio — BART decoder frozen, trains encoder + proj + lm_head only.

Usage:
    python train_bart_no_audio_frozen_decoder.py --debug
    python train_bart_no_audio_frozen_decoder.py --epochs 20 --batch-size 16
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
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer

from data.msrvtt import get_processed_layout, get_split_video_ids, normalize_dataset_mode
from models.bart_captioning_model_no_audio import BartCaptioningModelNoAudio


def freeze_decoder(model):
    """Freeze BART encoder and decoder, keep lm_head + visual encoder + proj trainable."""
    for p in model.bart.model.encoder.parameters():
        p.requires_grad = False
    for p in model.bart.model.decoder.parameters():
        p.requires_grad = False


class BartMSRVTTDatasetNoAudio(Dataset):
    def __init__(self, split, tokenizer, dataset_mode="subset", debug_n=None):
        dataset_mode = normalize_dataset_mode(dataset_mode)
        layout = get_processed_layout(dataset_mode)

        self.clip_dir = layout.raw_clip_root / split
        self.dino_dir = layout.raw_dino_root / split

        with open(layout.captions_root / f"{split}_captions.json") as f:
            self.captions = json.load(f)

        split_ids = get_split_video_ids(dataset_mode)[split]
        self.video_ids = [
            vid for vid in split_ids
            if (self.clip_dir / f"{vid}.npy").exists()
            and (self.dino_dir / f"{vid}.npy").exists()
            and vid in self.captions
        ]

        if debug_n:
            self.video_ids = self.video_ids[:debug_n]
        print(f"[{split}] {len(self.video_ids)} videos ready")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        clip = torch.tensor(np.load(self.clip_dir / f"{vid}.npy"), dtype=torch.float32)
        dino = torch.tensor(np.load(self.dino_dir / f"{vid}.npy"), dtype=torch.float32)
        caption = random.choice(self.captions[vid])
        return clip, dino, caption


def collate_fn(batch, tokenizer, max_caption_len=40):
    clips, dinos, captions = zip(*batch)
    clips = torch.stack(clips)
    dinos = torch.stack(dinos)

    enc = tokenizer(list(captions), padding=True, truncation=True,
                    max_length=max_caption_len, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return clips, dinos, labels


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for clips, dinos, labels in loader:
        clips  = clips.to(device)
        dinos  = dinos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(clips, dinos, labels=labels)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for clips, dinos, labels in loader:
        clips  = clips.to(device)
        dinos  = dinos.to(device)
        labels = labels.to(device)
        out = model(clips, dinos, labels=labels)
        total_loss += out.loss.item()
    return total_loss / len(loader)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-mode", default="subset")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--debug",        action="store_true")
    p.add_argument("--out-dir",      default="outputs/bart_no_audio_frozen_decoder_checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    debug_n = 8 if args.debug else None

    train_ds = BartMSRVTTDatasetNoAudio("train", tokenizer, args.dataset_mode, debug_n=debug_n)
    val_ds   = BartMSRVTTDatasetNoAudio("val",   tokenizer, args.dataset_mode, debug_n=debug_n)

    def collate(b):
        return collate_fn(b, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = BartCaptioningModelNoAudio().to(device)
    freeze_decoder(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss}, ckpt)
            print(f"  → saved {ckpt}")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
