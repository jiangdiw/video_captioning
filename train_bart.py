"""
Legacy quick-start training script for the older BART path.

This script now covers all four old variants:
- audio + trainable BART
- audio + frozen BART
- no-audio + trainable BART
- no-audio + frozen BART

Use `--no-audio` and `--freeze-bart` instead of separate duplicate scripts.
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

from data.msrvtt import get_processed_layout, get_split_video_ids_from_captions, normalize_dataset_mode
from models.bart_captioning_model import BartCaptioningModel
from models.bart_captioning_model_no_audio import BartCaptioningModelNoAudio


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BartMSRVTTDataset(Dataset):
    """Loads raw CLIP / DINOv2 features and optional VGGish features."""

    def __init__(self, split, tokenizer, dataset_mode="subset",
                 audio_dir=None, max_caption_len=40, debug_n=None, with_audio=True):
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        self.with_audio = with_audio

        dataset_mode = normalize_dataset_mode(dataset_mode)
        layout = get_processed_layout(dataset_mode)

        self.clip_dir  = layout.raw_clip_root / split
        self.dino_dir  = layout.raw_dino_root / split
        self.audio_dir = Path(audio_dir) / split if audio_dir else layout.raw_audio_root / split

        cap_path = layout.captions_root / f"{split}_captions.json"
        with open(cap_path) as f:
            self.captions = json.load(f)

        split_ids = get_split_video_ids_from_captions(layout.captions_root)[split]
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
        clip  = torch.tensor(np.load(self.clip_dir / f"{vid}.npy"), dtype=torch.float32)
        dino  = torch.tensor(np.load(self.dino_dir / f"{vid}.npy"), dtype=torch.float32)
        caption = random.choice(self.captions[vid])
        if not self.with_audio:
            return clip, dino, caption
        audio_path = self.audio_dir / f"{vid}.npy"
        audio = torch.tensor(np.load(audio_path), dtype=torch.float32) if audio_path.exists() else torch.zeros(1, 128)
        return clip, dino, audio, caption


def collate_fn_with_audio(batch, tokenizer, max_caption_len):
    clips, dinos, audios, captions = zip(*batch)
    clips = torch.stack(clips)   # (B, 40, 512)
    dinos = torch.stack(dinos)   # (B, 40, 768)

    # Pad audio to longest in batch
    max_t = max(a.shape[0] for a in audios)
    audio_pad = torch.zeros(len(audios), max_t, audios[0].shape[-1])
    audio_mask = torch.zeros(len(audios), max_t, dtype=torch.long)
    for i, a in enumerate(audios):
        audio_pad[i, :a.shape[0]] = a
        audio_mask[i, :a.shape[0]] = 1

    # Tokenize captions; BART will auto-create decoder_input_ids via shift_tokens_right
    enc = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=max_caption_len,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return clips, dinos, audio_pad, audio_mask, labels


def collate_fn_no_audio(batch, tokenizer, max_caption_len):
    clips, dinos, captions = zip(*batch)
    clips = torch.stack(clips)
    dinos = torch.stack(dinos)

    enc = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=max_caption_len,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return clips, dinos, labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, with_audio):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if with_audio:
            clips, dinos, audios, audio_mask, labels = batch
            clips  = clips.to(device)
            dinos  = dinos.to(device)
            audios = audios.to(device)
            audio_mask = audio_mask.to(device)
            labels = labels.to(device)
        else:
            clips, dinos, labels = batch
            clips = clips.to(device)
            dinos = dinos.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()
        if with_audio:
            out = model(clips, dinos, audios, audio_mask=audio_mask, decoder_input_ids=None, labels=labels)
        else:
            out = model(clips, dinos, decoder_input_ids=None, labels=labels)
        out.loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out.loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, with_audio):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        if with_audio:
            clips, dinos, audios, audio_mask, labels = batch
            clips  = clips.to(device)
            dinos  = dinos.to(device)
            audios = audios.to(device)
            audio_mask = audio_mask.to(device)
            labels = labels.to(device)
            out = model(clips, dinos, audios, audio_mask=audio_mask, decoder_input_ids=None, labels=labels)
        else:
            clips, dinos, labels = batch
            clips = clips.to(device)
            dinos = dinos.to(device)
            labels = labels.to(device)
            out = model(clips, dinos, decoder_input_ids=None, labels=labels)
        total_loss += out.loss.item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-mode", default="subset")
    p.add_argument("--audio-dir",    default="vggish_embeddings",
                   help="Root folder containing train/val/test vggish .npy files")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--freeze-bart",  action="store_true")
    p.add_argument("--no-audio",     action="store_true")
    p.add_argument("--debug",        action="store_true",
                   help="Use 8 videos per split for a quick sanity check")
    p.add_argument("--out-dir",      default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    with_audio = not args.no_audio

    debug_n = 8 if args.debug else None

    train_ds = BartMSRVTTDataset("train", tokenizer, args.dataset_mode,
                                  audio_dir=args.audio_dir, debug_n=debug_n, with_audio=with_audio)
    val_ds   = BartMSRVTTDataset("val",   tokenizer, args.dataset_mode,
                                  audio_dir=args.audio_dir, debug_n=debug_n, with_audio=with_audio)

    def collate(b):
        if with_audio:
            return collate_fn_with_audio(b, tokenizer, max_caption_len=40)
        return collate_fn_no_audio(b, tokenizer, max_caption_len=40)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate, num_workers=0)

    model_cls = BartCaptioningModel if with_audio else BartCaptioningModelNoAudio
    model = model_cls(freeze_bart=args.freeze_bart).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    if args.out_dir is None:
        if with_audio:
            args.out_dir = (
                "outputs/bart_frozen_decoder_checkpoints"
                if args.freeze_bart else "outputs/bart_checkpoints"
            )
        else:
            args.out_dir = (
                "outputs/bart_no_audio_frozen_decoder_checkpoints"
                if args.freeze_bart else "outputs/bart_no_audio_checkpoints"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, with_audio=with_audio)
        val_loss   = evaluate(model, val_loader, device, with_audio=with_audio)
        print(f"Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = out_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss}, ckpt)
            print(f"  → saved {ckpt}")

    print(f"\nDone. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
