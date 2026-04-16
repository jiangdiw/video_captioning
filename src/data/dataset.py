# src/data/dataset.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.split_videos_simple import get_splits
from src.data.vocabulary import Vocabulary

DATA_ROOT = Path("data")

class MSRVTTDataset(Dataset):
    def __init__(self, split, vocab, debug_n=None):
        """
        split    : "train" | "val" | "test"
        vocab    : Vocabulary instance
        debug_n  : if set, only use first N videos (for quick testing)
        """
        self.split = split
        self.vocab = vocab

        self.clip_dir  = DATA_ROOT / "processed/visual/clip_embedding"   / split
        self.dino_dir  = DATA_ROOT / "processed/visual/Dinov2_embedding"  / split
        self.audio_dir = DATA_ROOT / "processed/audio/vggish_embeddings"  / split

        # Load captions
        cap_path = DATA_ROOT / "processed/captions" / f"{split}_captions.json"
        with open(cap_path) as f:
            self.captions = json.load(f)

        # Get split video IDs from single source of truth
        train_files, val_files, test_files = get_splits()
        split_files = {"train": train_files,
                       "val":   val_files,
                       "test":  test_files}[split]

        # Only keep videos with ALL three embeddings AND captions
        self.video_ids = []
        for f in split_files:
            vid = f.stem
            if (self.clip_dir  / f"{vid}.npy").exists() and \
               (self.dino_dir  / f"{vid}.npy").exists() and \
               (self.audio_dir / f"{vid}.npy").exists() and \
               vid in self.captions:
                self.video_ids.append(vid)

        # Debug mode: limit to N videos
        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            print(f"[{split}] DEBUG MODE: using {len(self.video_ids)} videos")
        else:
            print(f"[{split}] {len(self.video_ids)} videos ready "
                  f"(skipped {len(split_files) - len(self.video_ids)})")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]

        clip_emb  = torch.tensor(
            np.load(self.clip_dir  / f"{vid_id}.npy"),
            dtype=torch.float32)     # (40, 512)
        dino_emb  = torch.tensor(
            np.load(self.dino_dir  / f"{vid_id}.npy"),
            dtype=torch.float32)     # (40, 768)
        audio_emb = torch.tensor(
            np.load(self.audio_dir / f"{vid_id}.npy"),
            dtype=torch.float32)     # (T,  128)

        # Randomly pick one caption per iteration
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)

        return clip_emb, dino_emb, audio_emb, encoded, vid_id


def collate_fn(batch):
    clips, dinos, audios, captions, vid_ids = zip(*batch)

    clips = torch.stack(clips)    # (B, 40, 512)
    dinos = torch.stack(dinos)    # (B, 40, 768)

    # Pad audio to max T in batch
    max_t        = max(a.shape[0] for a in audios)
    audio_padded = torch.zeros(len(audios), max_t, 128)
    for i, a in enumerate(audios):
        audio_padded[i, :a.shape[0]] = a   # (B, max_T, 128)

    # Pad captions to max length in batch
    max_len     = max(c.shape[0] for c in captions)
    cap_padded  = torch.zeros(len(captions), max_len, dtype=torch.long)
    cap_lengths = []
    for i, c in enumerate(captions):
        cap_padded[i, :c.shape[0]] = c
        cap_lengths.append(c.shape[0])

    return (clips, dinos, audio_padded,
            cap_padded, torch.tensor(cap_lengths), list(vid_ids))