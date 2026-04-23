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

DATA_ROOT = Path("data")

# ================================================================
# Dataset 1: PrecomputedDataset
# For teammates who only have approach2 embeddings from GitHub
# ================================================================
class PrecomputedDataset(Dataset):
    def __init__(self, split, vocab, debug_n=None):
        self.split = split
        self.vocab = vocab

        self.emb_dir = (DATA_ROOT
                        / "processed"
                        / "multimodal"
                        / "approach2_trimodal_cross_attn"
                        / split)

        cap_path = DATA_ROOT / "processed" / "captions" / f"{split}_captions.json"
        with open(cap_path) as f:
            self.captions = json.load(f)

        self.video_ids = []
        missing_caps   = []

        for npy_path in sorted(self.emb_dir.glob("*.npy")):
            vid = npy_path.stem
            if vid in self.captions:
                self.video_ids.append(vid)
            else:
                missing_caps.append(vid)

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            print(f"[{split}] DEBUG MODE: using {len(self.video_ids)} videos")
        else:
            print(f"[{split}] {len(self.video_ids)} videos ready")

        if missing_caps:
            print(f"[{split}] WARNING: {len(missing_caps)} embeddings "
                  f"had no matching captions")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id  = self.video_ids[idx]
        emb     = torch.tensor(
            np.load(self.emb_dir / f"{vid_id}.npy"),
            dtype=torch.float32)
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return emb, encoded, vid_id


# ================================================================
# Dataset 2: MSRVTTDataset
# For end-to-end training with raw clip/dino/audio embeddings
# __getitem__ returns 6 values including raw caption string
# ================================================================
class MSRVTTDataset(Dataset):
    def __init__(self, split, vocab, debug_n=None):
        self.split = split
        self.vocab = vocab

        self.clip_dir  = DATA_ROOT / "processed/visual/clip_embedding"  / split
        self.dino_dir  = DATA_ROOT / "processed/visual/Dinov2_embedding" / split
        self.audio_dir = DATA_ROOT / "processed/audio/vggish_embeddings" / split

        cap_path = DATA_ROOT / "processed/captions" / f"{split}_captions.json"
        with open(cap_path) as f:
            self.captions = json.load(f)

        train_files, val_files, test_files = get_splits()
        split_files = {"train": train_files,
                       "val":   val_files,
                       "test":  test_files}[split]

        self.video_ids = []
        for f in split_files:
            vid = f.stem
            if (self.clip_dir  / f"{vid}.npy").exists() and \
               (self.dino_dir  / f"{vid}.npy").exists() and \
               (self.audio_dir / f"{vid}.npy").exists() and \
               vid in self.captions:
                self.video_ids.append(vid)

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
            dtype=torch.float32)   # (40, 512)
        dino_emb  = torch.tensor(
            np.load(self.dino_dir  / f"{vid_id}.npy"),
            dtype=torch.float32)   # (40, 768)
        audio_emb = torch.tensor(
            np.load(self.audio_dir / f"{vid_id}.npy"),
            dtype=torch.float32)   # (T,  128)

        # randomly pick one caption
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)

        # return 6 values: embeddings + encoded + raw string + video id
        return clip_emb, dino_emb, audio_emb, encoded, caption, vid_id


# ================================================================
# Collate functions
# ================================================================
def collate_fn_precomputed(batch):
    """For PrecomputedDataset — returns 3 values"""
    embs, captions, vid_ids = zip(*batch)
    embs = torch.stack(embs)   # (B, 512)

    max_len    = max(c.shape[0] for c in captions)
    cap_padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, c in enumerate(captions):
        cap_padded[i, :c.shape[0]] = c

    return embs, cap_padded, list(vid_ids)


def collate_fn(batch):
    """
    For MSRVTTDataset — returns 6 values:
      clips, dinos, audios, cap_padded, raw_captions, vid_ids
    """
    clips, dinos, audios, captions, raw_captions, vid_ids = zip(*batch)

    clips = torch.stack(clips)   # (B, 40, 512)
    dinos = torch.stack(dinos)   # (B, 40, 768)

    # Pad audio to max T in batch
    max_t        = max(a.shape[0] for a in audios)
    audio_padded = torch.zeros(len(audios), max_t, 128)
    for i, a in enumerate(audios):
        audio_padded[i, :a.shape[0]] = a   # (B, max_T, 128)

    # Pad captions to max length in batch
    max_len    = max(c.shape[0] for c in captions)
    cap_padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, c in enumerate(captions):
        cap_padded[i, :c.shape[0]] = c

    # returns 6 values
    return (clips, dinos, audio_padded,
            cap_padded, list(raw_captions), list(vid_ids))