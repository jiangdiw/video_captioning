import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset

from data.msrvtt import get_processed_layout, get_split_video_ids, normalize_dataset_mode


def _resolve_layout(dataset_mode="subset", processed_root=None):
    dataset_mode = normalize_dataset_mode(dataset_mode)
    if processed_root is not None:
        processed_root = Path(processed_root)
        captions_root = processed_root / "captions"
        visual_root = processed_root / "visual"
        audio_root = processed_root / "audio"
        multimodal_root = processed_root / "multimodal"
        return {
            "dataset_mode": dataset_mode,
            "processed_root": processed_root,
            "captions_root": captions_root,
            "visual_root": visual_root,
            "audio_root": audio_root,
            "multimodal_root": multimodal_root,
        }
    layout = get_processed_layout(dataset_mode)
    return {
        "dataset_mode": dataset_mode,
        "processed_root": layout.processed_root,
        "captions_root": layout.captions_root,
        "visual_root": layout.visual_root,
        "audio_root": layout.audio_root,
        "multimodal_root": layout.multimodal_root,
    }


class PrecomputedDataset(Dataset):
    def __init__(self, split, vocab, variant="approach2", debug_n=None, dataset_mode="subset", processed_root=None):
        self.split = split
        self.vocab = vocab
        self.dataset_mode = normalize_dataset_mode(dataset_mode)

        variant_dirs = {
            "approach1": "approach1_visual_cross_attn_audio_concat",
            "approach2": "approach2_trimodal_cross_attn",
        }
        if variant not in variant_dirs:
            raise ValueError(f"Unsupported precomputed variant: {variant}")
        self.variant = variant

        layout = _resolve_layout(dataset_mode=self.dataset_mode, processed_root=processed_root)
        self.emb_dir = layout["multimodal_root"] / variant_dirs[variant] / split
        cap_path = layout["captions_root"] / f"{split}_captions.json"

        with open(cap_path) as f:
            self.captions = json.load(f)

        self.video_ids = []
        missing_caps = []
        for npy_path in sorted(self.emb_dir.glob("*.npy")):
            vid = npy_path.stem
            if vid in self.captions:
                self.video_ids.append(vid)
            else:
                missing_caps.append(vid)

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            print(f"[{split}/{variant}/{self.dataset_mode}] DEBUG MODE: using {len(self.video_ids)} videos")
        else:
            print(f"[{split}/{variant}/{self.dataset_mode}] {len(self.video_ids)} videos ready")

        if missing_caps:
            print(f"[{split}/{variant}/{self.dataset_mode}] WARNING: {len(missing_caps)} embeddings had no matching captions")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        emb = torch.tensor(np.load(self.emb_dir / f"{vid_id}.npy"), dtype=torch.float32)
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return emb, encoded, vid_id


class SequencePrecomputedDataset(Dataset):
    def __init__(self, split, vocab, variant="approach2", debug_n=None, dataset_mode="subset", processed_root=None):
        self.split = split
        self.vocab = vocab
        self.dataset_mode = normalize_dataset_mode(dataset_mode)

        variant_dirs = {
            "approach1": "approach1_visual_cross_attn_audio_concat_sequence",
            "approach2": "approach2_trimodal_cross_attn_sequence",
        }
        if variant not in variant_dirs:
            raise ValueError(f"Unsupported sequence-precomputed variant: {variant}")
        self.variant = variant

        layout = _resolve_layout(dataset_mode=self.dataset_mode, processed_root=processed_root)
        self.emb_dir = layout["multimodal_root"] / variant_dirs[variant] / split
        cap_path = layout["captions_root"] / f"{split}_captions.json"

        with open(cap_path) as f:
            self.captions = json.load(f)

        self.video_ids = []
        missing_caps = []
        for npz_path in sorted(self.emb_dir.glob("*.npz")):
            vid = npz_path.stem
            if vid in self.captions:
                self.video_ids.append(vid)
            else:
                missing_caps.append(vid)

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            print(f"[{split}/{variant}/sequence/{self.dataset_mode}] DEBUG MODE: using {len(self.video_ids)} videos")
        else:
            print(f"[{split}/{variant}/sequence/{self.dataset_mode}] {len(self.video_ids)} videos ready")

        if missing_caps:
            print(f"[{split}/{variant}/sequence/{self.dataset_mode}] WARNING: {len(missing_caps)} embeddings had no matching captions")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        payload = np.load(self.emb_dir / f"{vid_id}.npz")
        seq = torch.tensor(payload["seq"], dtype=torch.float32)
        pooled = torch.tensor(payload["pooled"], dtype=torch.float32)
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return seq, pooled, encoded, vid_id


class MSRVTTDataset(Dataset):
    def __init__(self, split, vocab, debug_n=None, dataset_mode="subset", processed_root=None):
        self.split = split
        self.vocab = vocab
        self.dataset_mode = normalize_dataset_mode(dataset_mode)

        layout = _resolve_layout(dataset_mode=self.dataset_mode, processed_root=processed_root)
        self.clip_dir = layout["visual_root"] / "clip_embedding" / split
        self.dino_dir = layout["visual_root"] / "Dinov2_embedding" / split
        self.audio_dir = layout["audio_root"] / "vggish_embeddings" / split

        cap_path = layout["captions_root"] / f"{split}_captions.json"
        with open(cap_path) as f:
            self.captions = json.load(f)

        split_ids = get_split_video_ids(self.dataset_mode)[split]
        self.video_ids = []
        for vid in split_ids:
            if (
                (self.clip_dir / f"{vid}.npy").exists()
                and (self.dino_dir / f"{vid}.npy").exists()
                and (self.audio_dir / f"{vid}.npy").exists()
                and vid in self.captions
            ):
                self.video_ids.append(vid)

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            print(f"[{split}/{self.dataset_mode}] DEBUG MODE: using {len(self.video_ids)} videos")
        else:
            print(f"[{split}/{self.dataset_mode}] {len(self.video_ids)} videos ready (skipped {len(split_ids) - len(self.video_ids)})")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid_id = self.video_ids[idx]
        clip_emb = torch.tensor(np.load(self.clip_dir / f"{vid_id}.npy"), dtype=torch.float32)
        dino_emb = torch.tensor(np.load(self.dino_dir / f"{vid_id}.npy"), dtype=torch.float32)
        audio_emb = torch.tensor(np.load(self.audio_dir / f"{vid_id}.npy"), dtype=torch.float32)
        caption = random.choice(self.captions[vid_id])
        encoded = torch.tensor(self.vocab.encode(caption), dtype=torch.long)
        return clip_emb, dino_emb, audio_emb, encoded, caption, vid_id


def collate_fn_precomputed(batch):
    embs, captions, vid_ids = zip(*batch)
    embs = torch.stack(embs)

    max_len = max(c.shape[0] for c in captions)
    cap_padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, caption in enumerate(captions):
        cap_padded[i, : caption.shape[0]] = caption

    return embs, cap_padded, list(vid_ids)


def collate_fn_sequence_precomputed(batch):
    seqs, pooleds, captions, vid_ids = zip(*batch)
    seqs = torch.stack(seqs)
    pooleds = torch.stack(pooleds)

    max_len = max(c.shape[0] for c in captions)
    cap_padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, caption in enumerate(captions):
        cap_padded[i, : caption.shape[0]] = caption

    return seqs, pooleds, cap_padded, list(vid_ids)


def collate_fn(batch):
    clips, dinos, audios, captions, raw_captions, vid_ids = zip(*batch)

    clips = torch.stack(clips)
    dinos = torch.stack(dinos)

    max_t = max(audio.shape[0] for audio in audios)
    audio_padded = torch.zeros(len(audios), max_t, 128)
    for i, audio in enumerate(audios):
        audio_padded[i, : audio.shape[0]] = audio

    max_len = max(caption.shape[0] for caption in captions)
    cap_padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, caption in enumerate(captions):
        cap_padded[i, : caption.shape[0]] = caption

    return clips, dinos, audio_padded, cap_padded, list(raw_captions), list(vid_ids)
