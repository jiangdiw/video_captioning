import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from data.msrvtt import get_processed_layout, normalize_dataset_mode
from data.split_videos_simple import get_splits
from models.multimodal_encoder import Approach1Encoder, Approach2Encoder


def load_embeddings(vid_id, split, layout):
    clip_path = layout.raw_clip_root / split / f"{vid_id}.npy"
    dino_path = layout.raw_dino_root / split / f"{vid_id}.npy"
    audio_path = layout.raw_audio_root / split / f"{vid_id}.npy"

    missing = [str(path) for path in [clip_path, dino_path, audio_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    clip_t = torch.tensor(np.load(str(clip_path)), dtype=torch.float32).unsqueeze(0)
    dino_t = torch.tensor(np.load(str(dino_path)), dtype=torch.float32).unsqueeze(0)
    audio_t = torch.tensor(np.load(str(audio_path)), dtype=torch.float32).unsqueeze(0)
    return clip_t, dino_t, audio_t


def process_split(model1, model2, files, split, layout):
    out1_dir = layout.approach1_root / split
    out2_dir = layout.approach2_root / split
    out1_seq_dir = layout.approach1_seq_root / split
    out2_seq_dir = layout.approach2_seq_root / split
    out1_dir.mkdir(parents=True, exist_ok=True)
    out2_dir.mkdir(parents=True, exist_ok=True)
    out1_seq_dir.mkdir(parents=True, exist_ok=True)
    out2_seq_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    failed = []

    for video_path in files:
        vid_id = video_path.stem
        out1 = out1_dir / f"{vid_id}.npy"
        out2 = out2_dir / f"{vid_id}.npy"
        out1_seq = out1_seq_dir / f"{vid_id}.npz"
        out2_seq = out2_seq_dir / f"{vid_id}.npz"

        if out1.exists() and out2.exists() and out1_seq.exists() and out2_seq.exists():
            print(f"  SKIP (exists): {vid_id}")
            continue

        try:
            clip_t, dino_t, audio_t = load_embeddings(vid_id, split, layout)
            with torch.no_grad():
                seq1, emb1 = model1(clip_t, dino_t, audio_t)
                seq2, emb2 = model2(clip_t, dino_t, audio_t)

            pooled1 = emb1.squeeze(0).numpy().astype(np.float32)
            pooled2 = emb2.squeeze(0).numpy().astype(np.float32)
            seq1_np = seq1.squeeze(0).numpy().astype(np.float32)
            seq2_np = seq2.squeeze(0).numpy().astype(np.float32)

            if not out1.exists():
                np.save(str(out1), pooled1)
            if not out2.exists():
                np.save(str(out2), pooled2)
            if not out1_seq.exists():
                np.savez_compressed(str(out1_seq), seq=seq1_np, pooled=pooled1)
            if not out2_seq.exists():
                np.savez_compressed(str(out2_seq), seq=seq2_np, pooled=pooled2)

            print(
                f"  OK [{split}]: {vid_id}  "
                f"approach1_seq={seq1.shape} pooled={emb1.shape}  "
                f"approach2_seq={seq2.shape} pooled={emb2.shape}"
            )
            ok += 1
        except Exception as error:
            print(f"  FAILED: {vid_id}  error={error}")
            failed.append(vid_id)

    print(f"\n[{split}] Done ✓  ok={ok}  failed={len(failed)}\n")
    return failed


def parse_args():
    parser = argparse.ArgumentParser(description="Build multimodal encoder outputs for MSR-VTT.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--video-dir", action="append", default=[], help="Optional additional directory to search for video*.mp4 files.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some expected videos are not present locally.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    model1 = Approach1Encoder().eval()
    model2 = Approach2Encoder().eval()

    print(f"dataset_mode: {dataset_mode}")
    print(f"Approach 1 output dim : {model1.init_dim}")
    print(f"Approach 2 output dim : {model2.init_dim}\n")

    train_files, val_files, test_files = get_splits(
        dataset_mode=dataset_mode,
        strict=not args.allow_missing,
        extra_video_dirs=args.video_dir,
    )

    all_failed = []
    for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        print(f"{'=' * 50}")
        print(f"Processing {split} ({len(files)} videos)")
        print(f"{'=' * 50}")
        all_failed.extend(process_split(model1, model2, files, split, layout))

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    for split in ["train", "val", "test"]:
        n1 = len(list((layout.approach1_root / split).glob("*.npy")))
        n2 = len(list((layout.approach2_root / split).glob("*.npy")))
        n1_seq = len(list((layout.approach1_seq_root / split).glob("*.npz")))
        n2_seq = len(list((layout.approach2_seq_root / split).glob("*.npz")))
        match = "✓" if n1 == n2 == n1_seq == n2_seq else "MISMATCH"
        print(
            f"  {split:6s}  approach1={n1}  approach2={n2}  "
            f"approach1_seq={n1_seq}  approach2_seq={n2_seq}  {match}"
        )

    if all_failed:
        log = layout.multimodal_root / "failed_multimodal.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(all_failed))
        print(f"\nFailed IDs → {log}")


if __name__ == "__main__":
    main()
