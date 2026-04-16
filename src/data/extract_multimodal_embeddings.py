# src/data/extract_multimodal_embeddings.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from src.data.split_videos_simple import get_splits
from src.models.multimodal_encoder import Approach1Encoder, Approach2Encoder

# -------------------
# PATHS
# -------------------
DATA_ROOT    = Path("data/processed")
CLIP_ROOT    = DATA_ROOT / "visual"  / "clip_embedding"
DINO_ROOT    = DATA_ROOT / "visual"  / "Dinov2_embedding"
AUDIO_ROOT   = DATA_ROOT / "audio"   / "vggish_embeddings"
APPROACH1_ROOT = DATA_ROOT / "multimodal" / "approach1_visual_cross_attn_audio_concat"
APPROACH2_ROOT = DATA_ROOT / "multimodal" / "approach2_trimodal_cross_attn"

# -------------------
# Load embeddings for one video
# -------------------
def load_embeddings(vid_id, split):
    clip_path  = CLIP_ROOT  / split / f"{vid_id}.npy"
    dino_path  = DINO_ROOT  / split / f"{vid_id}.npy"
    audio_path = AUDIO_ROOT / split / f"{vid_id}.npy"

    missing = []
    for p in [clip_path, dino_path, audio_path]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    clip_emb  = np.load(str(clip_path))    # (40, 512)
    dino_emb  = np.load(str(dino_path))    # (40, 768)
    audio_emb = np.load(str(audio_path))   # (T,  128)

    # Convert to torch tensors with batch dim
    clip_t  = torch.tensor(clip_emb,  dtype=torch.float32).unsqueeze(0)   # (1, 40, 512)
    dino_t  = torch.tensor(dino_emb,  dtype=torch.float32).unsqueeze(0)   # (1, 40, 768)
    audio_t = torch.tensor(audio_emb, dtype=torch.float32).unsqueeze(0)   # (1,  T, 128)

    return clip_t, dino_t, audio_t

# -------------------
# Process one split for both approaches
# -------------------
def process_split(model1, model2, files, split):
    out1_dir = APPROACH1_ROOT / split
    out2_dir = APPROACH2_ROOT / split
    out1_dir.mkdir(parents=True, exist_ok=True)
    out2_dir.mkdir(parents=True, exist_ok=True)

    ok     = 0
    failed = []

    for video_path in files:
        vid_id = video_path.stem
        out1   = out1_dir / f"{vid_id}.npy"
        out2   = out2_dir / f"{vid_id}.npy"

        # Skip if both already done
        if out1.exists() and out2.exists():
            print(f"  SKIP (exists): {vid_id}")
            continue

        try:
            clip_t, dino_t, audio_t = load_embeddings(vid_id, split)

            with torch.no_grad():
                emb1 = model1(clip_t, dino_t, audio_t)   # (1, 640)
                emb2 = model2(clip_t, dino_t, audio_t)   # (1, 512)

            np.save(str(out1), emb1.squeeze(0).numpy().astype(np.float32))
            np.save(str(out2), emb2.squeeze(0).numpy().astype(np.float32))

            print(f"  OK [{split}]: {vid_id}  "
                  f"approach1={emb1.shape}  approach2={emb2.shape}")
            ok += 1

        except Exception as e:
            print(f"  FAILED: {vid_id}  error={e}")
            failed.append(vid_id)

    print(f"\n[{split}] Done ✓  ok={ok}  failed={len(failed)}\n")
    return failed

# -------------------
# Main
# -------------------
def main():
    # Initialize both models in eval mode (no gradient needed)
    model1 = Approach1Encoder().eval()
    model2 = Approach2Encoder().eval()

    print(f"Approach 1 output dim : {model1.output_dim}")   # 640
    print(f"Approach 2 output dim : {model2.output_dim}\n") # 512

    train_files, val_files, test_files = get_splits()

    all_failed = []
    for split, files in [("train", train_files),
                         ("val",   val_files),
                         ("test",  test_files)]:
        print(f"{'='*50}")
        print(f"Processing {split} ({len(files)} videos)")
        print(f"{'='*50}")
        failed = process_split(model1, model2, files, split)
        all_failed.extend(failed)

    # Final summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    for split in ["train", "val", "test"]:
        n1 = len(list((APPROACH1_ROOT / split).glob("*.npy")))
        n2 = len(list((APPROACH2_ROOT / split).glob("*.npy")))
        match = "✓" if n1 == n2 else "⚠ MISMATCH"
        print(f"  {split:6s}  approach1={n1}  approach2={n2}  {match}")

    if all_failed:
        log = DATA_ROOT / "multimodal" / "failed_multimodal.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("\n".join(all_failed))
        print(f"\nFailed IDs → {log}")

if __name__ == "__main__":
    main()