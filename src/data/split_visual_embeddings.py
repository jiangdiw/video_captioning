# src/data/split_visual_embeddings.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import shutil
from src.data.split_videos_simple import get_splits

# -------------------
# PATHS
# -------------------
DATA_ROOT     = Path("data")
VISUAL_ROOT   = DATA_ROOT / "processed" / "visual"
FUSED_DIR     = VISUAL_ROOT / "tascc_fused"          # where your .npy files are now
CLIP_ROOT     = VISUAL_ROOT / "clip_embedding"        # output
DINO_ROOT     = VISUAL_ROOT / "Dinov2_embedding"      # output

# Dimensions
CLIP_DIM = 512
DINO_DIM = 768
TOTAL_DIM = CLIP_DIM + DINO_DIM  # 1280

# -------------------
# Main
# -------------------
def main():
    train_files, val_files, test_files = get_splits()

    # Map video_id → split name
    split_map = {}
    for f in train_files:
        split_map[f.stem] = "train"
    for f in val_files:
        split_map[f.stem] = "val"
    for f in test_files:
        split_map[f.stem] = "test"

    # Create output directories
    for split in ["train", "val", "test"]:
        (CLIP_ROOT / split).mkdir(parents=True, exist_ok=True)
        (DINO_ROOT / split).mkdir(parents=True, exist_ok=True)

    # Track results
    ok      = 0
    missing = []
    bad_shape = []

    all_npy = sorted(FUSED_DIR.glob("*.npy"))
    print(f"Found {len(all_npy)} .npy files in {FUSED_DIR}\n")

    for npy_path in all_npy:
        vid_id = npy_path.stem   # e.g. "video2"

        # Check this video is in our 2500
        if vid_id not in split_map:
            print(f"  SKIP (not in our 2500 split): {vid_id}")
            continue

        split = split_map[vid_id]

        # Load fused embedding
        fused = np.load(str(npy_path))   # expected (40, 1280)

        # Validate shape
        if fused.ndim != 2 or fused.shape[1] != TOTAL_DIM:
            print(f"  BAD SHAPE: {vid_id}  got {fused.shape}  expected (40, {TOTAL_DIM})")
            bad_shape.append(vid_id)
            continue

        # Split into CLIP and DINOv2
        clip_emb = fused[:, :CLIP_DIM]    # (40, 512)
        dino_emb = fused[:, CLIP_DIM:]    # (40, 768)

        # Save
        np.save(str(CLIP_ROOT / split / f"{vid_id}.npy"), clip_emb)
        np.save(str(DINO_ROOT / split / f"{vid_id}.npy"), dino_emb)

        print(f"  OK [{split}]: {vid_id}  clip={clip_emb.shape}  dino={dino_emb.shape}")
        ok += 1

    # Check if any videos in our split have no corresponding .npy
    all_found = {p.stem for p in all_npy}
    for vid_id in split_map:
        if vid_id not in all_found:
            missing.append(vid_id)

    # Summary
    print(f"\n{'='*50}")
    print(f"Total processed : {ok}")
    print(f"Bad shape       : {len(bad_shape)}")
    print(f"Missing .npy    : {len(missing)}")

    # Count per split
    for split in ["train", "val", "test"]:
        n_clip = len(list((CLIP_ROOT / split).glob("*.npy")))
        n_dino = len(list((DINO_ROOT / split).glob("*.npy")))
        print(f"  {split:6s}  clip={n_clip}  dino={n_dino}")

    if missing:
        log = VISUAL_ROOT / "missing_visual.txt"
        log.write_text("\n".join(sorted(missing)))
        print(f"\nMissing IDs logged to {log}")

    if bad_shape:
        log = VISUAL_ROOT / "bad_shape_visual.txt"
        log.write_text("\n".join(sorted(bad_shape)))
        print(f"Bad shape IDs logged to {log}")

if __name__ == "__main__":
    main()