import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.msrvtt import get_processed_layout, get_split_video_ids, normalize_dataset_mode


CLIP_DIM = 512
DINO_DIM = 768
TOTAL_DIM = CLIP_DIM + DINO_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Split TASCC fused features into CLIP and DINO branches.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--fused-dir", action="append", default=[], help="Directory containing fused TASCC features. Can be provided multiple times.")
    return parser.parse_args()


def candidate_fused_dirs():
    return [
        PROJECT_ROOT / "data" / "processed" / "visual" / "tascc_fused",
        PROJECT_ROOT / "features" / "tascc_fused",
        PROJECT_ROOT / "datas" / "feats" / "tascc_fused",
    ]


def resolve_fused_dirs(explicit_dirs=None):
    dirs = [Path(path) for path in (explicit_dirs or [])]
    if dirs:
        missing = [path for path in dirs if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing fused feature directories: " + ", ".join(str(path) for path in missing))
        return dirs
    existing = [path for path in candidate_fused_dirs() if path.exists()]
    if existing:
        return existing
    raise FileNotFoundError(
        "Could not find TASCC fused features. Expected one of: "
        + ", ".join(str(path) for path in candidate_fused_dirs())
    )


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    fused_dirs = resolve_fused_dirs(args.fused_dir)

    split_ids = get_split_video_ids(dataset_mode)
    split_map = {
        video_id: split
        for split, video_ids in split_ids.items()
        for video_id in video_ids
    }

    for split in ["train", "val", "test"]:
        (layout.raw_clip_root / split).mkdir(parents=True, exist_ok=True)
        (layout.raw_dino_root / split).mkdir(parents=True, exist_ok=True)

    ok = 0
    missing = []
    bad_shape = []

    fused_index = {}
    for fused_dir in fused_dirs:
        for path in sorted(fused_dir.glob("*.npy")):
            fused_index.setdefault(path.stem, path)
    all_npy = [fused_index[video_id] for video_id in sorted(fused_index.keys())]
    print(f"dataset_mode: {dataset_mode}")
    print(f"Using fused dirs: {', '.join(str(path) for path in fused_dirs)}")
    print(f"Found {len(all_npy)} unique .npy files\n")

    for npy_path in all_npy:
        vid_id = npy_path.stem
        if vid_id not in split_map:
            continue

        split = split_map[vid_id]
        fused = np.load(str(npy_path))
        if fused.ndim != 2 or fused.shape[1] != TOTAL_DIM:
            print(f"  BAD SHAPE: {vid_id}  got {fused.shape}  expected (T, {TOTAL_DIM})")
            bad_shape.append(vid_id)
            continue

        clip_emb = fused[:, :CLIP_DIM]
        dino_emb = fused[:, CLIP_DIM:]

        np.save(str(layout.raw_clip_root / split / f"{vid_id}.npy"), clip_emb)
        np.save(str(layout.raw_dino_root / split / f"{vid_id}.npy"), dino_emb)
        print(f"  OK [{split}]: {vid_id}  clip={clip_emb.shape}  dino={dino_emb.shape}")
        ok += 1

    all_found = {path.stem for path in all_npy}
    for split, video_ids in split_ids.items():
        for vid_id in video_ids:
            if vid_id not in all_found:
                missing.append(vid_id)

    print(f"\n{'=' * 50}")
    print(f"Total processed : {ok}")
    print(f"Bad shape       : {len(bad_shape)}")
    print(f"Missing .npy    : {len(missing)}")
    for split in ["train", "val", "test"]:
        n_clip = len(list((layout.raw_clip_root / split).glob('*.npy')))
        n_dino = len(list((layout.raw_dino_root / split).glob('*.npy')))
        print(f"  {split:6s}  clip={n_clip}  dino={n_dino}")

    if missing:
        log = layout.visual_root / "missing_visual.txt"
        log.write_text("\n".join(sorted(missing)))
        print(f"\nMissing IDs logged to {log}")

    if bad_shape:
        log = layout.visual_root / "bad_shape_visual.txt"
        log.write_text("\n".join(sorted(bad_shape)))
        print(f"Bad shape IDs logged to {log}")


if __name__ == "__main__":
    main()
