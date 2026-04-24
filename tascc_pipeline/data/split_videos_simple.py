import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.msrvtt import get_splits as resolve_splits, get_split_video_ids, normalize_dataset_mode


def get_splits(dataset_mode="subset", strict=True, extra_video_dirs=None):
    return resolve_splits(dataset_mode=dataset_mode, strict=strict, extra_video_dirs=extra_video_dirs)


def main():
    parser = argparse.ArgumentParser(description="Resolve MSR-VTT train/val/test video splits.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--video-dir", action="append", default=[], help="Optional additional directory to search for video*.mp4 files.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some split videos are not present locally.")
    args = parser.parse_args()

    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    train_files, val_files, test_files = get_splits(
        dataset_mode=dataset_mode,
        strict=not args.allow_missing,
        extra_video_dirs=args.video_dir,
    )
    split_ids = get_split_video_ids(dataset_mode)

    print(f"dataset_mode: {dataset_mode}")
    print(f"train : {len(train_files)} / {len(split_ids['train'])}")
    print(f"val   : {len(val_files)} / {len(split_ids['val'])}")
    print(f"test  : {len(test_files)} / {len(split_ids['test'])}")


if __name__ == "__main__":
    main()
