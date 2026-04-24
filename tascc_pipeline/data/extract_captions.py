import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from data.msrvtt import get_processed_layout, get_split_video_ids, get_metadata_path, normalize_dataset_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-split caption json files for MSR-VTT.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    layout.captions_root.mkdir(parents=True, exist_ok=True)

    json_path = get_metadata_path(dataset_mode)
    print(f"Loading {json_path}...")
    data = json.loads(json_path.read_text())

    captions = defaultdict(list)
    for sentence in data["sentences"]:
        captions[sentence["video_id"]].append(sentence["caption"])

    print(f"Total videos with captions : {len(captions)}")
    print(f"Total captions             : {sum(len(v) for v in captions.values())}")
    print(f"Dataset mode               : {dataset_mode}\n")

    split_map = get_split_video_ids(dataset_mode)
    total_missing = []

    for split, video_ids in split_map.items():
        split_captions = {}
        missing = []

        for vid_id in video_ids:
            if vid_id in captions:
                split_captions[vid_id] = captions[vid_id]
            else:
                missing.append(vid_id)

        out_path = layout.captions_root / f"{split}_captions.json"
        out_path.write_text(json.dumps(split_captions, indent=2))

        print(f"[{split}]")
        print(f"  Videos expected  : {len(video_ids)}")
        print(f"  Videos with caps : {len(split_captions)}")
        print(f"  Missing captions : {len(missing)}")
        print(f"  Total captions   : {sum(len(v) for v in split_captions.values())}")
        print(f"  Saved to         : {out_path}")

        if split_captions:
            sample_id = next(iter(split_captions))
            print(f"  Example ({sample_id}):")
            for cap in split_captions[sample_id][:3]:
                print(f"    - {cap}")
        print()

        if missing:
            total_missing.extend(missing)
            log = layout.captions_root / f"missing_captions_{split}.txt"
            log.write_text("\n".join(missing))
            print(f"  Missing IDs → {log}")

    print(f"{'=' * 55}")
    print("SUMMARY")
    print(f"{'=' * 55}")
    for split in ["train", "val", "test"]:
        out_path = layout.captions_root / f"{split}_captions.json"
        payload = json.loads(out_path.read_text())
        print(f"  {split:6s}  videos={len(payload)}  captions={sum(len(v) for v in payload.values())}")

    if total_missing:
        print(f"\nMissing captions for {len(total_missing)} videos")
    else:
        print("\nAll videos have captions")


if __name__ == "__main__":
    main()
