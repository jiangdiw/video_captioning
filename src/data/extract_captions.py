# src/data/extract_captions.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from collections import defaultdict
from src.data.split_videos_simple import get_splits

# -------------------
# PATHS
# -------------------
DATA_ROOT    = Path("data")
JSON_PATH    = DATA_ROOT / "raw" / "metadata" / "downsampled_2500.json"
CAPTION_DIR  = DATA_ROOT / "processed" / "captions"
CAPTION_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # Load downsampled JSON
    print(f"Loading {JSON_PATH}...")
    with open(JSON_PATH) as f:
        data = json.load(f)

    # Build video_id → list of captions
    captions = defaultdict(list)
    for s in data["sentences"]:
        captions[s["video_id"]].append(s["caption"])

    print(f"Total videos with captions : {len(captions)}")
    print(f"Total captions             : {sum(len(v) for v in captions.values())}")
    print(f"Captions per video         : {len(list(captions.values())[0])}\n")

    # Get splits from our single source of truth
    train_files, val_files, test_files = get_splits()

    split_map = {
        "train" : [f.stem for f in train_files],
        "val"   : [f.stem for f in val_files],
        "test"  : [f.stem for f in test_files],
    }

    # -------------------
    # Extract and save per split
    # -------------------
    total_missing = []

    for split, video_ids in split_map.items():
        split_captions = {}
        missing        = []

        for vid_id in video_ids:
            if vid_id in captions:
                split_captions[vid_id] = captions[vid_id]
            else:
                missing.append(vid_id)

        # Save to JSON
        out_path = CAPTION_DIR / f"{split}_captions.json"
        with open(out_path, "w") as f:
            json.dump(split_captions, f, indent=2)

        print(f"[{split}]")
        print(f"  Videos expected  : {len(video_ids)}")
        print(f"  Videos with caps : {len(split_captions)}")
        print(f"  Missing captions : {len(missing)}")
        print(f"  Total captions   : {sum(len(v) for v in split_captions.values())}")
        print(f"  Saved to         : {out_path}")

        # Show 2 examples
        sample_id = list(split_captions.keys())[0]
        print(f"  Example ({sample_id}):")
        for cap in split_captions[sample_id][:3]:
            print(f"    - {cap}")
        print()

        if missing:
            total_missing.extend(missing)
            log = CAPTION_DIR / f"missing_captions_{split}.txt"
            log.write_text("\n".join(missing))
            print(f"  ⚠ Missing IDs → {log}")

    # -------------------
    # Final summary
    # -------------------
    print(f"{'='*55}")
    print(f"SUMMARY")
    print(f"{'='*55}")
    for split in ["train", "val", "test"]:
        out_path = CAPTION_DIR / f"{split}_captions.json"
        with open(out_path) as f:
            d = json.load(f)
        print(f"  {split:6s}  videos={len(d)}  "
              f"captions={sum(len(v) for v in d.values())}")

    if total_missing:
        print(f"\n⚠ Total missing: {len(total_missing)} videos have no captions")
    else:
        print(f"\n✓ All videos have captions")

if __name__ == "__main__":
    main()