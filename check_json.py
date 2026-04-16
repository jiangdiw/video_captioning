# check_json.py (run from project root)
import json
from pathlib import Path

# Search everywhere under data/
all_jsons = list(Path("data").rglob("*.json"))
print(f"Found {len(all_jsons)} JSON files:\n")
for p in all_jsons:
    print(f"  {p}")

# Inspect each one
for p in all_jsons:
    print(f"\n{'='*55}")
    print(f"File: {p}")
    with open(p) as f:
        data = json.load(f)
    print(f"  Top-level keys : {list(data.keys())}")
    print(f"  Total videos   : {len(data.get('videos', []))}")
    print(f"  Total captions : {len(data.get('sentences', []))}")

    # Show first 2 entries of each
    videos = data.get("videos", [])
    if videos:
        print(f"  First video entry : {videos[0]}")

    sentences = data.get("sentences", [])
    if sentences:
        print(f"  First caption entry : {sentences[0]}")
        print(f"  Second caption entry: {sentences[1]}")