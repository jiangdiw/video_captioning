import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from misc.cocoeval import COCOScorer, suppress_stdout_stderr


def default_predictions_path() -> Path:
    return PROJECT_ROOT / "outputs" / "final_bart_full_stable_v1" / "test_predictions_xe.json"


def default_captions_path() -> Path:
    return Path("/Users/aglooney03/Video-Summarization/data/processed_full/captions/test_captions.json")


def default_metadata_path() -> Path:
    return PROJECT_ROOT / "dataset" / "MSR-VTT" / "test_videodatainfo.json"


def default_train_metadata_path(train_dataset_mode: str) -> Path:
    if train_dataset_mode == "subset":
        return PROJECT_ROOT / "dataset" / "MSR-VTT" / "downsampled_2500.json"
    return PROJECT_ROOT / "dataset" / "MSR-VTT" / "train_val_videodatainfo.json"


def build_category_map(metadata: dict) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for video in metadata.get("videos", []):
        if video.get("split") != "test":
            continue
        mapping[video["video_id"]] = int(video["category"])
    return mapping


def build_train_length_by_category(metadata: dict) -> dict[int, list[float]]:
    durations: dict[int, list[float]] = {}
    for video in metadata.get("videos", []):
        split = video.get("split")
        if split != "train":
            continue
        category = int(video["category"])
        duration = float(video["end time"]) - float(video["start time"])
        durations.setdefault(category, []).append(duration)
    return durations


def score_subset(
    scorer: COCOScorer,
    predictions: dict,
    references: dict,
    video_ids: list[str],
) -> dict:
    subset_predictions = {video_id: predictions[video_id] for video_id in video_ids}
    subset_references = {
        video_id: [{"image_id": video_id, "caption": caption} for caption in references[video_id]]
        for video_id in video_ids
    }
    with suppress_stdout_stderr():
        return scorer.score(subset_references, subset_predictions, video_ids)


def make_performance_chart(rows: list[dict], overall_cider: float, output_path: Path) -> None:
    categories = [str(row["category"]) for row in rows]
    cider_scores = [row["CIDEr"] for row in rows]
    bleu4_scores = [row["Bleu_4"] for row in rows]

    plt.figure(figsize=(12, 6))
    x = list(range(len(categories)))
    width = 0.42
    plt.bar([i - width / 2 for i in x], cider_scores, width=width, label="CIDEr")
    plt.bar([i + width / 2 for i in x], bleu4_scores, width=width, label="BLEU@4")
    plt.axhline(overall_cider, color="crimson", linestyle="--", linewidth=1.5, label=f"Overall CIDEr ({overall_cider:.3f})")
    plt.xticks(x, categories)
    plt.xlabel("MSR-VTT Category")
    plt.ylabel("Score")
    plt.title("Test Performance by Category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_train_length_chart(length_rows: list[dict], output_path: Path) -> None:
    categories = [str(row["category"]) for row in length_rows]
    avg_lengths = [row["avg_train_duration_sec"] for row in length_rows]

    plt.figure(figsize=(12, 6))
    plt.bar(categories, avg_lengths)
    plt.xlabel("MSR-VTT Category")
    plt.ylabel("Average Train Video Length (sec)")
    plt.title("Average Training Video Length by Category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Score saved test predictions by MSR-VTT category.")
    parser.add_argument("--predictions", default=str(default_predictions_path()))
    parser.add_argument("--captions", default=str(default_captions_path()))
    parser.add_argument("--metadata", default=str(default_metadata_path()))
    parser.add_argument("--train-dataset-mode", choices=["full", "subset"], default="full")
    parser.add_argument("--train-metadata", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions_path = Path(args.predictions).expanduser().resolve()
    captions_path = Path(args.captions).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    train_metadata_path = (
        Path(args.train_metadata).expanduser().resolve()
        if args.train_metadata
        else default_train_metadata_path(args.train_dataset_mode).resolve()
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = predictions_path.parent / "category_breakdown"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = json.loads(predictions_path.read_text())
    references = json.loads(captions_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    train_metadata = json.loads(train_metadata_path.read_text())
    category_map = build_category_map(metadata)
    train_lengths = build_train_length_by_category(train_metadata)

    valid_video_ids = sorted(
        [video_id for video_id in predictions if video_id in references and video_id in category_map],
        key=lambda value: int(value.replace("video", "")),
    )
    if not valid_video_ids:
        raise RuntimeError("No overlapping video_ids among predictions, captions, and metadata")

    videos_by_category: dict[int, list[str]] = {}
    for video_id in valid_video_ids:
        videos_by_category.setdefault(category_map[video_id], []).append(video_id)

    scorer = COCOScorer()
    rows: list[dict] = []
    for category_id in sorted(videos_by_category):
        video_ids = videos_by_category[category_id]
        metrics = score_subset(scorer, predictions, references, video_ids)
        rows.append(
            {
                "category": category_id,
                "num_videos": len(video_ids),
                "Bleu_1": metrics.get("Bleu_1", 0.0),
                "Bleu_2": metrics.get("Bleu_2", 0.0),
                "Bleu_3": metrics.get("Bleu_3", 0.0),
                "Bleu_4": metrics.get("Bleu_4", 0.0),
                "METEOR": metrics.get("METEOR", 0.0),
                "ROUGE_L": metrics.get("ROUGE_L", 0.0),
                "CIDEr": metrics.get("CIDEr", 0.0),
            }
        )

    rows_sorted_by_cider = sorted(rows, key=lambda row: row["CIDEr"], reverse=True)
    overall = score_subset(scorer, predictions, references, valid_video_ids)
    length_rows = []
    for category_id in sorted(train_lengths):
        durations = train_lengths[category_id]
        length_rows.append(
            {
                "category": category_id,
                "num_train_videos": len(durations),
                "avg_train_duration_sec": sum(durations) / max(len(durations), 1),
            }
        )
    train_length_lookup = {row["category"]: row for row in length_rows}
    for row in rows:
        extra = train_length_lookup.get(row["category"])
        if extra:
            row["num_train_videos"] = extra["num_train_videos"]
            row["avg_train_duration_sec"] = extra["avg_train_duration_sec"]

    make_performance_chart(rows=sorted(rows, key=lambda row: row["category"]), overall_cider=overall.get("CIDEr", 0.0), output_path=output_dir / "category_performance.png")
    make_train_length_chart(length_rows=sorted(length_rows, key=lambda row: row["category"]), output_path=output_dir / "category_train_avg_length.png")

    payload = {
        "predictions_path": str(predictions_path),
        "captions_path": str(captions_path),
        "metadata_path": str(metadata_path),
        "train_metadata_path": str(train_metadata_path),
        "num_videos": len(valid_video_ids),
        "overall": overall,
        "by_category": rows,
        "by_cider_desc": rows_sorted_by_cider,
        "train_length_by_category": length_rows,
    }
    (output_dir / "category_metrics.json").write_text(json.dumps(payload, indent=2))

    with (output_dir / "category_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "num_videos",
                "num_train_videos",
                "avg_train_duration_sec",
                "Bleu_1",
                "Bleu_2",
                "Bleu_3",
                "Bleu_4",
                "METEOR",
                "ROUGE_L",
                "CIDEr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append(f"Predictions: `{predictions_path}`")
    lines.append(f"Videos scored: `{len(valid_video_ids)}`")
    lines.append("")
    lines.append("Overall:")
    lines.append(
        f"- BLEU@4: `{overall.get('Bleu_4', 0.0):.4f}`"
    )
    lines.append(
        f"- CIDEr: `{overall.get('CIDEr', 0.0):.4f}`"
    )
    lines.append("")
    lines.append("| Category | Videos | BLEU@4 | CIDEr |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in rows_sorted_by_cider:
        lines.append(
            f"| {row['category']} | {row['num_videos']} | {row['Bleu_4']:.4f} | {row['CIDEr']:.4f} |"
        )
    lines.append("")
    lines.append("Artifacts:")
    lines.append(f"- `category_performance.png`")
    lines.append(f"- `category_train_avg_length.png`")
    (output_dir / "category_metrics.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {output_dir / 'category_metrics.json'}")
    print(f"Wrote {output_dir / 'category_metrics.csv'}")
    print(f"Wrote {output_dir / 'category_metrics.md'}")
    print("\nTop 5 categories by CIDEr:")
    for row in rows_sorted_by_cider[:5]:
        print(
            f"  category {row['category']:>2}: videos={row['num_videos']:<3} "
            f"CIDEr={row['CIDEr']:.4f} BLEU@4={row['Bleu_4']:.4f}"
        )
    print("\nBottom 5 categories by CIDEr:")
    for row in rows_sorted_by_cider[-5:]:
        print(
            f"  category {row['category']:>2}: videos={row['num_videos']:<3} "
            f"CIDEr={row['CIDEr']:.4f} BLEU@4={row['Bleu_4']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
