from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


VIDEO_CAPTIONING_ROOT = Path("/Users/aglooney03/video_captioning")
OPENJDK_BIN = Path("/opt/homebrew/opt/openjdk/bin")
if OPENJDK_BIN.exists():
    os.environ["PATH"] = f"{OPENJDK_BIN}:{os.environ.get('PATH', '')}"
sys.path.insert(0, str(VIDEO_CAPTIONING_ROOT))
sys.path.insert(0, str(VIDEO_CAPTIONING_ROOT / "coco-caption"))
os.chdir(str(VIDEO_CAPTIONING_ROOT))

from misc.cocoeval import COCOScorer  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for Dattalion caption metrics.")
    parser.add_argument("--refs-json", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method specification in the form name=/abs/path/to/test_predictions.json",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Paired comparison in the form stronger,weaker",
    )
    parser.add_argument("--metrics", default="CIDEr,Bleu_4")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_methods(method_args: list[str]) -> dict[str, Path]:
    methods: dict[str, Path] = {}
    for item in method_args:
        if "=" not in item:
            raise ValueError(f"Invalid --method value: {item}")
        name, raw_path = item.split("=", 1)
        methods[name.strip()] = Path(raw_path.strip()).expanduser().resolve()
    return methods


def normalize_predictions(raw) -> dict[str, list[dict[str, str]]]:
    predictions = {}
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = []
        for row in raw:
            video_id = row.get("video_id") or row.get("image_id")
            if not video_id:
                raise ValueError("Prediction row is missing video_id/image_id.")
            iterator.append((video_id, row.get("caption", "")))
    else:
        raise ValueError(f"Unsupported prediction payload type: {type(raw)}")

    for video_id, entries in iterator:
        if isinstance(entries, list):
            predictions[video_id] = entries
        elif isinstance(entries, dict):
            predictions[video_id] = [{"image_id": video_id, "caption": entries.get("caption", "")}]
        else:
            predictions[video_id] = [{"image_id": video_id, "caption": str(entries)}]
    return predictions


def score_method(refs: dict, predictions: dict) -> tuple[dict, dict]:
    ids = sorted(set(refs) & set(predictions))
    scorer = COCOScorer()
    metrics = scorer.score(refs, predictions, ids)
    per_image = {
        metric: {video_id: scorer.imgToEval[video_id][metric] for video_id in ids}
        for metric in metrics.keys()
    }
    return metrics, per_image


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    sample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[sample_indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_mean": float(means.mean()),
    }


def bootstrap_paired_diff_ci(a: np.ndarray, b: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    sample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    diffs = a[sample_indices].mean(axis=1) - b[sample_indices].mean(axis=1)
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_mean_diff": float(diffs.mean()),
    }


def write_markdown(path: Path, metrics: list[str], methods_summary: dict, pair_summary: dict) -> None:
    lines = [
        "# Dattalion Bootstrap Confidence Intervals",
        "",
        "## Per-method intervals",
        "",
        "| Method | Metric | Mean | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for method_name, payload in methods_summary.items():
        for metric in metrics:
            item = payload[metric]
            lines.append(
                f"| {method_name} | {metric} | {item['mean']:.4f} | [{item['ci_lower']:.4f}, {item['ci_upper']:.4f}] |"
            )
    if pair_summary:
        lines.extend(
            [
                "",
                "## Paired differences",
                "",
                "| Comparison | Metric | Mean Diff | 95% CI |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for label, payload in pair_summary.items():
            for metric in metrics:
                item = payload[metric]
                lines.append(
                    f"| {label} | {metric} | {item['mean_diff']:.4f} | [{item['ci_lower']:.4f}, {item['ci_upper']:.4f}] |"
                )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = load_json(args.refs_json.expanduser().resolve())
    method_paths = parse_methods(args.method)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    scored_methods = {}
    common_ids = None
    for name, path in method_paths.items():
        preds = normalize_predictions(load_json(path))
        method_metrics, per_image = score_method(refs, preds)
        ids = set(next(iter(per_image.values())).keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
        scored_methods[name] = {
            "predictions_path": str(path),
            "metrics": method_metrics,
            "per_image": per_image,
        }

    if common_ids is None:
        raise ValueError("No methods were scored.")
    common_ids = sorted(common_ids)

    methods_summary = {}
    for method_name, payload in scored_methods.items():
        methods_summary[method_name] = {}
        for metric in metrics:
            values = np.array([payload["per_image"][metric][video_id] for video_id in common_ids], dtype=float)
            methods_summary[method_name][metric] = bootstrap_mean_ci(values, args.n_bootstrap, args.seed)

    pair_summary = {}
    for raw_pair in args.pair:
        stronger, weaker = [item.strip() for item in raw_pair.split(",", 1)]
        label = f"{stronger} - {weaker}"
        pair_summary[label] = {}
        for metric in metrics:
            a = np.array(
                [scored_methods[stronger]["per_image"][metric][video_id] for video_id in common_ids],
                dtype=float,
            )
            b = np.array(
                [scored_methods[weaker]["per_image"][metric][video_id] for video_id in common_ids],
                dtype=float,
            )
            pair_summary[label][metric] = bootstrap_paired_diff_ci(a, b, args.n_bootstrap, args.seed)

    save_json(
        output_dir / "bootstrap_summary.json",
        {
            "refs_json": str(args.refs_json.expanduser().resolve()),
            "common_ids": common_ids,
            "metrics": metrics,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "methods": methods_summary,
            "pairs": pair_summary,
        },
    )

    rows = []
    for method_name, payload in methods_summary.items():
        for metric in metrics:
            item = payload[metric]
            rows.append(
                {
                    "type": "method",
                    "label": method_name,
                    "metric": metric,
                    "mean": item["mean"],
                    "ci_lower": item["ci_lower"],
                    "ci_upper": item["ci_upper"],
                }
            )
    for label, payload in pair_summary.items():
        for metric in metrics:
            item = payload[metric]
            rows.append(
                {
                    "type": "pair",
                    "label": label,
                    "metric": metric,
                    "mean": item["mean_diff"],
                    "ci_lower": item["ci_lower"],
                    "ci_upper": item["ci_upper"],
                }
            )
    with (output_dir / "bootstrap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "label", "metric", "mean", "ci_lower", "ci_upper"])
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(output_dir / "bootstrap_summary.md", metrics, methods_summary, pair_summary)
    print(json.dumps({"output_dir": str(output_dir), "methods": list(method_paths), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
