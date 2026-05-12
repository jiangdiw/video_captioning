import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the BART ablation suite end-to-end.")
    parser.add_argument(
        "--resource-root",
        default=str(PROJECT_ROOT),
        help="Root containing processed data and dataset metadata",
    )
    parser.add_argument("--output-root", default="outputs/ablation_suite")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs-full", type=int, default=20)
    parser.add_argument("--epochs-subset", type=int, default=20)
    parser.add_argument("--batch-size-full", type=int, default=8)
    parser.add_argument("--batch-size-subset", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-caption-len", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--build-missing-full-visuals",
        action="store_true",
        help="Allow rebuilding missing full-dataset TASCC CLIP/DINO features. Disabled by default.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def load_split_counts(resource_root: Path, dataset_mode: str) -> dict[str, int]:
    dataset_root = resource_root / "dataset" / "MSR-VTT"
    metadata_path = dataset_root / ("downsampled_2500.json" if dataset_mode == "subset" else "train_val_videodatainfo.json")
    raw = json.loads(metadata_path.read_text())
    counts = {"train": 0, "val": 0, "test": 0}
    for video in raw.get("videos", []):
        split = video.get("split", "")
        if split == "validate":
            split = "val"
        if split in counts:
            counts[split] += 1
    return counts


def count_visual_features(visual_root: Path) -> dict[str, dict[str, int]]:
    counts = {"clip": {}, "dino": {}}
    for split in ["train", "val", "test"]:
        counts["clip"][split] = sum(1 for _ in (visual_root / "clip_embedding" / split).glob("*.npy")) if (visual_root / "clip_embedding" / split).exists() else 0
        counts["dino"][split] = sum(1 for _ in (visual_root / "Dinov2_embedding" / split).glob("*.npy")) if (visual_root / "Dinov2_embedding" / split).exists() else 0
    return counts


def count_flat_fused_ids(*roots: Path) -> set[str]:
    ids: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.glob("video*.npy"):
            ids.add(path.stem.replace(" 2", ""))
    return ids


def resolve_full_fused_roots(resource_root: Path) -> list[Path]:
    candidates = [
        resource_root / "datas" / "feats" / "tascc_fused",
        Path("/Users/aglooney03/Video-Summarization/datas/feats/tascc_fused"),
        Path(
            "/Users/aglooney03/Library/CloudStorage/GoogleDrive-aidanlooney@g.harvard.edu/"
            "My Drive/undergrad-research/video_captioning_project/Video-Summarization/datas/feats/tascc_fused"
        ),
    ]
    existing = []
    seen = set()
    for path in candidates:
        if path.exists():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                existing.append(path)
    if len(existing) < 2:
        return []

    id_set = count_flat_fused_ids(*existing)
    train_val_ids = {f"video{i}" for i in range(7010)}
    test_ids = {f"video{i}" for i in range(7010, 10000)}
    if not train_val_ids.issubset(id_set):
        return []
    if not test_ids.issubset(id_set):
        return []
    return existing


def ensure_visual_features(
    resource_root: Path,
    dataset_mode: str,
    selector: str,
    visual_root: Path,
    device: str,
    debug_n: int | None,
) -> None:
    expected = load_split_counts(resource_root, dataset_mode)
    current = count_visual_features(visual_root)
    needs_build = False
    splits_to_build: list[str] = []
    for split in ["train", "val", "test"]:
        if current["clip"].get(split, 0) < expected[split] or current["dino"].get(split, 0) < expected[split]:
            needs_build = True
            splits_to_build.append(split)
    if not needs_build:
        print(f"Visual features already complete for selector={selector} dataset_mode={dataset_mode}")
        return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "build_visual_ablation_features.py"),
        "--resource-root",
        str(resource_root),
        "--dataset-mode",
        dataset_mode,
        "--selector",
        selector,
        "--visual-root",
        str(visual_root),
        "--splits",
        ",".join(splits_to_build),
        "--device",
        device,
    ]
    if debug_n is not None:
        command.extend(["--limit", str(debug_n)])
    run_command(command, cwd=PROJECT_ROOT)


def require_visual_features(
    *,
    resource_root: Path,
    dataset_mode: str,
    visual_root: Path,
    selector: str,
) -> None:
    expected = load_split_counts(resource_root, dataset_mode)
    current = count_visual_features(visual_root)
    missing_parts: list[str] = []
    for family in ["clip", "dino"]:
        for split in ["train", "val", "test"]:
            actual = current[family].get(split, 0)
            target = expected[split]
            if actual < target:
                missing_parts.append(f"{family}:{split}={actual}/{target}")
    if missing_parts:
        raise FileNotFoundError(
            f"Incomplete visual features for selector={selector} dataset_mode={dataset_mode} at {visual_root}. "
            f"Missing counts: {', '.join(missing_parts)}"
        )


def run_experiment(
    *,
    name: str,
    resource_root: Path,
    dataset_mode: str,
    modalities: str,
    freeze_decoder: bool,
    visual_root: Path,
    fused_visual_roots: list[Path] | None,
    output_root: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    num_beams: int,
    max_caption_len: int,
    seed: int,
    debug_n: int | None,
    skip_existing: bool,
) -> dict:
    run_dir = output_root / name
    metrics_path = run_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text())

    command = [
        sys.executable,
        str(PROJECT_ROOT / "run_bart_experiment.py"),
        "--resource-root",
        str(resource_root),
        "--dataset-mode",
        dataset_mode,
        "--modalities",
        modalities,
        "--visual-root",
        str(visual_root),
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--grad-clip",
        str(grad_clip),
        "--num-beams",
        str(num_beams),
        "--max-caption-len",
        str(max_caption_len),
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if fused_visual_roots:
        command.extend(
            [
                "--fused-visual-roots",
                ",".join(str(path) for path in fused_visual_roots),
            ]
        )
    if freeze_decoder:
        command.append("--freeze-decoder")
    if debug_n is not None:
        command.extend(["--debug-n", str(debug_n)])
    if skip_existing:
        command.append("--skip-if-complete")

    run_command(command, cwd=PROJECT_ROOT)
    return json.loads(metrics_path.read_text())


def save_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_md = output_root / "summary.md"
    summary_json.write_text(json.dumps(rows, indent=2))

    fieldnames = [
        "experiment",
        "dataset_mode",
        "selector",
        "modalities",
        "freeze_decoder",
        "Bleu_4",
        "CIDEr",
        "ROUGE_L",
        "METEOR",
        "best_epoch",
        "best_val_loss",
        "run_dir",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    lines = [
        "| Experiment | Mode | Selector | Modalities | Freeze Decoder | BLEU@4 | CIDEr | ROUGE_L | METEOR | Best Epoch |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {dataset_mode} | {selector} | {modalities} | {freeze_decoder} | {Bleu_4:.4f} | {CIDEr:.4f} | {ROUGE_L:.4f} | {METEOR:.4f} | {best_epoch} |".format(
                experiment=row["experiment"],
                dataset_mode=row["dataset_mode"],
                selector=row["selector"],
                modalities=row["modalities"],
                freeze_decoder=str(row["freeze_decoder"]).lower(),
                Bleu_4=float(row.get("Bleu_4", 0.0)),
                CIDEr=float(row.get("CIDEr", 0.0)),
                ROUGE_L=float(row.get("ROUGE_L", 0.0)),
                METEOR=float(row.get("METEOR", 0.0)),
                best_epoch=row.get("best_epoch", ""),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    resource_root = Path(args.resource_root).expanduser().resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()

    subset_visual_tascc = resource_root / "data" / "processed" / "visual"
    full_visual_tascc = resource_root / "data" / "processed_full" / "visual"
    subset_visual_hist = resource_root / "data" / "processed_hist_subset" / "visual"
    full_fused_roots: list[Path] | None = None

    require_visual_features(
        resource_root=resource_root,
        dataset_mode="subset",
        selector="tascc",
        visual_root=subset_visual_tascc,
    )
    if args.build_missing_full_visuals:
        ensure_visual_features(
            resource_root=resource_root,
            dataset_mode="full",
            selector="tascc",
            visual_root=full_visual_tascc,
            device=args.device,
            debug_n=args.debug_n,
        )
    else:
        full_counts = count_visual_features(full_visual_tascc)
        expected_full = load_split_counts(resource_root, "full")
        full_complete = all(
            full_counts[family].get(split, 0) >= expected_full[split]
            for family in ["clip", "dino"]
            for split in ["train", "val", "test"]
        )
        if full_complete:
            full_fused_roots = None
        else:
            full_fused_roots = resolve_full_fused_roots(resource_root)
            if not full_fused_roots:
                require_visual_features(
                    resource_root=resource_root,
                    dataset_mode="full",
                    selector="tascc",
                    visual_root=full_visual_tascc,
                )

    summary_rows: list[dict] = []

    freeze_runs = []
    for freeze_decoder in [False, True]:
        experiment = f"full_tascc_clip_dino_audio_{'freeze' if freeze_decoder else 'trainable'}"
        metrics = run_experiment(
            name=experiment,
            resource_root=resource_root,
            dataset_mode="full",
            modalities="clip_dino_audio",
            freeze_decoder=freeze_decoder,
            visual_root=full_visual_tascc,
            fused_visual_roots=full_fused_roots,
            output_root=output_root,
            device=args.device,
            epochs=args.epochs_full,
            batch_size=args.batch_size_full,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_beams=args.num_beams,
            max_caption_len=args.max_caption_len,
            seed=args.seed,
            debug_n=args.debug_n,
            skip_existing=args.skip_existing,
        )
        row = {
            "experiment": experiment,
            "dataset_mode": "full",
            "selector": "tascc",
            "modalities": "clip_dino_audio",
            "freeze_decoder": freeze_decoder,
            "run_dir": str((output_root / experiment).resolve()),
            **metrics,
        }
        freeze_runs.append(row)
        summary_rows.append(row)

    best_freeze = max(freeze_runs, key=lambda row: float(row.get("CIDEr", -1.0)))["freeze_decoder"]

    subset_runs = []
    for modalities in ["clip", "clip_dino", "clip_dino_audio"]:
        experiment = f"subset_tascc_{modalities}_{'freeze' if best_freeze else 'trainable'}"
        metrics = run_experiment(
            name=experiment,
            resource_root=resource_root,
            dataset_mode="subset",
            modalities=modalities,
            freeze_decoder=best_freeze,
            visual_root=subset_visual_tascc,
            fused_visual_roots=None,
            output_root=output_root,
            device=args.device,
            epochs=args.epochs_subset,
            batch_size=args.batch_size_subset,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            num_beams=args.num_beams,
            max_caption_len=args.max_caption_len,
            seed=args.seed,
            debug_n=args.debug_n,
            skip_existing=args.skip_existing,
        )
        row = {
            "experiment": experiment,
            "dataset_mode": "subset",
            "selector": "tascc",
            "modalities": modalities,
            "freeze_decoder": best_freeze,
            "run_dir": str((output_root / experiment).resolve()),
            **metrics,
        }
        subset_runs.append(row)
        summary_rows.append(row)

    best_subset_row = max(subset_runs, key=lambda row: float(row.get("CIDEr", -1.0)))
    best_modalities = best_subset_row["modalities"]

    ensure_visual_features(
        resource_root=resource_root,
        dataset_mode="subset",
        selector="histogram_topk",
        visual_root=subset_visual_hist,
        device=args.device,
        debug_n=args.debug_n,
    )

    histogram_experiment = f"subset_histogram_{best_modalities}_{'freeze' if best_freeze else 'trainable'}"
    histogram_metrics = run_experiment(
        name=histogram_experiment,
        resource_root=resource_root,
        dataset_mode="subset",
        modalities=best_modalities,
        freeze_decoder=best_freeze,
        visual_root=subset_visual_hist,
        fused_visual_roots=None,
        output_root=output_root,
        device=args.device,
        epochs=args.epochs_subset,
        batch_size=args.batch_size_subset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_beams=args.num_beams,
        max_caption_len=args.max_caption_len,
        seed=args.seed,
        debug_n=args.debug_n,
        skip_existing=args.skip_existing,
    )
    summary_rows.append(
        {
            "experiment": histogram_experiment,
            "dataset_mode": "subset",
            "selector": "histogram_topk",
            "modalities": best_modalities,
            "freeze_decoder": best_freeze,
            "run_dir": str((output_root / histogram_experiment).resolve()),
            **histogram_metrics,
        }
    )

    save_summary(output_root, summary_rows)
    print(json.dumps(
        {
            "best_freeze_decoder": best_freeze,
            "best_subset_modalities": best_modalities,
            "summary_json": str((output_root / "summary.json").resolve()),
            "summary_csv": str((output_root / "summary.csv").resolve()),
            "summary_md": str((output_root / "summary.md").resolve()),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
