#!/usr/bin/env python3
"""Generate fixed-length TASCC fused features for MSR-VTT."""

from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "novel_keyframe_extractor.py").exists() else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_keyframe_extractor import (  # noqa: E402
    choose_device,
    compress_to_fixed_budget,
    compute_adaptive_raw_budget,
    compute_semantic_scores,
    detect_scenes,
    embed_candidates,
    greedy_select,
    load_prompts,
    sample_video_candidates,
)


DEFAULT_VIDEO_DIR_CANDIDATES = [
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/TestVideo"),
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/TestVideo"),
    Path("dataset/MSR-VTT/full_dataset/TestVideo"),
    Path("dataset/MSR-VTT/full_dataset"),
    Path("dataset/MSR-VTT/videos_224"),
]


DEFAULT_EXTERNAL_OUTPUT_ROOTS = [
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/TASCC_MSRVTT"),
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/TASCC_MSRVTT"),
]


def choose_default_path(candidates: list[Path], fallback: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return fallback


def is_writable_directory(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(path, os.W_OK)


def choose_default_output_root() -> Path:
    for candidate in DEFAULT_EXTERNAL_OUTPUT_ROOTS:
        parent = candidate.parent
        if is_writable_directory(parent):
            return candidate
    return Path("datas/feats")


def parse_args() -> argparse.Namespace:
    default_output_root = choose_default_output_root()
    parser = argparse.ArgumentParser(description="Precompute TASCC fused features for MSR-VTT.")
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=choose_default_path(DEFAULT_VIDEO_DIR_CANDIDATES, Path("dataset/MSR-VTT/full_dataset")),
        help="Directory containing MSR-VTT mp4 files.",
    )
    parser.add_argument("--video-glob", type=str, default="*.mp4")
    parser.add_argument("--output-dir", type=Path, default=default_output_root / "tascc_fused")
    parser.add_argument("--timestamps-dir", type=Path, default=default_output_root / "tascc_timestamps")
    parser.add_argument("--manifest-path", type=Path, default=default_output_root / "tascc_manifest.json")
    parser.add_argument("--failure-log", type=Path, default=default_output_root / "tascc_failures.jsonl")
    parser.add_argument("--num-keyframes", type=int, default=40)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--duration-ref-sec", type=float, default=14.78)
    parser.add_argument("--complexity-weight", type=float, default=0.75)
    parser.add_argument("--min-raw-keyframes", type=int, default=40)
    parser.add_argument("--max-raw-keyframes", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--clip-model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model-name", type=str, default="facebook/dinov2-base")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROJECT_ROOT / "prompt_banks" / "generic_caption_prompts.txt",
    )
    parser.add_argument("--disable-default-prompts", action="store_true")
    parser.add_argument("--min-scene-duration", type=float, default=2.0)
    parser.add_argument("--max-scene-duration", type=float, default=20.0)
    parser.add_argument("--boundary-percentile", type=float, default=92.0)
    parser.add_argument("--min-keyframe-gap", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.34)
    parser.add_argument("--motion-weight", type=float, default=0.16)
    parser.add_argument("--histogram-weight", type=float, default=0.06)
    parser.add_argument("--quality-weight", type=float, default=0.14)
    parser.add_argument("--rarity-weight", type=float, default=0.14)
    parser.add_argument("--prompt-weight", type=float, default=0.22)
    parser.add_argument("--clip-fusion-weight", type=float, default=0.55)
    parser.add_argument("--dino-fusion-weight", type=float, default=0.45)
    parser.add_argument("--coverage-weight", type=float, default=0.55)
    parser.add_argument("--scene-bonus", type=float, default=0.15)
    parser.add_argument("--redundancy-penalty", type=float, default=0.30)
    parser.add_argument("--compression-scene-weight", type=float, default=0.50)
    parser.add_argument("--compression-duration-weight", type=float, default=0.20)
    parser.add_argument("--compression-prompt-weight", type=float, default=0.30)
    parser.add_argument("--compression-importance-weight", type=float, default=0.60)
    parser.add_argument("--compression-centrality-weight", type=float, default=0.25)
    parser.add_argument("--compression-boundary-weight", type=float, default=0.15)
    parser.add_argument("--knn-rarity", type=int, default=5)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-list-json", type=Path, default=None)
    return parser.parse_args()


def resolve_video_dir(video_dir: Path | None) -> Path:
    if video_dir is not None:
        return video_dir
    for candidate in DEFAULT_VIDEO_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find an MSR-VTT video directory. Pass --video-dir explicitly."
    )


def load_requested_video_ids(video_list_json: Path | None) -> set[str] | None:
    if video_list_json is None:
        return None
    payload = json.loads(video_list_json.read_text())
    requested_ids = set()
    for video in payload.get("videos", []):
        if "video_id" in video:
            requested_ids.add(str(video["video_id"]))
        elif "id" in video:
            requested_ids.add(f"video{video['id']}")
    return requested_ids


def video_sort_key(video_path: Path):
    stem = video_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (stem.rstrip(digits), int(digits))
    return (stem, stem)


def resample_embeddings(features: np.ndarray, timestamps: np.ndarray, target_len: int) -> np.ndarray:
    if features.shape[0] == target_len:
        return features.astype(np.float32, copy=False)
    if features.shape[0] == 1:
        return np.repeat(features.astype(np.float32, copy=False), target_len, axis=0)

    src = timestamps.astype(np.float32)
    src = src - src[0]
    if float(src[-1]) <= 1e-6:
        src = np.linspace(0.0, 1.0, num=features.shape[0], dtype=np.float32)
    else:
        src = src / src[-1]
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)

    upper = np.searchsorted(src, dst, side="right")
    upper = np.clip(upper, 1, features.shape[0] - 1)
    lower = upper - 1
    denom = np.clip(src[upper] - src[lower], 1e-6, None)
    alpha = ((dst - src[lower]) / denom).reshape(-1, 1).astype(np.float32)
    blended = features[lower] * (1.0 - alpha) + features[upper] * alpha
    blended[0] = features[0]
    blended[-1] = features[-1]
    return blended.astype(np.float32, copy=False)


def write_timestamp_csv(csv_path: Path, timestamps: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selected_rank", "timestamp_sec"])
        for rank, timestamp in enumerate(timestamps, start=1):
            writer.writerow([rank, f"{float(timestamp):.6f}"])


def append_failure_log(failure_log: Path, record: dict) -> None:
    try:
        with failure_log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as exc:
        if exc.errno != errno.ENOSPC:
            raise


def extract_fixed_features(video_path: Path, args: argparse.Namespace, prompts, device: str):
    candidates, metadata = sample_video_candidates(
        video_path=video_path,
        sample_fps=args.sample_fps,
        thumb_size=args.thumb_size,
        max_samples=args.max_samples,
    )
    if not candidates:
        raise RuntimeError("No candidate frames were sampled.")

    clip_array, dino_array, prompt_alignment, _ = embed_candidates(
        candidates=candidates,
        prompts=prompts,
        device=device,
        precision_arg=args.precision,
        clip_model_name=args.clip_model_name,
        dino_model_name=args.dino_model_name,
        batch_size=args.batch_size,
    )
    fused, similarity, importance, boundary_score = compute_semantic_scores(
        candidates=candidates,
        clip_array=clip_array,
        dino_array=dino_array,
        prompt_alignment=prompt_alignment,
        clip_weight=args.clip_fusion_weight,
        dino_weight=args.dino_fusion_weight,
        semantic_weight=args.semantic_weight,
        motion_weight=args.motion_weight,
        histogram_weight=args.histogram_weight,
        quality_weight=args.quality_weight,
        rarity_weight=args.rarity_weight,
        prompt_weight=args.prompt_weight,
        knn_rarity=args.knn_rarity,
    )
    scenes = detect_scenes(
        candidates=candidates,
        boundary_score=boundary_score,
        boundary_percentile=args.boundary_percentile,
        min_scene_duration=args.min_scene_duration,
        max_scene_duration=args.max_scene_duration,
    )
    raw_budget, _ = compute_adaptive_raw_budget(
        metadata=metadata,
        candidates=candidates,
        boundary_score=boundary_score,
        final_budget=args.num_keyframes,
        adaptive_budget=args.adaptive_budget,
        duration_ref_sec=args.duration_ref_sec,
        complexity_weight=args.complexity_weight,
        min_raw_keyframes=args.min_raw_keyframes,
        max_raw_keyframes=args.max_raw_keyframes,
        boundary_percentile=args.boundary_percentile,
    )
    raw_selected_indices = greedy_select(
        candidates=candidates,
        similarity=np.clip(similarity, 0.0, 1.0),
        importance=importance,
        scenes=scenes,
        num_keyframes=raw_budget,
        coverage_weight=args.coverage_weight,
        scene_bonus=args.scene_bonus,
        redundancy_penalty=args.redundancy_penalty,
        min_keyframe_gap=args.min_keyframe_gap,
    )
    selected_indices = compress_to_fixed_budget(
        candidates=candidates,
        raw_selected_indices=raw_selected_indices,
        scenes=scenes,
        similarity=np.clip(similarity, 0.0, 1.0),
        final_budget=min(args.num_keyframes, len(raw_selected_indices)),
        compression_scene_weight=args.compression_scene_weight,
        compression_duration_weight=args.compression_duration_weight,
        compression_prompt_weight=args.compression_prompt_weight,
        compression_importance_weight=args.compression_importance_weight,
        compression_centrality_weight=args.compression_centrality_weight,
        compression_boundary_weight=args.compression_boundary_weight,
    )
    selected_indices = sorted(selected_indices, key=lambda idx: candidates[idx].timestamp_sec)
    if not selected_indices:
        raise RuntimeError("No key frames were selected.")

    selected_timestamps = np.asarray(
        [candidates[idx].timestamp_sec for idx in selected_indices],
        dtype=np.float32,
    )
    fixed_features = resample_embeddings(fused[selected_indices], selected_timestamps, args.num_keyframes)
    return fixed_features, selected_timestamps, metadata, len(candidates), len(raw_selected_indices), len(selected_indices)


def main() -> int:
    args = parse_args()
    video_dir = resolve_video_dir(args.video_dir)
    requested_ids = load_requested_video_ids(args.video_list_json)
    prompts = load_prompts(args.prompt_file, use_defaults=not args.disable_default_prompts)
    device = choose_device(args.device)

    video_paths = sorted(video_dir.glob(args.video_glob), key=video_sort_key)
    if requested_ids is not None:
        video_paths = [path for path in video_paths if path.stem in requested_ids]
    if args.limit is not None:
        video_paths = video_paths[:args.limit]
    if not video_paths:
        raise RuntimeError(f"No videos found in {video_dir} matching {args.video_glob}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.timestamps_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.failure_log.parent.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    for video_path in tqdm(video_paths, desc="Precomputing TASCC features", unit="video"):
        output_path = args.output_dir / f"{video_path.stem}.npy"
        timestamp_path = args.timestamps_dir / f"{video_path.stem}_selected_timestamps.csv"
        if output_path.exists() and not args.overwrite:
            results.append({"video_id": video_path.stem, "status": "skipped_existing"})
            continue

        try:
            fixed_features, timestamps, metadata, sampled_count, raw_count, selected_count = extract_fixed_features(
                video_path=video_path,
                args=args,
                prompts=prompts,
                device=device,
            )
            np.save(output_path, fixed_features.astype(np.float32, copy=False))
            write_timestamp_csv(timestamp_path, timestamps)
            results.append(
                {
                    "video_id": video_path.stem,
                    "status": "ok",
                    "output_path": str(output_path),
                    "duration_sec": round(float(metadata.duration_sec), 4),
                    "sampled_candidates": sampled_count,
                    "raw_selected": raw_count,
                    "selected_before_resample": selected_count,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failure_record = {"video_id": video_path.stem, "error": str(exc)}
            failures.append(failure_record)
            results.append({"video_id": video_path.stem, "status": "failed", "error": str(exc)})
            append_failure_log(args.failure_log, failure_record)
            if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
                manifest = {
                    "video_dir": str(video_dir),
                    "output_dir": str(args.output_dir),
                    "timestamps_dir": str(args.timestamps_dir),
                    "device": device,
                    "precision": args.precision,
                    "num_keyframes": args.num_keyframes,
                    "processed_videos": len(video_paths),
                    "successful_videos": sum(1 for item in results if item["status"] == "ok"),
                    "skipped_videos": sum(1 for item in results if item["status"] == "skipped_existing"),
                    "failed_videos": len(failures),
                    "results": results,
                    "stopped_early": True,
                    "stop_reason": "no_space_left_on_device",
                }
                try:
                    args.manifest_path.write_text(json.dumps(manifest, indent=2))
                except OSError:
                    pass
                print(
                    json.dumps(
                        {
                            "error": "No space left on device",
                            "last_video": video_path.stem,
                            "completed_features": sum(1 for item in results if item["status"] == "ok"),
                            "output_dir": str(args.output_dir),
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
                return 1

    manifest = {
        "video_dir": str(video_dir),
        "output_dir": str(args.output_dir),
        "timestamps_dir": str(args.timestamps_dir),
        "device": device,
        "precision": args.precision,
        "num_keyframes": args.num_keyframes,
        "processed_videos": len(video_paths),
        "successful_videos": sum(1 for item in results if item["status"] == "ok"),
        "skipped_videos": sum(1 for item in results if item["status"] == "skipped_existing"),
        "failed_videos": len(failures),
        "results": results,
    }
    args.manifest_path.write_text(json.dumps(manifest, indent=2))

    if failures:
        print(json.dumps({"manifest_path": str(args.manifest_path), "failures": failures[:10]}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({"manifest_path": str(args.manifest_path), "processed": len(video_paths), "device": device}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
