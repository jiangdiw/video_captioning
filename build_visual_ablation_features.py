import argparse
import json
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Build CLIP/DINO visual features for TASCC or histogram-topk selectors.")
    parser.add_argument("--resource-root", required=True, help="Root containing dataset/MSR-VTT and processed data")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], required=True)
    parser.add_argument("--selector", choices=["tascc", "histogram_topk"], required=True)
    parser.add_argument("--visual-root", required=True, help="Output root containing clip_embedding/ and Dinov2_embedding/")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--num-keyframes", type=int, default=40)
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--disable-default-prompts", action="store_true")
    parser.add_argument("--min-keyframe-gap", type=float, default=1.0)
    return parser.parse_args()


def load_split_ids(resource_root: Path, dataset_mode: str) -> dict[str, list[str]]:
    dataset_root = resource_root / "dataset" / "MSR-VTT"
    metadata_path = dataset_root / ("downsampled_2500.json" if dataset_mode == "subset" else "train_val_videodatainfo.json")
    raw = json.loads(metadata_path.read_text())
    split_map = {"train": [], "val": [], "test": []}
    for video in raw.get("videos", []):
        split = video.get("split", "")
        if split == "validate":
            split = "val"
        if split in split_map:
            split_map[split].append(video["video_id"])
    for split in split_map:
        split_map[split] = sorted(split_map[split], key=lambda vid: int(vid.replace("video", "")))
    return split_map


def histogram_topk_select(candidates, num_keyframes: int, min_gap: float) -> list[int]:
    ranked = sorted(
        range(len(candidates)),
        key=lambda idx: (candidates[idx].histogram_diff_raw, candidates[idx].quality_raw),
        reverse=True,
    )
    chosen: list[int] = []
    for idx in ranked:
        timestamp = candidates[idx].timestamp_sec
        if all(abs(timestamp - candidates[existing].timestamp_sec) >= min_gap for existing in chosen):
            chosen.append(idx)
        if len(chosen) >= num_keyframes:
            break
    if len(chosen) < min(num_keyframes, len(candidates)):
        for idx in ranked:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= min(num_keyframes, len(candidates)):
                break
    return sorted(chosen[:num_keyframes], key=lambda idx: candidates[idx].timestamp_sec)


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


def main() -> int:
    args = parse_args()
    resource_root = Path(args.resource_root).expanduser().resolve()
    visual_root = Path(args.visual_root).expanduser().resolve()

    os.environ["VIDEO_CAPTIONING_RESOURCE_ROOT"] = str(resource_root)

    from data.msrvtt import build_video_index, normalize_dataset_mode
    from novel_keyframe_extractor import (
        compute_adaptive_raw_budget,
        compute_semantic_scores,
        compress_to_fixed_budget,
        detect_scenes,
        embed_candidates,
        greedy_select,
        load_prompts,
        sample_video_candidates,
    )

    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    requested_splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    split_ids = load_split_ids(resource_root, dataset_mode)
    video_index, video_dirs = build_video_index(dataset_mode)

    prompt_file = Path(args.prompt_file).expanduser() if args.prompt_file else None
    prompts = load_prompts(prompt_file, use_defaults=not args.disable_default_prompts)

    for split in requested_splits:
        (visual_root / "clip_embedding" / split).mkdir(parents=True, exist_ok=True)
        (visual_root / "Dinov2_embedding" / split).mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    missing = 0
    for split in requested_splits:
        ids = split_ids[split]
        if args.limit is not None:
            ids = ids[: args.limit]
        print(f"Processing {split}: {len(ids)} videos")
        for video_id in ids:
            clip_out = visual_root / "clip_embedding" / split / f"{video_id}.npy"
            dino_out = visual_root / "Dinov2_embedding" / split / f"{video_id}.npy"
            if not args.force and clip_out.exists() and dino_out.exists():
                skipped += 1
                continue
            video_path = video_index.get(video_id)
            if video_path is None:
                print(f"  MISSING VIDEO: {video_id}")
                missing += 1
                continue

            candidates, metadata = sample_video_candidates(
                video_path=video_path,
                sample_fps=args.sample_fps,
                thumb_size=args.thumb_size,
                max_samples=args.max_samples,
            )
            clip_array, dino_array, prompt_alignment, _ = embed_candidates(
                candidates=candidates,
                prompts=prompts if args.selector == "tascc" else [],
                device=args.device,
                precision_arg=args.precision,
                clip_model_name="openai/clip-vit-base-patch32",
                dino_model_name="facebook/dinov2-base",
                batch_size=args.batch_size,
            )

            if args.selector == "tascc":
                fused, similarity, importance, boundary_score = compute_semantic_scores(
                    candidates=candidates,
                    clip_array=clip_array,
                    dino_array=dino_array,
                    prompt_alignment=prompt_alignment,
                    clip_weight=0.55,
                    dino_weight=0.45,
                    semantic_weight=0.34,
                    motion_weight=0.16,
                    histogram_weight=0.06,
                    quality_weight=0.14,
                    rarity_weight=0.14,
                    prompt_weight=0.22,
                    knn_rarity=5,
                )
                scenes = detect_scenes(
                    candidates=candidates,
                    boundary_score=boundary_score,
                    boundary_percentile=92.0,
                    min_scene_duration=2.0,
                    max_scene_duration=20.0,
                )
                raw_budget, _ = compute_adaptive_raw_budget(
                    metadata=metadata,
                    candidates=candidates,
                    boundary_score=boundary_score,
                    final_budget=args.num_keyframes,
                    adaptive_budget=False,
                    duration_ref_sec=14.78,
                    complexity_weight=0.75,
                    min_raw_keyframes=args.num_keyframes,
                    max_raw_keyframes=args.num_keyframes,
                    boundary_percentile=92.0,
                )
                raw_selected = greedy_select(
                    candidates=candidates,
                    similarity=np.clip(similarity, 0.0, 1.0),
                    importance=importance,
                    scenes=scenes,
                    num_keyframes=raw_budget,
                    coverage_weight=0.25,
                    scene_bonus=0.08,
                    redundancy_penalty=0.18,
                    min_keyframe_gap=args.min_keyframe_gap,
                )
                selected = compress_to_fixed_budget(
                    candidates=candidates,
                    raw_selected_indices=raw_selected,
                    scenes=scenes,
                    similarity=np.clip(similarity, 0.0, 1.0),
                    final_budget=min(args.num_keyframes, len(raw_selected)),
                    compression_scene_weight=0.55,
                    compression_duration_weight=0.25,
                    compression_prompt_weight=0.20,
                    compression_importance_weight=0.60,
                    compression_centrality_weight=0.25,
                    compression_boundary_weight=0.15,
                )
            else:
                selected = histogram_topk_select(
                    candidates=candidates,
                    num_keyframes=args.num_keyframes,
                    min_gap=args.min_keyframe_gap,
                )

            selected_timestamps = np.asarray(
                [candidates[idx].timestamp_sec for idx in selected],
                dtype=np.float32,
            )
            clip_fixed = resample_embeddings(clip_array[selected], selected_timestamps, args.num_keyframes)
            dino_fixed = resample_embeddings(dino_array[selected], selected_timestamps, args.num_keyframes)

            np.save(clip_out, clip_fixed)
            np.save(dino_out, dino_fixed)
            processed += 1
            print(f"  OK [{split}] {video_id} selector={args.selector} selected={len(selected)} saved={clip_fixed.shape[0]}")

    print(json.dumps({
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "visual_root": str(visual_root),
        "video_dirs": [str(path) for path in video_dirs],
        "selector": args.selector,
        "dataset_mode": dataset_mode,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
