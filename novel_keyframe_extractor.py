#!/usr/bin/env python3
"""Prompt-conditioned, transfer-aware key frame extraction for video captioning.

This script implements a stronger alternative to frame-difference histograms:

1. Sample candidate frames at a low frame rate.
2. Encode each frame with two frozen foundation models:
   - CLIP for language-aligned semantics.
   - DINOv2 for transfer-robust visual semantics.
3. Score each candidate with a multi-objective importance function:
   - semantic change
   - motion salience
   - frame quality
   - rarity
   - prompt alignment (optional, useful for domain transfer)
4. Detect semantic scene boundaries.
5. Select key frames with scene seeding plus a coverage-aware greedy objective.

The output is designed for downstream video captioning pipelines:
selected frames are saved chronologically, along with per-frame metadata and
selected fused embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_GENERIC_PROMPTS = [
    "a person interacting with objects",
    "a close-up of an important object",
    "a scene change in a video",
    "a frame showing a clear action",
    "an indoor environment",
    "an outdoor environment",
    "a vehicle or transportation scene",
    "a group of people",
    "a damaged object or structure",
    "smoke fire or an emergency scene",
    "a rescue or response scene",
    "a landscape or wide scene",
]


_EMBEDDER_CACHE: dict[tuple[str, str, str], tuple[Any, Any, Any, Any]] = {}


@dataclass
class VideoMetadata:
    video_path: str
    native_fps: float
    sample_fps: float
    total_frames: int
    duration_sec: float
    sampled_frames: int


@dataclass
class FrameCandidate:
    sample_index: int
    frame_index: int
    timestamp_sec: float
    motion_raw: float
    histogram_diff_raw: float
    sharpness_raw: float
    exposure_raw: float
    quality_raw: float
    thumb_rgb: Any = None
    scene_id: int = -1
    semantic_change: float = 0.0
    boundary_score: float = 0.0
    motion_score: float = 0.0
    histogram_score: float = 0.0
    quality_score: float = 0.0
    rarity_score: float = 0.0
    prompt_alignment: float = 0.0
    local_importance: float = 0.0
    selection_gain: float = 0.0
    raw_selected_rank: int = 0
    compression_score: float = 0.0
    selected_rank: int = 0

    def to_serializable(self) -> dict[str, Any]:
        payload = {
            "sample_index": self.sample_index,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "scene_id": self.scene_id,
            "motion_raw": round(self.motion_raw, 6),
            "histogram_diff_raw": round(self.histogram_diff_raw, 6),
            "sharpness_raw": round(self.sharpness_raw, 6),
            "exposure_raw": round(self.exposure_raw, 6),
            "quality_raw": round(self.quality_raw, 6),
            "semantic_change": round(self.semantic_change, 6),
            "boundary_score": round(self.boundary_score, 6),
            "motion_score": round(self.motion_score, 6),
            "histogram_score": round(self.histogram_score, 6),
            "quality_score": round(self.quality_score, 6),
            "rarity_score": round(self.rarity_score, 6),
            "prompt_alignment": round(self.prompt_alignment, 6),
            "local_importance": round(self.local_importance, 6),
            "selection_gain": round(self.selection_gain, 6),
            "raw_selected_rank": self.raw_selected_rank,
            "compression_score": round(self.compression_score, 6),
            "selected_rank": self.selected_rank,
        }
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer-aware semantic key frame extraction for video captioning."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to an input video.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where selected frames and metadata will be written.",
    )
    parser.add_argument(
        "--num-keyframes",
        type=int,
        default=40,
        help="Final fixed key-frame budget presented to the captioning model.",
    )
    parser.add_argument(
        "--adaptive-budget",
        action="store_true",
        help="Use an adaptive raw key-frame budget before fixed-budget compression.",
    )
    parser.add_argument(
        "--duration-ref-sec",
        type=float,
        default=14.78,
        help="Reference duration in seconds used to scale the raw budget.",
    )
    parser.add_argument(
        "--complexity-weight",
        type=float,
        default=0.75,
        help="How strongly scene-boundary density inflates the raw budget.",
    )
    parser.add_argument(
        "--min-raw-keyframes",
        type=int,
        default=40,
        help="Minimum adaptive raw budget before compression.",
    )
    parser.add_argument(
        "--max-raw-keyframes",
        type=int,
        default=96,
        help="Maximum adaptive raw budget before compression.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Candidate sampling rate before scoring.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device for foundation models.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Numerical precision for GPU inference. CPU always uses fp32.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CLIP and DINOv2 inference.",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model id for CLIP.",
    )
    parser.add_argument(
        "--dino-model-name",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model id for DINOv2.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file with one transfer/domain prompt per line.",
    )
    parser.add_argument(
        "--disable-default-prompts",
        action="store_true",
        help="Disable the built-in generic prompt bank.",
    )
    parser.add_argument(
        "--min-scene-duration",
        type=float,
        default=2.0,
        help="Minimum scene length in seconds.",
    )
    parser.add_argument(
        "--max-scene-duration",
        type=float,
        default=20.0,
        help="Split long scenes after this duration to preserve coverage.",
    )
    parser.add_argument(
        "--boundary-percentile",
        type=float,
        default=92.0,
        help="Percentile threshold used for semantic scene boundary detection.",
    )
    parser.add_argument(
        "--min-keyframe-gap",
        type=float,
        default=1.0,
        help="Minimum temporal gap in seconds between selected key frames.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.34,
        help="Weight for semantic change in local importance.",
    )
    parser.add_argument(
        "--motion-weight",
        type=float,
        default=0.16,
        help="Weight for motion salience in local importance.",
    )
    parser.add_argument(
        "--histogram-weight",
        type=float,
        default=0.06,
        help="Weight for classical histogram difference in local importance.",
    )
    parser.add_argument(
        "--quality-weight",
        type=float,
        default=0.14,
        help="Weight for image quality in local importance.",
    )
    parser.add_argument(
        "--rarity-weight",
        type=float,
        default=0.14,
        help="Weight for rarity in local importance.",
    )
    parser.add_argument(
        "--prompt-weight",
        type=float,
        default=0.22,
        help="Weight for prompt alignment in local importance.",
    )
    parser.add_argument(
        "--clip-fusion-weight",
        type=float,
        default=0.55,
        help="CLIP weight inside fused embeddings.",
    )
    parser.add_argument(
        "--dino-fusion-weight",
        type=float,
        default=0.45,
        help="DINOv2 weight inside fused embeddings.",
    )
    parser.add_argument(
        "--coverage-weight",
        type=float,
        default=0.55,
        help="Weight for coverage gain during greedy selection.",
    )
    parser.add_argument(
        "--scene-bonus",
        type=float,
        default=0.15,
        help="Bonus for selecting a frame from a scene not yet represented.",
    )
    parser.add_argument(
        "--redundancy-penalty",
        type=float,
        default=0.30,
        help="Penalty on similarity to already selected frames.",
    )
    parser.add_argument(
        "--compression-scene-weight",
        type=float,
        default=0.50,
        help="Scene weight when allocating fixed-budget compression quotas.",
    )
    parser.add_argument(
        "--compression-duration-weight",
        type=float,
        default=0.20,
        help="Scene duration weight when allocating fixed-budget compression quotas.",
    )
    parser.add_argument(
        "--compression-prompt-weight",
        type=float,
        default=0.30,
        help="Prompt-alignment weight when allocating fixed-budget compression quotas.",
    )
    parser.add_argument(
        "--compression-importance-weight",
        type=float,
        default=0.60,
        help="Local importance weight inside per-scene fixed-budget compression.",
    )
    parser.add_argument(
        "--compression-centrality-weight",
        type=float,
        default=0.25,
        help="Scene centrality weight inside per-scene fixed-budget compression.",
    )
    parser.add_argument(
        "--compression-boundary-weight",
        type=float,
        default=0.15,
        help="Boundary strength weight inside per-scene fixed-budget compression.",
    )
    parser.add_argument(
        "--knn-rarity",
        type=int,
        default=5,
        help="Neighborhood size used to estimate rarity.",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=224,
        help="Resize candidates to this square size for scoring and contact sheets.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of sampled candidates.",
    )
    parser.add_argument(
        "--save-contact-sheet",
        action="store_true",
        help="Write a contact sheet of selected key frames.",
    )
    parser.add_argument(
        "--save-raw-stage",
        action="store_true",
        help="When adaptive budgeting is enabled, also save the larger raw selected set.",
    )
    return parser.parse_args()


def robust_minmax(values: Sequence[float], lower: float = 5.0, upper: float = 95.0):
    import numpy as np

    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = float(np.percentile(arr, lower))
    hi = float(np.percentile(arr, upper))
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def l2_normalize(array):
    import numpy as np

    arr = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-8, None)


def load_prompts(prompt_file: Path | None, use_defaults: bool) -> list[str]:
    prompts: list[str] = []
    if use_defaults:
        prompts.extend(DEFAULT_GENERIC_PROMPTS)
    if prompt_file is not None:
        file_prompts = [
            line.strip()
            for line in prompt_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        prompts.extend(file_prompts)
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def sample_video_candidates(
    video_path: Path,
    sample_fps: float,
    thumb_size: int,
    max_samples: int | None,
) -> tuple[list[FrameCandidate], VideoMetadata]:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0.0:
        native_fps = max(sample_fps, 1.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / native_fps if total_frames > 0 else 0.0
    stride = max(1, int(round(native_fps / max(sample_fps, 1e-6))))

    candidates: list[FrameCandidate] = []
    prev_gray = None
    prev_hist = None
    frame_index = 0

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if frame_index % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            thumb_rgb = cv2.resize(
                frame_rgb,
                (thumb_size, thumb_size),
                interpolation=cv2.INTER_AREA,
            )
            flow_rgb = cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2GRAY)
            hist_rgb = cv2.resize(frame_rgb, (96, 96), interpolation=cv2.INTER_AREA)

            if prev_gray is None:
                motion = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                motion = float(np.linalg.norm(flow, axis=2).mean())
            prev_gray = gray

            hist = np.histogramdd(
                hist_rgb.reshape(-1, 3),
                bins=(8, 8, 8),
                range=((0, 256), (0, 256), (0, 256)),
            )[0].astype(np.float32)
            hist = hist.reshape(-1)
            hist /= np.clip(hist.sum(), 1e-8, None)
            if prev_hist is None:
                histogram_diff = 0.0
            else:
                prev_centered = prev_hist - prev_hist.mean()
                hist_centered = hist - hist.mean()
                denom = float(
                    np.linalg.norm(prev_centered) * np.linalg.norm(hist_centered)
                )
                if denom < 1e-8:
                    corr = 1.0 if np.allclose(prev_hist, hist) else 0.0
                else:
                    corr = float(np.dot(prev_centered, hist_centered) / denom)
                histogram_diff = float(1.0 - corr)
            prev_hist = hist

            thumb_gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
            sharpness = float(cv2.Laplacian(thumb_gray, cv2.CV_32F).var())
            exposure = float(max(0.0, 1.0 - abs((thumb_gray.mean() / 255.0) - 0.5) / 0.5))
            quality = sharpness * (0.5 + 0.5 * exposure)
            timestamp_sec = frame_index / native_fps

            candidates.append(
                FrameCandidate(
                    sample_index=len(candidates),
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    motion_raw=motion,
                    histogram_diff_raw=histogram_diff,
                    sharpness_raw=sharpness,
                    exposure_raw=exposure,
                    quality_raw=quality,
                    thumb_rgb=thumb_rgb,
                )
            )
            if max_samples is not None and len(candidates) >= max_samples:
                break
        frame_index += 1

    capture.release()

    metadata = VideoMetadata(
        video_path=str(video_path),
        native_fps=native_fps,
        sample_fps=sample_fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        sampled_frames=len(candidates),
    )
    return candidates, metadata


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_precision(device: str, precision_arg: str):
    try:
        import torch
    except ImportError:
        return None

    if device == "cpu":
        return None
    if precision_arg == "fp32":
        return None
    if precision_arg == "fp16":
        return torch.float16
    if precision_arg == "bf16":
        return torch.bfloat16
    if device == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return None


def autocast_context(torch_module, device: str, precision):
    if precision is None:
        return nullcontext()
    if device == "cuda":
        return torch_module.autocast(device_type="cuda", dtype=precision)
    if device == "mps":
        try:
            return torch_module.autocast(device_type="mps", dtype=precision)
        except (TypeError, RuntimeError, ValueError):
            return nullcontext()
    return nullcontext()


def load_embedding_stack(
    device: str,
    clip_model_name: str,
    dino_model_name: str,
):
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

    cache_key = (device, clip_model_name, dino_model_name)
    cached = _EMBEDDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    dino_model = AutoModel.from_pretrained(dino_model_name).to(device)
    dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
    dino_model.eval()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cached = (clip_model, clip_processor, dino_model, dino_processor)
    _EMBEDDER_CACHE[cache_key] = cached
    return cached


def batch_iterable(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def unwrap_embedding_output(output):
    if hasattr(output, "float"):
        return output
    for attr in ("image_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return value
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden[:, 0]
    raise TypeError(f"Unsupported embedding output type: {type(output)!r}")


def embed_candidates(
    candidates: Sequence[FrameCandidate],
    prompts: Sequence[str],
    device: str,
    precision_arg: str,
    clip_model_name: str,
    dino_model_name: str,
    batch_size: int,
):
    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    images = [Image.fromarray(candidate.thumb_rgb) for candidate in candidates]

    precision = resolve_precision(device, precision_arg)

    clip_model, clip_processor, dino_model, dino_processor = load_embedding_stack(
        device=device,
        clip_model_name=clip_model_name,
        dino_model_name=dino_model_name,
    )

    clip_features: list[np.ndarray] = []
    dino_features: list[np.ndarray] = []

    for image_batch in tqdm(
        list(batch_iterable(images, batch_size)),
        desc="Embedding frames",
        unit="batch",
    ):
        clip_inputs = clip_processor(images=list(image_batch), return_tensors="pt")
        clip_inputs = {key: value.to(device) for key, value in clip_inputs.items()}

        dino_inputs = dino_processor(images=list(image_batch), return_tensors="pt")
        dino_inputs = {key: value.to(device) for key, value in dino_inputs.items()}

        autocast_ctx = autocast_context(torch, device, precision)
        with torch.inference_mode():
            with autocast_ctx:
                clip_out = clip_model.get_image_features(**clip_inputs)
                dino_out = dino_model(**dino_inputs)

        clip_tensor = unwrap_embedding_output(clip_out)
        pooled = unwrap_embedding_output(dino_out)
        clip_features.append(clip_tensor.float().cpu().numpy())
        dino_features.append(pooled.float().cpu().numpy())

    clip_array = l2_normalize(np.concatenate(clip_features, axis=0))
    dino_array = l2_normalize(np.concatenate(dino_features, axis=0))

    prompt_alignment = np.zeros(len(candidates), dtype=np.float32)
    prompt_payload: dict[str, Any] = {"prompts": list(prompts), "used": False}
    if prompts:
        text_inputs = clip_processor(
            text=list(prompts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
        autocast_ctx = autocast_context(torch, device, precision)
        with torch.inference_mode():
            with autocast_ctx:
                text_features = clip_model.get_text_features(**text_inputs)
        text_tensor = unwrap_embedding_output(text_features)
        text_array = l2_normalize(text_tensor.float().cpu().numpy())
        prompt_alignment = (clip_array @ text_array.T).max(axis=1)
        prompt_payload["used"] = True

    return clip_array, dino_array, prompt_alignment, prompt_payload


def compute_semantic_scores(
    candidates: Sequence[FrameCandidate],
    clip_array,
    dino_array,
    prompt_alignment,
    clip_weight: float,
    dino_weight: float,
    semantic_weight: float,
    motion_weight: float,
    histogram_weight: float,
    quality_weight: float,
    rarity_weight: float,
    prompt_weight: float,
    knn_rarity: int,
):
    import numpy as np

    fused = np.concatenate(
        [clip_weight * clip_array, dino_weight * dino_array],
        axis=1,
    )
    fused = l2_normalize(fused)
    similarity = np.clip(fused @ fused.T, -1.0, 1.0)

    semantic_change = np.zeros(len(candidates), dtype=np.float32)
    boundary = np.zeros(len(candidates), dtype=np.float32)

    for index in range(len(candidates)):
        neighbors: list[float] = []
        if index > 0:
            prev_distance = 1.0 - similarity[index, index - 1]
            neighbors.append(prev_distance)
            boundary[index] = prev_distance
        if index + 1 < len(candidates):
            next_distance = 1.0 - similarity[index, index + 1]
            neighbors.append(next_distance)
        semantic_change[index] = float(sum(neighbors) / max(len(neighbors), 1))

    rarity = np.zeros(len(candidates), dtype=np.float32)
    for index in range(len(candidates)):
        row = np.delete(similarity[index], index)
        if row.size == 0:
            rarity[index] = 0.0
            continue
        nearest = np.sort(row)[-min(knn_rarity, row.size) :]
        rarity[index] = 1.0 - float(nearest.mean())

    motion_score = robust_minmax([candidate.motion_raw for candidate in candidates])
    histogram_score = robust_minmax([candidate.histogram_diff_raw for candidate in candidates])
    quality_score = robust_minmax([candidate.quality_raw for candidate in candidates])
    semantic_score = robust_minmax(semantic_change)
    boundary_score = robust_minmax(boundary)
    rarity_score = robust_minmax(rarity)
    prompt_score = robust_minmax(prompt_alignment) if np.any(prompt_alignment) else prompt_alignment

    importance = (
        semantic_weight * semantic_score
        + motion_weight * motion_score
        + histogram_weight * histogram_score
        + quality_weight * quality_score
        + rarity_weight * rarity_score
        + prompt_weight * prompt_score
    )

    for idx, candidate in enumerate(candidates):
        candidate.semantic_change = float(semantic_score[idx])
        candidate.boundary_score = float(boundary_score[idx])
        candidate.motion_score = float(motion_score[idx])
        candidate.histogram_score = float(histogram_score[idx])
        candidate.quality_score = float(quality_score[idx])
        candidate.rarity_score = float(rarity_score[idx])
        candidate.prompt_alignment = float(prompt_score[idx])
        candidate.local_importance = float(importance[idx])

    return fused, similarity, importance, boundary_score


def local_peaks(values) -> list[int]:
    peaks: list[int] = []
    if len(values) < 3:
        return list(range(1, len(values)))
    for index in range(1, len(values) - 1):
        if values[index] >= values[index - 1] and values[index] > values[index + 1]:
            peaks.append(index)
    return peaks


def split_long_scene(
    start: int,
    end: int,
    timestamps: Sequence[float],
    max_scene_duration: float,
) -> list[int]:
    if end - start <= 1:
        return []
    duration = timestamps[end - 1] - timestamps[start]
    if duration <= max_scene_duration:
        return []

    pieces = int(math.ceil(duration / max_scene_duration))
    if pieces <= 1:
        return []
    splits: list[int] = []
    for piece in range(1, pieces):
        ratio = piece / pieces
        split_index = start + int(round((end - start) * ratio))
        if start < split_index < end:
            splits.append(split_index)
    return splits


def detect_scenes(
    candidates: Sequence[FrameCandidate],
    boundary_score,
    boundary_percentile: float,
    min_scene_duration: float,
    max_scene_duration: float,
) -> list[tuple[int, int]]:
    import numpy as np

    if not candidates:
        return []

    timestamps = [candidate.timestamp_sec for candidate in candidates]
    scene_starts = [0]

    peaks = local_peaks(boundary_score)
    threshold = (
        float(np.percentile(boundary_score[1:], boundary_percentile))
        if len(boundary_score) > 2
        else 1.0
    )
    last_start_ts = timestamps[0]

    for peak in peaks:
        if boundary_score[peak] < threshold:
            continue
        if timestamps[peak] - last_start_ts < min_scene_duration:
            continue
        scene_starts.append(peak)
        last_start_ts = timestamps[peak]

    scene_starts.append(len(candidates))
    scene_starts = sorted(set(scene_starts))

    refined_starts = [scene_starts[0]]
    for start, end in zip(scene_starts[:-1], scene_starts[1:]):
        refined_starts.extend(split_long_scene(start, end, timestamps, max_scene_duration))
        refined_starts.append(end)

    refined_starts = sorted(set(refined_starts))
    scenes = list(zip(refined_starts[:-1], refined_starts[1:]))

    for scene_id, (start, end) in enumerate(scenes):
        for index in range(start, end):
            candidates[index].scene_id = scene_id
    return scenes


def compute_adaptive_raw_budget(
    metadata: VideoMetadata,
    candidates: Sequence[FrameCandidate],
    boundary_score,
    final_budget: int,
    adaptive_budget: bool,
    duration_ref_sec: float,
    complexity_weight: float,
    min_raw_keyframes: int,
    max_raw_keyframes: int,
    boundary_percentile: float,
) -> tuple[int, dict[str, float]]:
    import numpy as np

    diagnostics = {
        "duration_factor": 1.0,
        "complexity_factor": 1.0,
        "boundary_density": 0.0,
        "raw_budget": float(final_budget),
    }
    if not adaptive_budget:
        raw_budget = min(final_budget, len(candidates))
        diagnostics["raw_budget"] = float(raw_budget)
        return raw_budget, diagnostics

    effective_ref = max(duration_ref_sec, 1e-6)
    duration_factor = math.sqrt(max(metadata.duration_sec, 1e-6) / effective_ref)
    diagnostics["duration_factor"] = float(duration_factor)

    if len(boundary_score) > 2:
        threshold = float(np.percentile(boundary_score[1:], boundary_percentile))
        boundary_density = float((boundary_score[1:] >= threshold).mean())
    else:
        boundary_density = 0.0
    complexity_factor = 1.0 + complexity_weight * boundary_density
    diagnostics["boundary_density"] = boundary_density
    diagnostics["complexity_factor"] = float(complexity_factor)

    raw_budget = int(round(final_budget * duration_factor * complexity_factor))
    raw_budget = max(raw_budget, max(final_budget, min_raw_keyframes))
    raw_budget = min(raw_budget, max_raw_keyframes, len(candidates))
    diagnostics["raw_budget"] = float(raw_budget)
    return raw_budget, diagnostics


def choose_scene_seeds(
    candidates: Sequence[FrameCandidate],
    similarity,
    scenes: Sequence[tuple[int, int]],
    num_keyframes: int,
) -> list[int]:
    import numpy as np

    scene_items: list[tuple[float, int]] = []
    for start, end in scenes:
        if end <= start:
            continue
        indices = list(range(start, end))
        scene_similarity = similarity[np.ix_(indices, indices)]
        centrality = scene_similarity.mean(axis=1)
        importance = np.asarray([candidates[idx].local_importance for idx in indices], dtype=np.float32)
        representative_score = 0.65 * importance + 0.35 * centrality
        best_offset = int(np.argmax(representative_score))
        best_index = indices[best_offset]
        scene_strength = float(importance.mean() + 0.35 * importance.max() + 0.05 * math.log1p(len(indices)))
        scene_items.append((scene_strength, best_index))

    scene_items.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in scene_items[: min(num_keyframes, len(scene_items))]]


def is_far_enough(
    candidate_idx: int,
    selected: Sequence[int],
    candidates: Sequence[FrameCandidate],
    min_keyframe_gap: float,
) -> bool:
    timestamp = candidates[candidate_idx].timestamp_sec
    return all(abs(timestamp - candidates[idx].timestamp_sec) >= min_keyframe_gap for idx in selected)


def greedy_select(
    candidates: Sequence[FrameCandidate],
    similarity,
    importance,
    scenes: Sequence[tuple[int, int]],
    num_keyframes: int,
    coverage_weight: float,
    scene_bonus: float,
    redundancy_penalty: float,
    min_keyframe_gap: float,
) -> list[int]:
    import numpy as np

    if not candidates:
        return []

    scene_seed_indices = choose_scene_seeds(candidates, similarity, scenes, num_keyframes)
    selected: list[int] = []
    current_coverage = np.zeros(len(candidates), dtype=np.float32)
    represented_scenes: set[int] = set()

    for seed in scene_seed_indices:
        if len(selected) >= num_keyframes:
            break
        if not is_far_enough(seed, selected, candidates, min_keyframe_gap):
            continue
        selected.append(seed)
        current_coverage = np.maximum(current_coverage, similarity[:, seed])
        represented_scenes.add(candidates[seed].scene_id)
        candidates[seed].selection_gain = float(importance[seed])

    while len(selected) < min(num_keyframes, len(candidates)):
        best_index = None
        best_gain = -1e9

        for index in range(len(candidates)):
            if index in selected:
                continue
            if not is_far_enough(index, selected, candidates, min_keyframe_gap):
                continue

            coverage_gain = np.maximum(current_coverage, similarity[:, index]).sum() - current_coverage.sum()
            coverage_gain = float(coverage_gain / max(len(candidates), 1))
            redundancy = max((similarity[index, chosen] for chosen in selected), default=0.0)
            scene_reward = scene_bonus if candidates[index].scene_id not in represented_scenes else 0.0

            gain = (
                float(importance[index])
                + coverage_weight * coverage_gain
                + scene_reward
                - redundancy_penalty * float(redundancy)
            )

            if gain > best_gain:
                best_gain = gain
                best_index = index

        if best_index is None:
            for index in range(len(candidates)):
                if index in selected:
                    continue
                coverage_gain = np.maximum(current_coverage, similarity[:, index]).sum() - current_coverage.sum()
                coverage_gain = float(coverage_gain / max(len(candidates), 1))
                redundancy = max((similarity[index, chosen] for chosen in selected), default=0.0)
                scene_reward = scene_bonus if candidates[index].scene_id not in represented_scenes else 0.0
                gain = (
                    float(importance[index])
                    + coverage_weight * coverage_gain
                    + scene_reward
                    - redundancy_penalty * float(redundancy)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_index = index

        if best_index is None:
            break

        selected.append(best_index)
        current_coverage = np.maximum(current_coverage, similarity[:, best_index])
        represented_scenes.add(candidates[best_index].scene_id)
        candidates[best_index].selection_gain = float(best_gain)

    selected.sort(key=lambda idx: candidates[idx].timestamp_sec)
    for rank, index in enumerate(selected, start=1):
        candidates[index].raw_selected_rank = rank
    return selected


def allocate_scene_quotas(
    scene_scores: dict[int, float],
    available_counts: dict[int, int],
    budget: int,
) -> dict[int, int]:
    ranked_scenes = sorted(scene_scores, key=lambda scene_id: scene_scores[scene_id], reverse=True)
    if budget <= 0 or not ranked_scenes:
        return {}

    if len(ranked_scenes) >= budget:
        return {scene_id: 1 for scene_id in ranked_scenes[:budget]}

    quotas = {scene_id: 1 for scene_id in ranked_scenes}
    remaining = budget - len(ranked_scenes)

    while remaining > 0:
        best_scene = None
        best_score = -1e9
        for scene_id in ranked_scenes:
            if quotas[scene_id] >= available_counts.get(scene_id, 0):
                continue
            gain = scene_scores[scene_id] / (quotas[scene_id] + 0.5)
            if gain > best_score:
                best_score = gain
                best_scene = scene_id
        if best_scene is None:
            break
        quotas[best_scene] += 1
        remaining -= 1
    return quotas


def compress_to_fixed_budget(
    candidates: Sequence[FrameCandidate],
    raw_selected_indices: Sequence[int],
    scenes: Sequence[tuple[int, int]],
    similarity,
    final_budget: int,
    compression_scene_weight: float,
    compression_duration_weight: float,
    compression_prompt_weight: float,
    compression_importance_weight: float,
    compression_centrality_weight: float,
    compression_boundary_weight: float,
) -> list[int]:
    import numpy as np

    if len(raw_selected_indices) <= final_budget:
        final_indices = sorted(raw_selected_indices, key=lambda idx: candidates[idx].timestamp_sec)
        for rank, index in enumerate(final_indices, start=1):
            candidates[index].compression_score = candidates[index].local_importance
            candidates[index].selected_rank = rank
        return final_indices

    raw_by_scene: dict[int, list[int]] = {}
    for index in raw_selected_indices:
        raw_by_scene.setdefault(candidates[index].scene_id, []).append(index)

    total_duration = max(
        sum(
            max(
                1e-6,
                candidates[end - 1].timestamp_sec - candidates[start].timestamp_sec
                if end - start > 1
                else 1e-6,
            )
            for start, end in scenes
        ),
        1e-6,
    )

    scene_scores: dict[int, float] = {}
    available_counts: dict[int, int] = {}
    for scene_id, scene_indices in raw_by_scene.items():
        start, end = scenes[scene_id]
        duration = (
            candidates[end - 1].timestamp_sec - candidates[start].timestamp_sec
            if end - start > 1
            else 1e-6
        )
        duration_score = duration / total_duration
        scene_candidates = candidates[start:end]
        scene_importance = float(
            sum(candidate.local_importance for candidate in scene_candidates) / max(len(scene_candidates), 1)
        )
        scene_prompt = float(
            sum(candidate.prompt_alignment for candidate in scene_candidates) / max(len(scene_candidates), 1)
        )
        scene_scores[scene_id] = (
            compression_scene_weight * scene_importance
            + compression_duration_weight * duration_score
            + compression_prompt_weight * scene_prompt
        )
        available_counts[scene_id] = len(scene_indices)

    quotas = allocate_scene_quotas(scene_scores, available_counts, final_budget)

    chosen: list[int] = []
    remainder_pool: list[tuple[float, int]] = []
    for scene_id, scene_indices in raw_by_scene.items():
        if not scene_indices:
            continue
        scene_similarity = similarity[np.ix_(scene_indices, scene_indices)]
        centrality = scene_similarity.mean(axis=1)
        scored_indices: list[tuple[float, int]] = []
        for offset, index in enumerate(scene_indices):
            score = (
                compression_importance_weight * candidates[index].local_importance
                + compression_centrality_weight * float(centrality[offset])
                + compression_boundary_weight * candidates[index].boundary_score
            )
            candidates[index].compression_score = float(score)
            scored_indices.append((float(score), index))
        scored_indices.sort(key=lambda item: item[0], reverse=True)
        keep = quotas.get(scene_id, 0)
        chosen.extend(index for _, index in scored_indices[:keep])
        remainder_pool.extend(scored_indices[keep:])

    if len(chosen) < final_budget:
        remainder_pool.sort(key=lambda item: item[0], reverse=True)
        chosen.extend(index for _, index in remainder_pool[: final_budget - len(chosen)])

    final_indices = sorted(chosen[:final_budget], key=lambda idx: candidates[idx].timestamp_sec)
    for rank, index in enumerate(final_indices, start=1):
        candidates[index].selected_rank = rank
    return final_indices


def save_selected_frames(
    video_path: Path,
    output_dir: Path,
    candidates: Sequence[FrameCandidate],
    selected_indices: Sequence[int],
    subdir: str = "frames",
) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to reopen video for saving frames: {video_path}")

    written_files: list[str] = []
    frames_dir = output_dir / subdir
    frames_dir.mkdir(parents=True, exist_ok=True)

    for rank, index in enumerate(selected_indices, start=1):
        candidate = candidates[index]
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
        ok, frame_bgr = capture.read()
        if not ok:
            continue
        filename = (
            f"kf_{rank:03d}_t{candidate.timestamp_sec:07.2f}_f{candidate.frame_index:07d}.jpg"
        )
        destination = frames_dir / filename
        cv2.imwrite(str(destination), frame_bgr)
        written_files.append(str(destination))

    capture.release()
    return written_files


def save_contact_sheet(
    output_dir: Path,
    candidates: Sequence[FrameCandidate],
    selected_indices: Sequence[int],
) -> str:
    from PIL import Image, ImageDraw

    if not selected_indices:
        return ""

    tile = candidates[selected_indices[0]].thumb_rgb.shape[0]
    caption_height = 28
    columns = min(4, len(selected_indices))
    rows = int(math.ceil(len(selected_indices) / columns))
    canvas = Image.new(
        "RGB",
        (columns * tile, rows * (tile + caption_height)),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)

    for position, index in enumerate(selected_indices):
        row = position // columns
        col = position % columns
        x = col * tile
        y = row * (tile + caption_height)
        tile_image = Image.fromarray(candidates[index].thumb_rgb)
        canvas.paste(tile_image, (x, y))
        label = f"#{position + 1}  {candidates[index].timestamp_sec:.1f}s"
        draw.text((x + 6, y + tile + 6), label, fill=(20, 20, 20))

    contact_path = output_dir / "contact_sheet.jpg"
    canvas.save(contact_path, quality=95)
    return str(contact_path)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    metadata: VideoMetadata,
    budget_diagnostics: dict[str, float],
    prompt_payload: dict[str, Any],
    candidates: Sequence[FrameCandidate],
    raw_selected_indices: Sequence[int],
    selected_indices: Sequence[int],
    raw_written_files: Sequence[str],
    written_files: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }

    csv_path = output_dir / "candidate_scores.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0].to_serializable().keys()))
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate.to_serializable())

    payload = {
        "config": config,
        "video_metadata": metadata.__dict__,
        "budget_diagnostics": budget_diagnostics,
        "prompt_info": prompt_payload,
        "raw_selected_indices": list(raw_selected_indices),
        "selected_indices": list(selected_indices),
        "raw_selected_frames": [
            candidates[index].to_serializable() | {"saved_path": raw_written_files[position]}
            for position, index in enumerate(raw_selected_indices)
            if position < len(raw_written_files)
        ],
        "selected_frames": [
            candidates[index].to_serializable() | {"saved_path": written_files[position]}
            for position, index in enumerate(selected_indices)
            if position < len(written_files)
        ],
        "num_scenes": len({candidate.scene_id for candidate in candidates if candidate.scene_id >= 0}),
    }
    (output_dir / "selection_metadata.json").write_text(json.dumps(payload, indent=2))


def main() -> int:
    import numpy as np

    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompt_file, use_defaults=not args.disable_default_prompts)

    print("Sampling candidate frames...", file=sys.stderr)
    try:
        candidates, metadata = sample_video_candidates(
            video_path=args.video,
            sample_fps=args.sample_fps,
            thumb_size=args.thumb_size,
            max_samples=args.max_samples,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency while sampling video frames. Install packages from "
            "requirements.txt before running the extractor."
        ) from exc
    if not candidates:
        raise RuntimeError("No frames were sampled from the input video.")

    device = choose_device(args.device)
    print(f"Embedding {len(candidates)} sampled frames on {device}...", file=sys.stderr)
    try:
        clip_array, dino_array, prompt_alignment, prompt_payload = embed_candidates(
            candidates=candidates,
            prompts=prompts,
            device=device,
            precision_arg=args.precision,
            clip_model_name=args.clip_model_name,
            dino_model_name=args.dino_model_name,
            batch_size=args.batch_size,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency while loading CLIP/DINOv2 components. Install packages "
            "from requirements.txt before running the extractor."
        ) from exc

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

    print("Detecting semantic scenes...", file=sys.stderr)
    scenes = detect_scenes(
        candidates=candidates,
        boundary_score=boundary_score,
        boundary_percentile=args.boundary_percentile,
        min_scene_duration=args.min_scene_duration,
        max_scene_duration=args.max_scene_duration,
    )

    raw_budget, budget_diagnostics = compute_adaptive_raw_budget(
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

    print(
        f"Selecting coverage-aware raw set ({raw_budget} frames before compression)...",
        file=sys.stderr,
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

    print(
        f"Compressing raw set to fixed decoder budget ({args.num_keyframes})...",
        file=sys.stderr,
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

    raw_written_files: list[str] = []
    if args.save_raw_stage and len(raw_selected_indices) > len(selected_indices):
        raw_written_files = save_selected_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            candidates=candidates,
            selected_indices=raw_selected_indices,
            subdir="frames_raw",
        )

    written_files = save_selected_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        candidates=candidates,
        selected_indices=selected_indices,
    )

    np.save(args.output_dir / "selected_fused_embeddings.npy", fused[selected_indices])
    np.save(args.output_dir / "selected_clip_embeddings.npy", clip_array[selected_indices])
    np.save(args.output_dir / "selected_dino_embeddings.npy", dino_array[selected_indices])
    if args.save_raw_stage and len(raw_selected_indices) > len(selected_indices):
        np.save(args.output_dir / "raw_selected_fused_embeddings.npy", fused[raw_selected_indices])
        np.save(args.output_dir / "raw_selected_clip_embeddings.npy", clip_array[raw_selected_indices])
        np.save(args.output_dir / "raw_selected_dino_embeddings.npy", dino_array[raw_selected_indices])

    if args.save_contact_sheet:
        contact_sheet = save_contact_sheet(
            output_dir=args.output_dir,
            candidates=candidates,
            selected_indices=selected_indices,
        )
        if contact_sheet:
            print(f"Saved contact sheet to {contact_sheet}", file=sys.stderr)

    write_metadata(
        output_dir=args.output_dir,
        args=args,
        metadata=metadata,
        budget_diagnostics=budget_diagnostics,
        prompt_payload=prompt_payload,
        candidates=candidates,
        raw_selected_indices=raw_selected_indices,
        selected_indices=selected_indices,
        raw_written_files=raw_written_files,
        written_files=written_files,
    )

    print(
        json.dumps(
            {
                "raw_selected_keyframes": len(raw_selected_indices),
                "selected_keyframes": len(selected_indices),
                "output_dir": str(args.output_dir),
                "sampled_frames": len(candidates),
                "device": device,
                "num_scenes": len(scenes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
