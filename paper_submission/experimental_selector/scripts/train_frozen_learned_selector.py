from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_long_video_evidence_reranker import (  # type: ignore  # noqa: E402
    DEFAULT_ADAPTED_RUN,
    DEFAULT_FALLBACK_PROPOSALS,
    DEFAULT_PROPOSALS,
    DEFAULT_WORKSPACE,
    Candidate,
    SplitCiderScorer,
    build_target_idf,
    candidate_pool_oracle,
    compute_candidate_features,
    load_json,
    make_ground_truth,
    prepare_split,
    save_json,
    score_rows,
    serialise,
    source_counts,
    switch_rows,
    tokenize,
)


DEFAULT_OUTPUT = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "frozen_learned_selector_v1"
)
DEFAULT_MSRVTT_CAPTIONS_ROOT = Path("/Users/aglooney03/video_captioning/data/processed/captions")

BASE_FEATURES = [
    "model_rel",
    "model_rank_penalty",
    "visual_alignment",
    "evidence_support",
    "target_lexical",
    "domain_count",
    "source_bias",
    "is_bart",
    "is_visual",
    "length_penalty",
    "repetition_penalty",
    "bad_penalty",
]
EXTRA_FEATURES = [
    "model_score_abs",
    "model_rank_abs",
    "caption_length",
    "unique_token_ratio",
    "domain_density",
    "is_adaptive_blip",
    "is_evidence_summary",
    "consensus_mean_all",
    "consensus_max_all",
    "consensus_mean_bart",
    "consensus_mean_visual",
    "consensus_to_baseline",
    "consensus_cross_source",
    "consensus_support_025",
    "clip_video_mean",
    "clip_video_max",
    "clip_video_rel",
    "clip_video_zscore",
    "clip_video_rank_penalty",
]
FEATURE_NAMES = BASE_FEATURES + EXTRA_FEATURES
DEFAULT_CLIP_TEXT_MODEL = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a video-grouped, frozen learned selector over the LV-ECR "
            "candidate pool. Labels are sentence-level CIDEr on development "
            "splits only; held-out test references are used only for final scoring."
        )
    )
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument(
        "--refs-root",
        default="",
        help=(
            "Optional directory containing reference JSON files. Defaults to "
            "<workspace-root>/processed_clean/captions."
        ),
    )
    parser.add_argument(
        "--refs-filename-template",
        default="{split}_captions.json",
        help=(
            "Filename template under --refs-root. Use "
            "'{split}_captions_normalized.json' for the v3 MSR-VTT-style references."
        ),
    )
    parser.add_argument("--adapted-run-dir", default=str(DEFAULT_ADAPTED_RUN))
    parser.add_argument("--proposals-json", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--fallback-proposals-json", default=str(DEFAULT_FALLBACK_PROPOSALS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dev-splits", default="train,val")
    parser.add_argument(
        "--candidate-sources",
        choices=["bart", "bart_blip", "bart_blip_summary"],
        default="bart_blip_summary",
    )
    parser.add_argument("--max-blip-candidates", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--learner", choices=["ridge", "pairwise", "extra_trees"], default="ridge")
    parser.add_argument("--min-pair-delta", type=float, default=0.0)
    parser.add_argument("--msrvtt-pretrain-captions-root", default=str(DEFAULT_MSRVTT_CAPTIONS_ROOT))
    parser.add_argument(
        "--msrvtt-pretrain-videos",
        type=int,
        default=0,
        help="Number of MSR-VTT videos to use for reference-consensus pretraining; 0 disables pretraining.",
    )
    parser.add_argument("--msrvtt-pretrain-splits", default="train,val")
    parser.add_argument(
        "--pretrain-weight",
        type=float,
        default=0.10,
        help="Sample/pair weight for MSR-VTT pretraining examples relative to Dattalion examples.",
    )
    parser.add_argument(
        "--msrvtt-beam-run-dir",
        default="",
        help=(
            "Directory containing beam_headroom_{split}_*.json files generated "
            "from the tuned MSR-VTT captioner. When provided, these generated "
            "beam candidates take precedence over reference-consensus proxy "
            "pretraining."
        ),
    )
    parser.add_argument(
        "--msrvtt-beam-splits",
        default="train,val",
        help="Comma-separated MSR-VTT beam splits to use for generated-candidate pretraining.",
    )
    parser.add_argument(
        "--enable-clip-video-text-features",
        action="store_true",
        help=(
            "Attach reference-free CLIP text/video compatibility features. This "
            "uses cached 512-d CLIP visual features from fused .npy files and a "
            "CLIP text encoder."
        ),
    )
    parser.add_argument(
        "--clip-feature-roots",
        default="",
        help=(
            "Comma-separated roots containing fused CLIP/DINO .npy files. If "
            "omitted with --enable-clip-video-text-features, known local "
            "Dattalion and MSR-VTT fused-feature roots are searched."
        ),
    )
    parser.add_argument("--clip-text-model-name", default=DEFAULT_CLIP_TEXT_MODEL)
    parser.add_argument("--clip-feature-device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--clip-text-batch-size", type=int, default=64)
    parser.add_argument("--clip-progress-every", type=int, default=50)
    parser.add_argument(
        "--alphas",
        default="0.01,0.03,0.1,0.3,1,3,10,30,100",
        help=(
            "Comma-separated regularization strengths for grouped CV. Ridge uses "
            "these as alpha values; pairwise logistic ranking uses C = 1 / alpha."
        ),
    )
    parser.add_argument(
        "--switch-margins",
        default="0.0,0.01,0.025,0.05",
        help="Comma-separated predicted-CIDEr margins required before replacing the top BART caption.",
    )
    parser.add_argument("--current-reported-cider", type=float, default=0.20687050726627432)
    return parser.parse_args()


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value")
    return values


def parse_csv_list(raw: str, name: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def parse_path_list(raw: str) -> list[Path]:
    return [Path(item.strip()).expanduser().resolve() for item in raw.split(",") if item.strip()]


def all_rows(candidates_by_video: dict[str, list[Candidate]], ids: list[str] | None = None) -> list[Candidate]:
    video_ids = ids if ids is not None else sorted(candidates_by_video)
    return [row for video_id in video_ids for row in candidates_by_video[video_id]]


def fit_consensus_vectorizer(*candidate_maps: dict[str, list[Candidate]]) -> TfidfVectorizer:
    texts = [row.caption for candidate_map in candidate_maps for row in all_rows(candidate_map)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)
    vectorizer.fit(texts)
    return vectorizer


def mean_similarity(sims: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(np.mean(sims[indices]))


def attach_consensus_features(
    candidates_by_video: dict[str, list[Candidate]],
    baseline_by_video: dict[str, Candidate],
    vectorizer: TfidfVectorizer,
) -> None:
    for video_id, rows in candidates_by_video.items():
        captions = [row.caption for row in rows]
        matrix = vectorizer.transform(captions).toarray().astype(float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > 0.0,
        )
        sims = np.nan_to_num(normalized @ normalized.T, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_caption = baseline_by_video[video_id].caption
        baseline_idx = next(
            (idx for idx, row in enumerate(rows) if row.caption == baseline_caption),
            0,
        )
        bart_indices = [idx for idx, row in enumerate(rows) if row.source == "bart"]
        visual_indices = [idx for idx, row in enumerate(rows) if row.source != "bart"]

        for idx, row in enumerate(rows):
            other_indices = [j for j in range(len(rows)) if j != idx]
            other_bart = [j for j in bart_indices if j != idx]
            other_visual = [j for j in visual_indices if j != idx]
            cross_source = [j for j in other_indices if rows[j].source != row.source]
            other_sims = sims[idx, other_indices] if other_indices else np.array([])
            tokens = tokenize(row.caption)
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            row.features.update(
                {
                    "model_score_abs": float(row.model_score if row.model_score is not None else -99.0),
                    "model_rank_abs": float(row.model_rank if row.model_rank is not None else 99.0),
                    "caption_length": float(len(tokens)),
                    "unique_token_ratio": float(unique_ratio),
                    "domain_density": float(row.features.get("domain_count", 0.0) / math.sqrt(max(len(tokens), 1))),
                    "is_adaptive_blip": 1.0 if row.source == "adaptive_blip" else 0.0,
                    "is_evidence_summary": 1.0 if row.source == "evidence_summary" else 0.0,
                    "consensus_mean_all": float(np.mean(other_sims)) if other_indices else 0.0,
                    "consensus_max_all": float(np.max(other_sims)) if other_indices else 0.0,
                    "consensus_mean_bart": mean_similarity(sims[idx], other_bart),
                    "consensus_mean_visual": mean_similarity(sims[idx], other_visual),
                    "consensus_to_baseline": float(sims[idx, baseline_idx]),
                    "consensus_cross_source": mean_similarity(sims[idx], cross_source),
                    "consensus_support_025": float(np.sum(other_sims >= 0.25)) if other_indices else 0.0,
                }
            )


def resolved_video_id(video_id: str) -> str:
    if "::" in video_id:
        return video_id.split("::")[-1]
    return video_id


def choose_clip_device(device_arg: str):
    import torch

    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_clip_feature_roots(workspace: Path) -> list[Path]:
    candidates = [
        workspace / "features" / "tascc_fused_adaptive120_all",
        workspace / "features" / "tascc_fused",
        workspace / "features" / "tascc_fused_test_adaptive80",
        workspace / "features" / "tascc_fused_trainval_adaptive80",
        Path("/Users/aglooney03/Video-Summarization/features/tascc_fused"),
        Path("/Users/aglooney03/Video-Summarization/datas/feats/tascc_fused"),
        Path(
            "/Users/aglooney03/Library/CloudStorage/GoogleDrive-aidanlooney@g.harvard.edu/My Drive/"
            "undergrad-research/video_captioning_project/Video-Summarization/datas/feats/tascc_fused"
        ),
    ]
    existing: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.exists():
            continue
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


class ClipTextVideoScorer:
    def __init__(
        self,
        feature_roots: list[Path],
        model_name: str,
        device_arg: str,
        batch_size: int,
    ) -> None:
        if not feature_roots:
            raise ValueError("CLIP video-text features requested but no feature roots are available.")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        self.feature_roots = feature_roots
        self.device = choose_clip_device(device_arg)
        self.batch_size = batch_size
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.text_cache: dict[str, np.ndarray] = {}
        self.video_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def find_feature_path(self, video_id: str) -> Path | None:
        raw_id = resolved_video_id(video_id)
        for root in self.feature_roots:
            candidate = root / f"{raw_id}.npy"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=float), where=norms > 0.0)

    @staticmethod
    def normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 0.0:
            return np.zeros_like(vector, dtype=float)
        return vector.astype(float) / float(norm)

    def video_vectors(self, video_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        raw_id = resolved_video_id(video_id)
        if raw_id in self.video_cache:
            return self.video_cache[raw_id]
        path = self.find_feature_path(video_id)
        if path is None:
            return None
        arr = np.load(path)
        if arr.ndim == 1:
            clip = arr[:512].reshape(1, -1)
        else:
            clip = arr[:, :512]
        frame_vectors = self.normalize_rows(np.asarray(clip, dtype=float))
        mean_vector = self.normalize_vector(frame_vectors.mean(axis=0))
        self.video_cache[raw_id] = (mean_vector, frame_vectors)
        return self.video_cache[raw_id]

    def encode_texts(self, captions: list[str]) -> np.ndarray:
        missing = []
        for caption in captions:
            key = caption.lower()
            if key not in self.text_cache:
                missing.append(caption)
        if missing:
            import torch

            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    output = self.model.get_text_features(**encoded)
                if hasattr(output, "pooler_output"):
                    features = output.pooler_output
                elif hasattr(output, "last_hidden_state"):
                    features = output.last_hidden_state[:, 0, :]
                else:
                    features = output
                features_np = features.detach().cpu().numpy().astype(float)
                features_np = self.normalize_rows(features_np)
                for caption, vector in zip(batch, features_np):
                    self.text_cache[caption.lower()] = vector
        return np.vstack([self.text_cache[caption.lower()] for caption in captions])


def attach_clip_video_text_features(
    candidates_by_video: dict[str, list[Candidate]],
    baseline_by_video: dict[str, Candidate],
    scorer: ClipTextVideoScorer,
    label: str,
    progress_every: int,
) -> None:
    attached = 0
    missing = 0
    for index, (video_id, rows) in enumerate(candidates_by_video.items(), start=1):
        video_vectors = scorer.video_vectors(video_id)
        if video_vectors is None:
            missing += 1
            continue
        mean_vector, frame_vectors = video_vectors
        captions = [row.caption for row in rows]
        text_vectors = scorer.encode_texts(captions)
        mean_scores = text_vectors @ mean_vector
        frame_scores = text_vectors @ frame_vectors.T
        max_scores = frame_scores.max(axis=1) if frame_scores.size else mean_scores
        baseline_caption = baseline_by_video[video_id].caption
        baseline_idx = next(
            (idx for idx, row in enumerate(rows) if row.caption == baseline_caption),
            0,
        )
        mean_center = float(np.mean(mean_scores))
        mean_std = float(np.std(mean_scores))
        ranks = np.argsort(np.argsort(-mean_scores))
        for idx, row in enumerate(rows):
            row.features.update(
                {
                    "clip_video_mean": float(mean_scores[idx]),
                    "clip_video_max": float(max_scores[idx]),
                    "clip_video_rel": float(mean_scores[idx] - mean_scores[baseline_idx]),
                    "clip_video_zscore": float((mean_scores[idx] - mean_center) / mean_std) if mean_std > 1e-12 else 0.0,
                    "clip_video_rank_penalty": -float(ranks[idx]),
                }
            )
        attached += 1
        if progress_every > 0 and index % progress_every == 0:
            print(
                f"[clip_features:{label}] processed={index} attached={attached} missing={missing}",
                flush=True,
            )
    if progress_every > 0:
        print(
            f"[clip_features:{label}] done processed={len(candidates_by_video)} attached={attached} missing={missing}",
            flush=True,
        )


def feature_matrix(rows: list[Candidate]) -> np.ndarray:
    return np.array(
        [[float(row.features.get(name, 0.0)) for name in FEATURE_NAMES] for row in rows],
        dtype=float,
    )


class RidgeRanker:
    def __init__(self, alpha: float) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, solver="lsqr")

    def fit(
        self,
        rows: list[Candidate],
        pretrain_rows: list[Candidate] | None = None,
        pretrain_weight: float = 0.0,
    ) -> "RidgeRanker":
        training_rows = list(rows)
        weights = [1.0 for _ in training_rows]
        if pretrain_rows and pretrain_weight > 0.0:
            training_rows.extend(pretrain_rows)
            weights.extend([float(pretrain_weight) for _ in pretrain_rows])
        x_scaled = self.scaler.fit_transform(feature_matrix(training_rows))
        y = np.array([row.sentence_cider for row in training_rows], dtype=float)
        self.model.fit(x_scaled, y, sample_weight=np.array(weights, dtype=float))
        return self

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(x_raw))


class PairwiseRanker:
    def __init__(self, alpha: float, min_pair_delta: float, seed: int) -> None:
        self.alpha = alpha
        self.min_pair_delta = min_pair_delta
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=1.0 / max(alpha, 1e-12),
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=seed,
        )

    def fit(
        self,
        rows: list[Candidate],
        pretrain_rows: list[Candidate] | None = None,
        pretrain_weight: float = 0.0,
    ) -> "PairwiseRanker":
        training_rows = list(rows)
        if pretrain_rows and pretrain_weight > 0.0:
            training_rows.extend(pretrain_rows)
        x_raw = feature_matrix(training_rows)
        x_scaled = self.scaler.fit_transform(x_raw)
        labels = np.array([row.sentence_cider for row in training_rows], dtype=float)
        cursor = len(rows)
        pair_x, pair_y, pair_weights = build_pairwise_examples(
            rows,
            x_scaled[:cursor],
            labels[:cursor],
            min_delta=self.min_pair_delta,
            example_weight=1.0,
        )
        if pretrain_rows and pretrain_weight > 0.0:
            pre_x, pre_y, pre_weights = build_pairwise_examples(
                pretrain_rows,
                x_scaled[cursor:],
                labels[cursor:],
                min_delta=self.min_pair_delta,
                example_weight=pretrain_weight,
            )
            pair_x = np.vstack([pair_x, pre_x])
            pair_y = np.concatenate([pair_y, pre_y])
            pair_weights = np.concatenate([pair_weights, pre_weights])
        self.model.fit(pair_x, pair_y, sample_weight=pair_weights)
        return self

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x_raw)
        return x_scaled @ self.model.coef_.reshape(-1)


class ExtraTreesRanker:
    def __init__(self, alpha: float, seed: int) -> None:
        min_samples_leaf = int(max(1, round(alpha)))
        self.model = ExtraTreesRegressor(
            n_estimators=600,
            min_samples_leaf=min_samples_leaf,
            max_features=0.8,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(
        self,
        rows: list[Candidate],
        pretrain_rows: list[Candidate] | None = None,
        pretrain_weight: float = 0.0,
    ) -> "ExtraTreesRanker":
        training_rows = list(rows)
        weights = [1.0 for _ in training_rows]
        if pretrain_rows and pretrain_weight > 0.0:
            training_rows.extend(pretrain_rows)
            weights.extend([float(pretrain_weight) for _ in pretrain_rows])
        x_raw = feature_matrix(training_rows)
        y = np.array([row.sentence_cider for row in training_rows], dtype=float)
        self.model.fit(x_raw, y, sample_weight=np.array(weights, dtype=float))
        return self

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        return self.model.predict(x_raw)


def build_pairwise_examples(
    rows: list[Candidate],
    x_scaled: np.ndarray,
    labels: np.ndarray,
    min_delta: float,
    example_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_features: list[np.ndarray] = []
    pair_labels: list[int] = []
    pair_weights: list[float] = []
    by_video: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_video.setdefault(row.video_id, []).append(idx)

    for indices in by_video.values():
        for left_pos, left_idx in enumerate(indices):
            for right_idx in indices[left_pos + 1 :]:
                delta = labels[left_idx] - labels[right_idx]
                if abs(delta) <= min_delta:
                    continue
                high, low = (left_idx, right_idx) if delta > 0 else (right_idx, left_idx)
                pair_features.append(x_scaled[high] - x_scaled[low])
                pair_labels.append(1)
                pair_weights.append(float(example_weight))
                pair_features.append(x_scaled[low] - x_scaled[high])
                pair_labels.append(0)
                pair_weights.append(float(example_weight))
    if not pair_features:
        raise ValueError("No pairwise training examples were created; lower --min-pair-delta.")
    return (
        np.vstack(pair_features),
        np.array(pair_labels, dtype=int),
        np.array(pair_weights, dtype=float),
    )


def train_model(
    rows: list[Candidate],
    alpha: float,
    learner: str,
    min_pair_delta: float,
    seed: int,
    pretrain_rows: list[Candidate] | None = None,
    pretrain_weight: float = 0.0,
):
    if learner == "extra_trees":
        return ExtraTreesRanker(alpha=alpha, seed=seed).fit(
            rows,
            pretrain_rows=pretrain_rows,
            pretrain_weight=pretrain_weight,
        )
    if learner == "pairwise":
        return PairwiseRanker(alpha=alpha, min_pair_delta=min_pair_delta, seed=seed).fit(
            rows,
            pretrain_rows=pretrain_rows,
            pretrain_weight=pretrain_weight,
        )
    return RidgeRanker(alpha=alpha).fit(
        rows,
        pretrain_rows=pretrain_rows,
        pretrain_weight=pretrain_weight,
    )


def select_with_model(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    baseline_by_video: dict[str, Candidate],
    model,
    switch_margin: float,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for video_id in refs:
        rows = candidates_by_video[video_id]
        predictions = model.predict(feature_matrix(rows))
        for row, score in zip(rows, predictions):
            row.features["learned_score"] = float(score)
        baseline_caption = baseline_by_video[video_id].caption
        baseline_idx = next(
            (idx for idx, row in enumerate(rows) if row.caption == baseline_caption),
            0,
        )
        best_idx = int(np.argmax(predictions))
        if best_idx != baseline_idx and predictions[best_idx] - predictions[baseline_idx] < switch_margin:
            selected.append(rows[baseline_idx])
        else:
            selected.append(rows[best_idx])
    return selected


def select_baselines(
    refs: dict[str, list[str]],
    baseline_by_video: dict[str, Candidate],
) -> list[Candidate]:
    return [baseline_by_video[video_id] for video_id in refs]


def grouped_cv_predictions(
    dev_refs: dict[str, list[str]],
    dev_candidates: dict[str, list[Candidate]],
    dev_baseline: dict[str, Candidate],
    alpha: float,
    switch_margin: float,
    n_splits: int,
    learner: str,
    min_pair_delta: float,
    seed: int,
    pretrain_rows: list[Candidate] | None = None,
    pretrain_weight: float = 0.0,
) -> list[Candidate]:
    video_ids = np.array(list(dev_refs))
    split_count = min(n_splits, len(video_ids))
    if split_count < 2:
        raise ValueError("At least two development videos are required for grouped CV.")
    groups = video_ids.copy()
    selected: list[Candidate] = []
    for train_idx, heldout_idx in GroupKFold(n_splits=split_count).split(video_ids, groups=groups):
        train_ids = video_ids[train_idx].tolist()
        heldout_ids = video_ids[heldout_idx].tolist()
        model = train_model(
            all_rows(dev_candidates, train_ids),
            alpha=alpha,
            learner=learner,
            min_pair_delta=min_pair_delta,
            seed=seed,
            pretrain_rows=pretrain_rows,
            pretrain_weight=pretrain_weight,
        )
        heldout_refs = {video_id: dev_refs[video_id] for video_id in heldout_ids}
        heldout_candidates = {video_id: dev_candidates[video_id] for video_id in heldout_ids}
        heldout_baseline = {video_id: dev_baseline[video_id] for video_id in heldout_ids}
        selected.extend(
            select_with_model(
                heldout_refs,
                heldout_candidates,
                heldout_baseline,
                model,
                switch_margin=switch_margin,
            )
        )
    selected_by_id = {row.video_id: row for row in selected}
    return [selected_by_id[video_id] for video_id in dev_refs]


def score_rows_quiet(rows: list[Candidate]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        return score_rows(rows)


def coefficient_summary(model) -> list[dict[str, float | str]]:
    coefficient_name = "standardized_coefficient"
    if isinstance(model, ExtraTreesRanker):
        coefs = model.model.feature_importances_
        coefficient_name = "feature_importance"
    elif isinstance(model, PairwiseRanker):
        coefs = model.model.coef_.reshape(-1)
    else:
        coefs = model.model.coef_
    rows = [
        {"feature": name, coefficient_name: float(coef)}
        for name, coef in zip(FEATURE_NAMES, coefs)
    ]
    rows.sort(key=lambda item: abs(float(item[coefficient_name])), reverse=True)
    return rows


def write_markdown(output_dir: Path, summary: dict) -> None:
    learner = summary.get("learner", "ridge")
    learner_description = (
        "a pairwise logistic ranker over within-video candidate comparisons"
        if learner == "pairwise"
        else "an ExtraTrees regressor to predict sentence-level CIDEr"
        if learner == "extra_trees"
        else "a Ridge model to predict sentence-level CIDEr"
    )
    lines = [
        "# Frozen Learned Selector",
        "",
        "## Method",
        "",
        (
            f"The selector trains {learner_description} from "
            "candidate-only features on the frozen development splits. Hyperparameters "
            "are selected by video-grouped cross-validation; test references are used "
            "only for final scoring."
        ),
    ]
    if summary.get("msrvtt_beam_pretrain_enabled"):
        lines.extend(
            [
                "",
                (
                    f"MSR-VTT beam pretraining uses `{summary['msrvtt_pretrain_videos']}` "
                    f"videos and `{summary['msrvtt_pretrain_candidates']}` generated "
                    f"beam candidates with pretraining weight `{summary['pretrain_weight']}`. "
                    "Labels are sentence-level CIDEr scores for captions generated by the "
                    "tuned MSR-VTT captioner."
                ),
            ]
        )
    elif summary.get("msrvtt_pretrain_enabled"):
        lines.extend(
            [
                "",
                (
                    f"MSR-VTT pretraining uses `{summary['msrvtt_pretrain_videos']}` videos "
                    f"with pretraining weight `{summary['pretrain_weight']}`. Labels are "
                    "leave-one-out reference-consensus CIDEr scores."
                ),
            ]
        )
    lines.extend(
        [
        "",
        "## Test Results",
        "",
        "| Method | CIDEr | BLEU-4 | Source counts |",
        "| --- | ---: | ---: | --- |",
        ]
    )
    for key, label in [
        ("test_baseline_metrics", "clean_baseline"),
        ("test_metrics", "learned_selector"),
        ("test_oracle_metrics", "oracle"),
    ]:
        metrics = summary[key]
        source_key = {
            "test_baseline_metrics": "test_baseline_source_counts",
            "test_metrics": "test_source_counts",
            "test_oracle_metrics": "test_oracle_source_counts",
        }[key]
        lines.append(
            f"| {label} | {metrics['CIDEr']:.4f} | {metrics.get('Bleu_4', 0.0):.4f} | `{summary[source_key]}` |"
        )
    lines.extend(
        [
            "",
            "## Cross-Validation Choice",
            "",
            f"- Best alpha: `{summary['best_alpha']}`",
            f"- Best switch margin: `{summary['best_switch_margin']}`",
            f"- OOF dev CIDEr: `{summary['best_cv_result']['oof_metrics']['CIDEr']:.4f}`",
            f"- Test CIDEr gain over clean baseline: `{summary['test_cider_gain_vs_clean_baseline']:.4f}`",
            "",
            "## Notes",
            "",
            "- The oracle row is a candidate-pool diagnostic, not a deployable result.",
            "- The learned selector is frozen before test scoring, but normalized-reference results should still be treated as sensitivity analysis.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def load_msrvtt_pretrain_refs(
    captions_root: Path,
    splits: list[str],
    max_videos: int,
    seed: int,
) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for split in splits:
        path = captions_root / f"{split}_captions.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing MSR-VTT caption split: {path}")
        payload = load_json(path)
        for video_id, captions in payload.items():
            cleaned = [" ".join(str(caption).strip().split()) for caption in captions]
            cleaned = [caption for caption in cleaned if caption]
            if len(cleaned) >= 2:
                refs[f"msrvtt::{video_id}"] = cleaned
    if max_videos > 0 and len(refs) > max_videos:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(list(refs), size=max_videos, replace=False).tolist())
        refs = {video_id: refs[video_id] for video_id in selected}
    return refs


def build_msrvtt_pretrain_candidates(
    captions_root: Path,
    splits: list[str],
    max_videos: int,
    seed: int,
    idf: dict[str, float],
) -> tuple[dict[str, list[Candidate]], dict[str, Candidate]]:
    refs = load_msrvtt_pretrain_refs(captions_root, splits, max_videos, seed)
    if not refs:
        return {}, {}
    scorer = SplitCiderScorer(make_ground_truth(refs))
    candidates_by_video: dict[str, list[Candidate]] = {}
    baseline_by_video: dict[str, Candidate] = {}
    empty_profile = {"frame_count": 0, "group_support": {}, "token_counts": {}}

    for video_id, captions in refs.items():
        rows: list[Candidate] = []
        seen: set[str] = set()
        for rank, caption in enumerate(captions):
            caption = " ".join(caption.strip().split())
            key = caption.lower()
            if not caption or key in seen:
                continue
            seen.add(key)
            loo_refs = [other for idx, other in enumerate(captions) if idx != rank]
            if not loo_refs:
                loo_refs = captions
            rows.append(
                Candidate(
                    video_id=video_id,
                    caption=caption,
                    references=captions,
                    source="bart",
                    model_score=0.0,
                    model_rank=0,
                    sentence_cider=scorer.score(loo_refs, caption),
                )
            )
        if not rows:
            continue
        baseline = rows[0]
        baseline_by_video[video_id] = baseline
        for row in rows:
            compute_candidate_features(row, baseline, empty_profile, idf)
        candidates_by_video[video_id] = rows
    return candidates_by_video, baseline_by_video


def build_msrvtt_beam_pretrain_candidates(
    run_dir: Path,
    splits: list[str],
    idf: dict[str, float],
) -> tuple[dict[str, list[Candidate]], dict[str, Candidate], dict]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing MSR-VTT beam run directory: {run_dir}")

    merged_by_video: dict[str, dict[str, dict[str, float | str]]] = {}
    references_by_video: dict[str, list[str]] = {}
    source_files: list[str] = []
    decode_settings: list[dict] = []

    for split in splits:
        paths = sorted(run_dir.glob(f"beam_headroom_{split}_*.json"))
        if not paths:
            raise FileNotFoundError(
                f"No beam_headroom_{split}_*.json files found in MSR-VTT beam directory: {run_dir}"
            )
        for path in paths:
            payload = load_json(path)
            source_files.append(str(path))
            decode_settings.append(
                {
                    "file": str(path),
                    "split": payload.get("split", split),
                    "decode": payload.get("decode", {}),
                    "checkpoint": payload.get("checkpoint"),
                    "videos": len(payload.get("per_video", [])),
                }
            )
            for record in payload.get("per_video", []):
                raw_video_id = str(record.get("video_id", "")).strip()
                if not raw_video_id:
                    continue
                video_id = f"msrvtt_beam::{split}::{raw_video_id}"
                references = [
                    " ".join(str(reference).strip().split())
                    for reference in record.get("references", [])
                ]
                references = [reference for reference in references if reference]
                if not references:
                    continue
                references_by_video.setdefault(video_id, references)
                caption_map = merged_by_video.setdefault(video_id, {})
                for item in record.get("all_candidates", []):
                    caption = " ".join(str(item.get("caption", "")).strip().split())
                    if not caption:
                        continue
                    key = caption.lower()
                    candidate = {
                        "caption": caption,
                        "model_score": float(item.get("model_score", -99.0)),
                        "cider": float(item.get("cider", 0.0)),
                    }
                    previous = caption_map.get(key)
                    if previous is None:
                        caption_map[key] = candidate
                    elif float(candidate["model_score"]) > float(previous["model_score"]):
                        candidate["cider"] = max(float(candidate["cider"]), float(previous["cider"]))
                        caption_map[key] = candidate

    candidates_by_video: dict[str, list[Candidate]] = {}
    baseline_by_video: dict[str, Candidate] = {}
    empty_profile = {"frame_count": 0, "group_support": {}, "token_counts": {}}

    for video_id in sorted(merged_by_video):
        references = references_by_video[video_id]
        ranked_candidates = sorted(
            merged_by_video[video_id].values(),
            key=lambda item: float(item["model_score"]),
            reverse=True,
        )
        rows: list[Candidate] = []
        for rank, item in enumerate(ranked_candidates):
            rows.append(
                Candidate(
                    video_id=video_id,
                    caption=str(item["caption"]),
                    references=references,
                    source="bart",
                    model_score=float(item["model_score"]),
                    model_rank=rank,
                    sentence_cider=float(item["cider"]),
                )
            )
        if not rows:
            continue
        baseline = rows[0]
        baseline_by_video[video_id] = baseline
        for row in rows:
            compute_candidate_features(row, baseline, empty_profile, idf)
        candidates_by_video[video_id] = rows

    metadata = {
        "run_dir": str(run_dir),
        "splits": splits,
        "source_files": source_files,
        "decode_settings": decode_settings,
        "videos": len(candidates_by_video),
        "candidates": int(sum(len(rows) for rows in candidates_by_video.values())),
    }
    return candidates_by_video, baseline_by_video, metadata


def main() -> None:
    args = parse_args()
    np.random.default_rng(args.seed)
    workspace = Path(args.workspace_root).expanduser().resolve()
    captions_root = (
        Path(args.refs_root).expanduser().resolve()
        if args.refs_root.strip()
        else workspace / "processed_clean" / "captions"
    )
    adapted_run = Path(args.adapted_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = Path(args.proposals_json).expanduser().resolve()
    if not proposals_path.exists():
        proposals_path = Path(args.fallback_proposals_json).expanduser().resolve()
    proposals_payload = load_json(proposals_path)

    refs_by_split = {
        split: load_json(captions_root / args.refs_filename_template.format(split=split))
        for split in ["train", "val", "test"]
    }
    scorer_by_split = {
        split: SplitCiderScorer(make_ground_truth(refs))
        for split, refs in refs_by_split.items()
    }
    dev_splits = parse_csv_list(args.dev_splits, "--dev-splits")
    if any(split not in refs_by_split for split in dev_splits):
        raise ValueError(f"Unknown dev split in --dev-splits: {args.dev_splits}")

    memory_refs = {}
    for split in dev_splits:
        memory_refs.update(refs_by_split[split])
    idf = build_target_idf(memory_refs)

    candidates_by_split = {}
    baseline_by_split = {}
    for split in ["train", "val", "test"]:
        candidates, baseline, _ = prepare_split(
            split=split,
            refs=refs_by_split[split],
            run_dir=adapted_run,
            scorer=scorer_by_split[split],
            proposals_payload=proposals_payload,
            idf=idf,
            candidate_sources=args.candidate_sources,
            max_blip_candidates=args.max_blip_candidates,
        )
        candidates_by_split[split] = candidates
        baseline_by_split[split] = baseline

    dev_refs = {}
    dev_candidates: dict[str, list[Candidate]] = {}
    dev_baseline: dict[str, Candidate] = {}
    for split in dev_splits:
        dev_refs.update(refs_by_split[split])
        dev_candidates.update(candidates_by_split[split])
        dev_baseline.update(baseline_by_split[split])

    pretrain_candidates: dict[str, list[Candidate]] = {}
    pretrain_baseline: dict[str, Candidate] = {}
    pretrain_metadata: dict = {}
    pretrain_type = "none"
    msrvtt_beam_run_dir = args.msrvtt_beam_run_dir.strip()
    if msrvtt_beam_run_dir and args.pretrain_weight > 0.0:
        msrvtt_splits = parse_csv_list(args.msrvtt_beam_splits, "--msrvtt-beam-splits")
        pretrain_candidates, pretrain_baseline, pretrain_metadata = build_msrvtt_beam_pretrain_candidates(
            run_dir=Path(msrvtt_beam_run_dir).expanduser().resolve(),
            splits=msrvtt_splits,
            idf=idf,
        )
        pretrain_type = "msrvtt_generated_beams"
    elif args.msrvtt_pretrain_videos > 0 and args.pretrain_weight > 0.0:
        msrvtt_splits = parse_csv_list(args.msrvtt_pretrain_splits, "--msrvtt-pretrain-splits")
        pretrain_candidates, pretrain_baseline = build_msrvtt_pretrain_candidates(
            captions_root=Path(args.msrvtt_pretrain_captions_root).expanduser().resolve(),
            splits=msrvtt_splits,
            max_videos=args.msrvtt_pretrain_videos,
            seed=args.seed,
            idf=idf,
        )
        pretrain_metadata = {
            "captions_root": str(Path(args.msrvtt_pretrain_captions_root).expanduser().resolve()),
            "splits": msrvtt_splits,
            "videos": len(pretrain_candidates),
            "candidates": int(sum(len(rows) for rows in pretrain_candidates.values())),
        }
        pretrain_type = "msrvtt_reference_consensus"

    consensus_vectorizer = (
        fit_consensus_vectorizer(dev_candidates, pretrain_candidates)
        if pretrain_candidates
        else fit_consensus_vectorizer(dev_candidates)
    )
    for split in ["train", "val", "test"]:
        attach_consensus_features(candidates_by_split[split], baseline_by_split[split], consensus_vectorizer)
    if pretrain_candidates:
        attach_consensus_features(pretrain_candidates, pretrain_baseline, consensus_vectorizer)

    clip_feature_roots: list[Path] = []
    if args.enable_clip_video_text_features:
        clip_feature_roots = (
            parse_path_list(args.clip_feature_roots)
            if args.clip_feature_roots.strip()
            else default_clip_feature_roots(workspace)
        )
        clip_scorer = ClipTextVideoScorer(
            feature_roots=clip_feature_roots,
            model_name=args.clip_text_model_name,
            device_arg=args.clip_feature_device,
            batch_size=args.clip_text_batch_size,
        )
        for split in ["train", "val", "test"]:
            attach_clip_video_text_features(
                candidates_by_split[split],
                baseline_by_split[split],
                clip_scorer,
                label=f"dattalion_{split}",
                progress_every=args.clip_progress_every,
            )
        if pretrain_candidates:
            attach_clip_video_text_features(
                pretrain_candidates,
                pretrain_baseline,
                clip_scorer,
                label="msrvtt_pretrain",
                progress_every=args.clip_progress_every,
            )
    pretrain_rows = all_rows(pretrain_candidates) if pretrain_candidates else []

    alphas = parse_float_list(args.alphas, "--alphas")
    switch_margins = parse_float_list(args.switch_margins, "--switch-margins")
    cv_results = []
    for alpha in alphas:
        for margin in switch_margins:
            selected_oof = grouped_cv_predictions(
                dev_refs=dev_refs,
                dev_candidates=dev_candidates,
                dev_baseline=dev_baseline,
                alpha=alpha,
                switch_margin=margin,
                n_splits=args.n_splits,
                learner=args.learner,
                min_pair_delta=args.min_pair_delta,
                seed=args.seed,
                pretrain_rows=pretrain_rows,
                pretrain_weight=args.pretrain_weight,
            )
            oof_metrics = score_rows_quiet(selected_oof)
            cv_results.append(
                {
                    "alpha": alpha,
                    "switch_margin": margin,
                    "oof_metrics": oof_metrics,
                    "oof_mean_sentence_cider": float(np.mean([row.sentence_cider for row in selected_oof])),
                    "oof_switches": len(switch_rows(selected_oof, dev_baseline)),
                    "oof_source_counts": source_counts(selected_oof),
                }
            )
    cv_results.sort(
        key=lambda row: (
            row["oof_metrics"]["CIDEr"],
            row["oof_mean_sentence_cider"],
            -row["oof_switches"],
        ),
        reverse=True,
    )
    best = cv_results[0]
    best_alpha = float(best["alpha"])
    best_margin = float(best["switch_margin"])
    final_model = train_model(
        all_rows(dev_candidates),
        alpha=best_alpha,
        learner=args.learner,
        min_pair_delta=args.min_pair_delta,
        seed=args.seed,
        pretrain_rows=pretrain_rows,
        pretrain_weight=args.pretrain_weight,
    )

    baseline_dev_rows = select_baselines(dev_refs, dev_baseline)
    baseline_test_rows = select_baselines(refs_by_split["test"], baseline_by_split["test"])
    selected_dev = select_with_model(dev_refs, dev_candidates, dev_baseline, final_model, best_margin)
    selected_test = select_with_model(
        refs_by_split["test"],
        candidates_by_split["test"],
        baseline_by_split["test"],
        final_model,
        best_margin,
    )
    oracle_dev = candidate_pool_oracle(dev_refs, dev_candidates)
    oracle_test = candidate_pool_oracle(refs_by_split["test"], candidates_by_split["test"])

    baseline_dev_metrics = score_rows_quiet(baseline_dev_rows)
    baseline_test_metrics = score_rows_quiet(baseline_test_rows)
    dev_metrics = score_rows_quiet(selected_dev)
    test_metrics = score_rows_quiet(selected_test)
    oracle_dev_metrics = score_rows_quiet(oracle_dev)
    oracle_test_metrics = score_rows_quiet(oracle_test)

    save_json(output_dir / "cv_results_top50.json", cv_results[:50])
    save_json(output_dir / "dev_predictions.json", serialise(selected_dev, dev_baseline))
    save_json(output_dir / "dev_oof_predictions.json", serialise(
        grouped_cv_predictions(
            dev_refs,
            dev_candidates,
            dev_baseline,
            best_alpha,
            best_margin,
            args.n_splits,
            args.learner,
            args.min_pair_delta,
            args.seed,
            pretrain_rows,
            args.pretrain_weight,
        ),
        dev_baseline,
    ))
    save_json(output_dir / "test_predictions.json", serialise(selected_test, baseline_by_split["test"]))
    save_json(output_dir / "test_baseline_predictions.json", serialise(baseline_test_rows, baseline_by_split["test"]))
    save_json(output_dir / "test_oracle_predictions.json", serialise(oracle_test, baseline_by_split["test"]))

    summary = {
        "method": "video-grouped frozen learned candidate selector",
        "method_short_name": "FLCS",
        "learner": args.learner,
        "min_pair_delta": args.min_pair_delta,
        "msrvtt_pretrain_enabled": bool(pretrain_rows),
        "msrvtt_pretrain_type": pretrain_type,
        "msrvtt_pretrain_captions_root": str(Path(args.msrvtt_pretrain_captions_root).expanduser().resolve()),
        "msrvtt_pretrain_splits": args.msrvtt_pretrain_splits,
        "msrvtt_pretrain_videos": len(pretrain_candidates),
        "msrvtt_pretrain_candidates": len(pretrain_rows),
        "msrvtt_beam_pretrain_enabled": pretrain_type == "msrvtt_generated_beams",
        "msrvtt_beam_run_dir": str(Path(msrvtt_beam_run_dir).expanduser().resolve()) if msrvtt_beam_run_dir else None,
        "msrvtt_beam_splits": args.msrvtt_beam_splits,
        "msrvtt_pretrain_metadata": pretrain_metadata,
        "pretrain_weight": args.pretrain_weight if pretrain_rows else 0.0,
        "proposals_json": str(proposals_path),
        "proposal_method": proposals_payload.get("method"),
        "workspace_root": str(workspace),
        "refs_root": str(captions_root),
        "refs_filename_template": args.refs_filename_template,
        "clip_video_text_features_enabled": bool(args.enable_clip_video_text_features),
        "clip_text_model_name": args.clip_text_model_name if args.enable_clip_video_text_features else None,
        "clip_feature_roots": [str(path) for path in clip_feature_roots],
        "candidate_sources": args.candidate_sources,
        "max_blip_candidates": args.max_blip_candidates,
        "dev_splits": dev_splits,
        "feature_names": FEATURE_NAMES,
        "best_alpha": best_alpha,
        "best_switch_margin": best_margin,
        "best_cv_result": best,
        "cv_results_top10": cv_results[:10],
        "coefficients_by_abs_value": coefficient_summary(final_model),
        "dev_baseline_metrics": baseline_dev_metrics,
        "dev_metrics": dev_metrics,
        "dev_oracle_metrics": oracle_dev_metrics,
        "dev_cider_gain_vs_clean_baseline": dev_metrics["CIDEr"] - baseline_dev_metrics["CIDEr"],
        "dev_switches": switch_rows(selected_dev, dev_baseline),
        "dev_source_counts": source_counts(selected_dev),
        "test_baseline_metrics": baseline_test_metrics,
        "test_metrics": test_metrics,
        "test_oracle_metrics": oracle_test_metrics,
        "test_cider_gain_vs_clean_baseline": test_metrics["CIDEr"] - baseline_test_metrics["CIDEr"],
        "test_cider_gain_vs_current_reported_baseline": test_metrics["CIDEr"] - args.current_reported_cider,
        "test_switches": switch_rows(selected_test, baseline_by_split["test"]),
        "test_source_counts": source_counts(selected_test),
        "test_baseline_source_counts": source_counts(baseline_test_rows),
        "test_oracle_source_counts": source_counts(oracle_test),
        "test_num_candidates": int(sum(len(rows) for rows in candidates_by_split["test"].values())),
        "dev_num_candidates": int(sum(len(rows) for rows in dev_candidates.values())),
        "interpretation_note": (
            "Model hyperparameters are selected by video-grouped cross-validation over the "
            "development splits, then the model is refit on all development videos before "
            "held-out test scoring. Sentence-level CIDEr labels are never read for choosing "
            "test captions; they are stored only for diagnostics and oracle analysis."
        ),
    }
    save_json(output_dir / "summary.json", summary)
    write_markdown(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
