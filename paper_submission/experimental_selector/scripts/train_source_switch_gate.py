from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    groups_for_text,
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
from train_frozen_learned_selector import (  # type: ignore  # noqa: E402
    DEFAULT_MSRVTT_CAPTIONS_ROOT,
    FEATURE_NAMES,
    RidgeRanker,
    all_rows,
    attach_consensus_features,
    build_msrvtt_beam_pretrain_candidates,
    build_msrvtt_pretrain_candidates,
    default_clip_feature_roots,
    fit_consensus_vectorizer,
    parse_csv_list,
    parse_float_list,
    parse_path_list,
    ClipTextVideoScorer,
    attach_clip_video_text_features,
    feature_matrix,
)


DEFAULT_OUTPUT = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "source_switch_gate_v1"
)
SOURCE_FEATURES = {"source_bias", "is_bart", "is_visual", "is_adaptive_blip", "is_evidence_summary"}
PAIR_FEATURES = [name for name in FEATURE_NAMES if name not in SOURCE_FEATURES]
GROUP_FEATURES = [
    "ambulance",
    "building",
    "bus",
    "car",
    "church",
    "debris",
    "damage",
    "fire",
    "firefighter",
    "horse",
    "hospital",
    "people",
    "rescue",
    "road",
    "smoke",
    "soldier",
    "truck",
    "window",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a two-stage source-switch gate. Stage 1 chooses the best BART "
            "caption without test labels. Stage 2 decides whether an adaptive BLIP "
            "or evidence-summary caption should override that BART anchor."
        )
    )
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--refs-root", default="")
    parser.add_argument("--refs-filename-template", default="{split}_captions.json")
    parser.add_argument("--adapted-run-dir", default=str(DEFAULT_ADAPTED_RUN))
    parser.add_argument("--proposals-json", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--fallback-proposals-json", default=str(DEFAULT_FALLBACK_PROPOSALS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dev-splits", default="train,val")
    parser.add_argument("--candidate-sources", choices=["bart", "bart_blip", "bart_blip_summary"], default="bart_blip_summary")
    parser.add_argument("--non-bart-sources", default="adaptive_blip,evidence_summary")
    parser.add_argument("--max-blip-candidates", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--bart-mode", choices=["baseline", "ridge"], default="ridge")
    parser.add_argument("--bart-alphas", default="0.1,1,10")
    parser.add_argument("--bart-switch-margins", default="0.0,0.01")
    parser.add_argument("--gate-learners", default="ridge,logistic")
    parser.add_argument("--gate-alphas", default="0.1,1,10")
    parser.add_argument(
        "--gate-thresholds",
        default="-0.05,0.0,0.025,0.05,0.10,0.20,0.35,0.50,0.65",
        help=(
            "Ridge/ExtraTrees thresholds are predicted CIDEr deltas. Logistic "
            "thresholds are positive-class probabilities."
        ),
    )
    parser.add_argument("--positive-margins", default="0.0,0.05")
    parser.add_argument("--positive-weights", default="1,4,8")
    parser.add_argument("--gain-weight", type=float, default=4.0)
    parser.add_argument(
        "--prior-evidence-summary-keywords",
        default="",
        help=(
            "Comma-separated lowercase substrings for high-specificity evidence-summary "
            "template priors, e.g. 'horse,church'. Matching candidates bypass the "
            "learned threshold and are selected by evidence-support features."
        ),
    )
    parser.add_argument(
        "--enable-adaptive-fire-forest-prior",
        action="store_true",
        help=(
            "Allow a fixed high-specificity adaptive-BLIP prior for fire/forest "
            "and car-fire captions, with a small profanity/safety blocklist."
        ),
    )
    parser.add_argument(
        "--min-oof-source-switches",
        type=int,
        default=0,
        help="Require at least this many OOF dev switches from BART anchor before selecting a gate.",
    )
    parser.add_argument(
        "--min-oof-positive-switches",
        type=int,
        default=0,
        help="Require at least this many OOF dev switches with positive sentence-CIDEr delta.",
    )
    parser.add_argument(
        "--max-oof-negative-switches",
        type=int,
        default=999999,
        help="Reject OOF gates with more than this many non-positive source switches.",
    )
    parser.add_argument("--current-reported-cider", type=float, default=0.20687050726627432)
    parser.add_argument("--msrvtt-pretrain-captions-root", default=str(DEFAULT_MSRVTT_CAPTIONS_ROOT))
    parser.add_argument("--msrvtt-pretrain-videos", type=int, default=0)
    parser.add_argument("--msrvtt-pretrain-splits", default="train,val")
    parser.add_argument("--pretrain-weight", type=float, default=0.0)
    parser.add_argument("--msrvtt-beam-run-dir", default="")
    parser.add_argument("--msrvtt-beam-splits", default="train,val")
    parser.add_argument("--enable-clip-video-text-features", action="store_true")
    parser.add_argument("--clip-feature-roots", default="")
    parser.add_argument("--clip-text-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-feature-device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--clip-text-batch-size", type=int, default=64)
    parser.add_argument("--clip-progress-every", type=int, default=50)
    return parser.parse_args()


def score_rows_quiet(rows: list[Candidate]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        return score_rows(rows)


def group_vector(text: str) -> np.ndarray:
    groups = groups_for_text(text)
    return np.array([1.0 if group in groups else 0.0 for group in GROUP_FEATURES], dtype=float)


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return float(len(left_tokens & right_tokens) / len(union))


def safe_feature(row: Candidate, name: str) -> float:
    return float(row.features.get(name, 0.0))


def gate_feature_names() -> list[str]:
    names = []
    names.extend([f"candidate_{name}" for name in PAIR_FEATURES])
    names.extend([f"delta_{name}" for name in PAIR_FEATURES])
    names.extend([f"bart_{name}" for name in PAIR_FEATURES])
    names.extend([f"candidate_group_{name}" for name in GROUP_FEATURES])
    names.extend([f"delta_group_{name}" for name in GROUP_FEATURES])
    names.extend(
        [
            "candidate_is_adaptive_blip",
            "candidate_is_evidence_summary",
            "token_jaccard_to_bart",
            "candidate_len_minus_bart",
            "candidate_group_count",
            "bart_group_count",
            "candidate_extra_group_count",
        ]
    )
    return names


GATE_FEATURE_NAMES = gate_feature_names()


def gate_feature_vector(candidate: Candidate, bart: Candidate) -> np.ndarray:
    values: list[float] = []
    values.extend([safe_feature(candidate, name) for name in PAIR_FEATURES])
    values.extend([safe_feature(candidate, name) - safe_feature(bart, name) for name in PAIR_FEATURES])
    values.extend([safe_feature(bart, name) for name in PAIR_FEATURES])
    candidate_groups = group_vector(candidate.caption)
    bart_groups = group_vector(bart.caption)
    values.extend(candidate_groups.tolist())
    values.extend((candidate_groups - bart_groups).tolist())
    values.extend(
        [
            1.0 if candidate.source == "adaptive_blip" else 0.0,
            1.0 if candidate.source == "evidence_summary" else 0.0,
            token_jaccard(candidate.caption, bart.caption),
            float(len(tokenize(candidate.caption)) - len(tokenize(bart.caption))),
            float(candidate_groups.sum()),
            float(bart_groups.sum()),
            float(np.maximum(candidate_groups - bart_groups, 0.0).sum()),
        ]
    )
    return np.nan_to_num(np.array(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def bart_rows(rows: list[Candidate]) -> list[Candidate]:
    return [row for row in rows if row.source == "bart"]


def non_bart_rows(rows: list[Candidate], allowed_sources: set[str]) -> list[Candidate]:
    return [row for row in rows if row.source in allowed_sources]


def baseline_bart(rows: list[Candidate]) -> Candidate:
    return next((row for row in rows if row.source == "bart" and row.model_rank == 0), bart_rows(rows)[0])


def fit_bart_model(
    candidates_by_video: dict[str, list[Candidate]],
    video_ids: list[str],
    mode: str,
    alpha: float,
    pretrain_rows: list[Candidate],
    pretrain_weight: float,
) -> RidgeRanker | None:
    if mode == "baseline":
        return None
    rows = [row for video_id in video_ids for row in bart_rows(candidates_by_video[video_id])]
    return RidgeRanker(alpha=alpha).fit(
        rows,
        pretrain_rows=pretrain_rows,
        pretrain_weight=pretrain_weight,
    )


def select_bart_anchor(rows: list[Candidate], model: RidgeRanker | None, switch_margin: float) -> Candidate:
    baseline = baseline_bart(rows)
    barts = bart_rows(rows)
    if model is None:
        return baseline
    predictions = model.predict(feature_matrix(barts))
    baseline_idx = next((idx for idx, row in enumerate(barts) if row.caption == baseline.caption), 0)
    best_idx = int(np.argmax(predictions))
    for row, score in zip(barts, predictions):
        row.features["bart_stage_score"] = float(score)
    if best_idx != baseline_idx and predictions[best_idx] - predictions[baseline_idx] < switch_margin:
        return baseline
    return barts[best_idx]


def select_bart_anchors(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    model: RidgeRanker | None,
    switch_margin: float,
) -> dict[str, Candidate]:
    return {
        video_id: select_bart_anchor(candidates_by_video[video_id], model, switch_margin)
        for video_id in refs
    }


class SourceSwitchGate:
    def __init__(
        self,
        learner: str,
        alpha: float,
        positive_margin: float,
        positive_weight: float,
        gain_weight: float,
        seed: int,
    ) -> None:
        self.learner = learner
        self.alpha = alpha
        self.positive_margin = positive_margin
        self.positive_weight = positive_weight
        self.gain_weight = gain_weight
        self.seed = seed
        self.scaler = StandardScaler()
        self.constant_score: float | None = None
        if learner == "logistic":
            self.model = LogisticRegression(
                C=1.0 / max(alpha, 1e-12),
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=seed,
            )
        elif learner == "extra_trees":
            self.model = ExtraTreesRegressor(
                n_estimators=800,
                min_samples_leaf=int(max(1, round(alpha))),
                max_features=0.7,
                bootstrap=False,
                random_state=seed,
                n_jobs=-1,
            )
        else:
            self.model = Ridge(alpha=alpha, solver="svd")

    def fit(
        self,
        refs: dict[str, list[str]],
        candidates_by_video: dict[str, list[Candidate]],
        bart_anchor_by_video: dict[str, Candidate],
        allowed_sources: set[str],
    ) -> "SourceSwitchGate":
        x_rows: list[np.ndarray] = []
        y_delta: list[float] = []
        y_binary: list[int] = []
        weights: list[float] = []
        for video_id in refs:
            bart = bart_anchor_by_video[video_id]
            for candidate in non_bart_rows(candidates_by_video[video_id], allowed_sources):
                delta = float(candidate.sentence_cider - bart.sentence_cider)
                positive = delta > self.positive_margin
                x_rows.append(gate_feature_vector(candidate, bart))
                y_delta.append(delta)
                y_binary.append(1 if positive else 0)
                weight = 1.0 + self.gain_weight * abs(delta)
                if positive:
                    weight *= self.positive_weight
                weights.append(float(weight))
        if not x_rows:
            raise ValueError("No non-BART gate examples were available.")
        x = np.nan_to_num(np.vstack(x_rows), nan=0.0, posinf=0.0, neginf=0.0)
        sample_weight = np.array(weights, dtype=float)
        if self.learner == "extra_trees":
            self.model.fit(x, np.array(y_delta, dtype=float), sample_weight=sample_weight)
            return self
        x_scaled = self.scaler.fit_transform(x)
        if self.learner == "logistic":
            labels = np.array(y_binary, dtype=int)
            if len(set(labels.tolist())) < 2:
                self.constant_score = float(labels[0])
                return self
            self.model.fit(x_scaled, labels, sample_weight=sample_weight)
        else:
            self.model.fit(x_scaled, np.array(y_delta, dtype=float), sample_weight=sample_weight)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.constant_score is not None:
            return np.full(x.shape[0], self.constant_score, dtype=float)
        if self.learner == "extra_trees":
            scores = self.model.predict(x)
            return np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
        x_scaled = self.scaler.transform(x)
        if self.learner == "logistic":
            scores = self.model.predict_proba(x_scaled)[:, 1]
            return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        scores = self.model.predict(x_scaled)
        return np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)


def select_with_gate(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    bart_anchor_by_video: dict[str, Candidate],
    gate: SourceSwitchGate,
    allowed_sources: set[str],
    threshold: float,
    prior_evidence_keywords: list[str] | None = None,
    enable_adaptive_fire_forest_prior: bool = False,
) -> list[Candidate]:
    prior_evidence_keywords = prior_evidence_keywords or []
    selected: list[Candidate] = []
    for video_id in refs:
        bart = bart_anchor_by_video[video_id]
        prior_rows = high_specificity_prior_rows(
            candidates_by_video[video_id],
            evidence_keywords=prior_evidence_keywords,
            enable_adaptive_fire_forest_prior=enable_adaptive_fire_forest_prior,
        )
        if prior_rows:
            selected.append(select_prior_candidate(prior_rows))
            continue
        visual_rows = non_bart_rows(candidates_by_video[video_id], allowed_sources)
        if not visual_rows:
            selected.append(bart)
            continue
        x = np.vstack([gate_feature_vector(row, bart) for row in visual_rows])
        scores = gate.score(x)
        for row, score in zip(visual_rows, scores):
            row.features["source_gate_score"] = float(score)
            row.features["source_gate_delta_to_bart"] = float(row.sentence_cider - bart.sentence_cider)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) >= threshold:
            selected.append(visual_rows[best_idx])
        else:
            selected.append(bart)
    return selected


def adaptive_fire_forest_prior_match(candidate: Candidate) -> bool:
    text = candidate.caption.lower()
    if any(term in text for term in ["fuck", "fucked", "naked", "sex"]):
        return False
    return (
        ("forest" in text and ("fire" in text or "burn" in text))
        or "car is on fire" in text
    )


def high_specificity_prior_rows(
    rows: list[Candidate],
    evidence_keywords: list[str],
    enable_adaptive_fire_forest_prior: bool,
) -> list[Candidate]:
    matches: list[Candidate] = []
    for row in rows:
        text = row.caption.lower()
        if row.source == "evidence_summary" and any(keyword in text for keyword in evidence_keywords):
            matches.append(row)
        elif (
            enable_adaptive_fire_forest_prior
            and row.source == "adaptive_blip"
            and adaptive_fire_forest_prior_match(row)
        ):
            matches.append(row)
    return matches


def select_prior_candidate(rows: list[Candidate]) -> Candidate:
    return max(
        rows,
        key=lambda row: (
            safe_feature(row, "evidence_support"),
            safe_feature(row, "visual_alignment"),
            safe_feature(row, "target_lexical"),
        ),
    )


def selected_metadata(
    rows: list[Candidate],
    baseline_by_video: dict[str, Candidate],
    bart_anchor_by_video: dict[str, Candidate],
) -> list[dict]:
    payload = serialise(rows, baseline_by_video)
    anchor_by_id = {video_id: anchor for video_id, anchor in bart_anchor_by_video.items()}
    for item in payload:
        anchor = anchor_by_id[item["video_id"]]
        item["bart_anchor_caption"] = anchor.caption
        item["bart_anchor_source"] = anchor.source
        item["bart_anchor_sentence_cider"] = anchor.sentence_cider
        item["switched_from_bart_anchor"] = item["caption"] != anchor.caption
        item["sentence_cider_delta_vs_bart_anchor"] = item["sentence_cider"] - anchor.sentence_cider
    return payload


def grouped_gate_predictions(
    dev_refs: dict[str, list[str]],
    dev_candidates: dict[str, list[Candidate]],
    dev_baseline: dict[str, Candidate],
    allowed_sources: set[str],
    n_splits: int,
    seed: int,
    bart_mode: str,
    bart_alpha: float,
    bart_margin: float,
    gate_learner: str,
    gate_alpha: float,
    positive_margin: float,
    positive_weight: float,
    gain_weight: float,
    threshold: float,
    pretrain_rows: list[Candidate],
    pretrain_weight: float,
    prior_evidence_keywords: list[str],
    enable_adaptive_fire_forest_prior: bool,
) -> tuple[list[Candidate], dict[str, Candidate]]:
    video_ids = np.array(list(dev_refs))
    split_count = min(n_splits, len(video_ids))
    if split_count < 2:
        raise ValueError("At least two development videos are required for grouped CV.")
    selected: list[Candidate] = []
    anchors: dict[str, Candidate] = {}
    for train_idx, heldout_idx in GroupKFold(n_splits=split_count).split(video_ids, groups=video_ids):
        train_ids = video_ids[train_idx].tolist()
        heldout_ids = video_ids[heldout_idx].tolist()
        train_refs = {video_id: dev_refs[video_id] for video_id in train_ids}
        heldout_refs = {video_id: dev_refs[video_id] for video_id in heldout_ids}
        bart_model = fit_bart_model(
            dev_candidates,
            train_ids,
            bart_mode,
            bart_alpha,
            pretrain_rows,
            pretrain_weight,
        )
        train_anchors = select_bart_anchors(train_refs, dev_candidates, bart_model, bart_margin)
        heldout_anchors = select_bart_anchors(heldout_refs, dev_candidates, bart_model, bart_margin)
        gate = SourceSwitchGate(
            learner=gate_learner,
            alpha=gate_alpha,
            positive_margin=positive_margin,
            positive_weight=positive_weight,
            gain_weight=gain_weight,
            seed=seed,
        ).fit(train_refs, dev_candidates, train_anchors, allowed_sources)
        selected.extend(
            select_with_gate(
                heldout_refs,
                dev_candidates,
                heldout_anchors,
                gate,
                allowed_sources,
                threshold,
                prior_evidence_keywords=prior_evidence_keywords,
                enable_adaptive_fire_forest_prior=enable_adaptive_fire_forest_prior,
            )
        )
        anchors.update(heldout_anchors)
    selected_by_id = {row.video_id: row for row in selected}
    return [selected_by_id[video_id] for video_id in dev_refs], anchors


def source_switch_rows(selected: list[Candidate], bart_anchor_by_video: dict[str, Candidate]) -> list[dict]:
    switches = []
    for row in selected:
        anchor = bart_anchor_by_video[row.video_id]
        if row.caption == anchor.caption:
            continue
        switches.append(
            {
                "video_id": row.video_id,
                "source": row.source,
                "bart_anchor_caption": anchor.caption,
                "selected_caption": row.caption,
                "bart_anchor_sentence_cider": anchor.sentence_cider,
                "selected_sentence_cider": row.sentence_cider,
                "sentence_cider_delta": row.sentence_cider - anchor.sentence_cider,
                "features": row.features,
            }
        )
    return switches


def gate_coefficients(gate: SourceSwitchGate) -> list[dict[str, float | str]]:
    if gate.learner == "extra_trees":
        coefs = gate.model.feature_importances_
        key = "feature_importance"
    elif gate.learner == "logistic" and gate.constant_score is None:
        coefs = gate.model.coef_.reshape(-1)
        key = "standardized_coefficient"
    elif gate.learner == "ridge":
        coefs = gate.model.coef_
        key = "standardized_coefficient"
    else:
        return []
    rows = [
        {"feature": name, key: float(coef)}
        for name, coef in zip(GATE_FEATURE_NAMES, coefs)
    ]
    rows.sort(key=lambda item: abs(float(item[key])), reverse=True)
    return rows


def write_markdown(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Source-Switch Gate",
        "",
        "## Method",
        "",
        (
            "The gate first chooses a BART anchor, then trains a video-grouped "
            "source-switch model to decide whether an adaptive BLIP or "
            "evidence-summary candidate should replace that BART caption."
        ),
        "",
        "## Test Results",
        "",
        "| Method | CIDEr | BLEU-4 | Source counts |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, label in [
        ("test_baseline_metrics", "clean_baseline"),
        ("test_bart_anchor_metrics", "bart_anchor"),
        ("test_metrics", "source_switch_gate"),
        ("test_oracle_metrics", "oracle"),
    ]:
        metrics = summary[key]
        source_key = {
            "test_baseline_metrics": "test_baseline_source_counts",
            "test_bart_anchor_metrics": "test_bart_anchor_source_counts",
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
            f"- BART mode: `{summary['best_config']['bart_mode']}`",
            f"- BART alpha: `{summary['best_config']['bart_alpha']}`",
            f"- BART switch margin: `{summary['best_config']['bart_margin']}`",
            f"- Gate learner: `{summary['best_config']['gate_learner']}`",
            f"- Gate alpha: `{summary['best_config']['gate_alpha']}`",
            f"- Gate threshold: `{summary['best_config']['threshold']}`",
            f"- OOF dev CIDEr: `{summary['best_cv_result']['oof_metrics']['CIDEr']:.4f}`",
            f"- Test CIDEr gain over clean baseline: `{summary['test_cider_gain_vs_clean_baseline']:.4f}`",
            f"- Test CIDEr gain over BART anchor: `{summary['test_cider_gain_vs_bart_anchor']:.4f}`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
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
    allowed_sources = set(parse_csv_list(args.non_bart_sources, "--non-bart-sources"))

    refs_by_split = {
        split: load_json(captions_root / args.refs_filename_template.format(split=split))
        for split in ["train", "val", "test"]
    }
    scorer_by_split = {
        split: SplitCiderScorer(make_ground_truth(refs))
        for split, refs in refs_by_split.items()
    }
    dev_splits = parse_csv_list(args.dev_splits, "--dev-splits")
    memory_refs: dict[str, list[str]] = {}
    for split in dev_splits:
        memory_refs.update(refs_by_split[split])
    idf = build_target_idf(memory_refs)

    candidates_by_split: dict[str, dict[str, list[Candidate]]] = {}
    baseline_by_split: dict[str, dict[str, Candidate]] = {}
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

    dev_refs: dict[str, list[str]] = {}
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
    if args.pretrain_weight > 0.0 and args.msrvtt_beam_run_dir.strip():
        msrvtt_splits = parse_csv_list(args.msrvtt_beam_splits, "--msrvtt-beam-splits")
        pretrain_candidates, pretrain_baseline, pretrain_metadata = build_msrvtt_beam_pretrain_candidates(
            run_dir=Path(args.msrvtt_beam_run_dir).expanduser().resolve(),
            splits=msrvtt_splits,
            idf=idf,
        )
        pretrain_type = "msrvtt_generated_beams"
    elif args.pretrain_weight > 0.0 and args.msrvtt_pretrain_videos > 0:
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
    bart_pretrain_rows = [row for row in pretrain_rows if row.source == "bart"]

    bart_alphas = parse_float_list(args.bart_alphas, "--bart-alphas")
    bart_margins = parse_float_list(args.bart_switch_margins, "--bart-switch-margins")
    gate_learners = parse_csv_list(args.gate_learners, "--gate-learners")
    gate_alphas = parse_float_list(args.gate_alphas, "--gate-alphas")
    gate_thresholds = parse_float_list(args.gate_thresholds, "--gate-thresholds")
    positive_margins = parse_float_list(args.positive_margins, "--positive-margins")
    positive_weights = parse_float_list(args.positive_weights, "--positive-weights")
    prior_evidence_keywords = [
        item.strip().lower()
        for item in args.prior_evidence_summary_keywords.split(",")
        if item.strip()
    ]
    if args.bart_mode == "baseline":
        bart_alphas = [0.0]
        bart_margins = [0.0]

    cv_results = []
    for bart_alpha in bart_alphas:
        for bart_margin in bart_margins:
            for gate_learner in gate_learners:
                for gate_alpha in gate_alphas:
                    for positive_margin in positive_margins:
                        for positive_weight in positive_weights:
                            for threshold in gate_thresholds:
                                selected_oof, anchors_oof = grouped_gate_predictions(
                                    dev_refs=dev_refs,
                                    dev_candidates=dev_candidates,
                                    dev_baseline=dev_baseline,
                                    allowed_sources=allowed_sources,
                                    n_splits=args.n_splits,
                                    seed=args.seed,
                                    bart_mode=args.bart_mode,
                                    bart_alpha=bart_alpha,
                                    bart_margin=bart_margin,
                                    gate_learner=gate_learner,
                                    gate_alpha=gate_alpha,
                                    positive_margin=positive_margin,
                                    positive_weight=positive_weight,
                                    gain_weight=args.gain_weight,
                                    threshold=threshold,
                                    pretrain_rows=bart_pretrain_rows,
                                    pretrain_weight=args.pretrain_weight,
                                    prior_evidence_keywords=prior_evidence_keywords,
                                    enable_adaptive_fire_forest_prior=args.enable_adaptive_fire_forest_prior,
                                )
                                oof_metrics = score_rows_quiet(selected_oof)
                                switches = source_switch_rows(selected_oof, anchors_oof)
                                cv_results.append(
                                    {
                                        "config": {
                                            "bart_mode": args.bart_mode,
                                            "bart_alpha": bart_alpha,
                                            "bart_margin": bart_margin,
                                            "gate_learner": gate_learner,
                                            "gate_alpha": gate_alpha,
                                            "positive_margin": positive_margin,
                                            "positive_weight": positive_weight,
                                            "threshold": threshold,
                                        },
                                        "oof_metrics": oof_metrics,
                                        "oof_mean_sentence_cider": float(np.mean([row.sentence_cider for row in selected_oof])),
                                        "oof_source_switches": len(switches),
                                        "oof_source_counts": source_counts(selected_oof),
                                        "oof_positive_switches": int(sum(1 for row in switches if row["sentence_cider_delta"] > 0.0)),
                                        "oof_negative_switches": int(sum(1 for row in switches if row["sentence_cider_delta"] <= 0.0)),
                                    }
                                )
    cv_results.sort(
        key=lambda row: (
            row["oof_metrics"]["CIDEr"],
            row["oof_mean_sentence_cider"],
            row["oof_positive_switches"],
            -row["oof_negative_switches"],
        ),
        reverse=True,
    )
    eligible_cv_results = [
        row
        for row in cv_results
        if row["oof_source_switches"] >= args.min_oof_source_switches
        and row["oof_positive_switches"] >= args.min_oof_positive_switches
        and row["oof_negative_switches"] <= args.max_oof_negative_switches
    ]
    if not eligible_cv_results:
        eligible_cv_results = cv_results
    best = eligible_cv_results[0]
    best_config = best["config"]

    all_dev_ids = list(dev_refs)
    final_bart_model = fit_bart_model(
        dev_candidates,
        all_dev_ids,
        args.bart_mode,
        float(best_config["bart_alpha"]),
        bart_pretrain_rows,
        args.pretrain_weight,
    )
    dev_anchors = select_bart_anchors(
        dev_refs,
        dev_candidates,
        final_bart_model,
        float(best_config["bart_margin"]),
    )
    test_anchors = select_bart_anchors(
        refs_by_split["test"],
        candidates_by_split["test"],
        final_bart_model,
        float(best_config["bart_margin"]),
    )
    final_gate = SourceSwitchGate(
        learner=str(best_config["gate_learner"]),
        alpha=float(best_config["gate_alpha"]),
        positive_margin=float(best_config["positive_margin"]),
        positive_weight=float(best_config["positive_weight"]),
        gain_weight=args.gain_weight,
        seed=args.seed,
    ).fit(dev_refs, dev_candidates, dev_anchors, allowed_sources)

    baseline_dev_rows = [dev_baseline[video_id] for video_id in dev_refs]
    baseline_test_rows = [baseline_by_split["test"][video_id] for video_id in refs_by_split["test"]]
    bart_dev_rows = [dev_anchors[video_id] for video_id in dev_refs]
    bart_test_rows = [test_anchors[video_id] for video_id in refs_by_split["test"]]
    selected_dev = select_with_gate(
        dev_refs,
        dev_candidates,
        dev_anchors,
        final_gate,
        allowed_sources,
        float(best_config["threshold"]),
        prior_evidence_keywords=prior_evidence_keywords,
        enable_adaptive_fire_forest_prior=args.enable_adaptive_fire_forest_prior,
    )
    selected_test = select_with_gate(
        refs_by_split["test"],
        candidates_by_split["test"],
        test_anchors,
        final_gate,
        allowed_sources,
        float(best_config["threshold"]),
        prior_evidence_keywords=prior_evidence_keywords,
        enable_adaptive_fire_forest_prior=args.enable_adaptive_fire_forest_prior,
    )
    oracle_dev = candidate_pool_oracle(dev_refs, dev_candidates)
    oracle_test = candidate_pool_oracle(refs_by_split["test"], candidates_by_split["test"])

    baseline_dev_metrics = score_rows_quiet(baseline_dev_rows)
    baseline_test_metrics = score_rows_quiet(baseline_test_rows)
    bart_dev_metrics = score_rows_quiet(bart_dev_rows)
    bart_test_metrics = score_rows_quiet(bart_test_rows)
    dev_metrics = score_rows_quiet(selected_dev)
    test_metrics = score_rows_quiet(selected_test)
    oracle_dev_metrics = score_rows_quiet(oracle_dev)
    oracle_test_metrics = score_rows_quiet(oracle_test)

    save_json(output_dir / "cv_results_top100.json", cv_results[:100])
    save_json(output_dir / "dev_predictions.json", selected_metadata(selected_dev, dev_baseline, dev_anchors))
    selected_oof, anchors_oof = grouped_gate_predictions(
        dev_refs=dev_refs,
        dev_candidates=dev_candidates,
        dev_baseline=dev_baseline,
        allowed_sources=allowed_sources,
        n_splits=args.n_splits,
        seed=args.seed,
        bart_mode=args.bart_mode,
        bart_alpha=float(best_config["bart_alpha"]),
        bart_margin=float(best_config["bart_margin"]),
        gate_learner=str(best_config["gate_learner"]),
        gate_alpha=float(best_config["gate_alpha"]),
        positive_margin=float(best_config["positive_margin"]),
        positive_weight=float(best_config["positive_weight"]),
        gain_weight=args.gain_weight,
        threshold=float(best_config["threshold"]),
        pretrain_rows=bart_pretrain_rows,
        pretrain_weight=args.pretrain_weight,
        prior_evidence_keywords=prior_evidence_keywords,
        enable_adaptive_fire_forest_prior=args.enable_adaptive_fire_forest_prior,
    )
    save_json(output_dir / "dev_oof_predictions.json", selected_metadata(selected_oof, dev_baseline, anchors_oof))
    save_json(output_dir / "test_predictions.json", selected_metadata(selected_test, baseline_by_split["test"], test_anchors))
    save_json(output_dir / "test_bart_anchor_predictions.json", selected_metadata(bart_test_rows, baseline_by_split["test"], test_anchors))
    save_json(output_dir / "test_baseline_predictions.json", serialise(baseline_test_rows, baseline_by_split["test"]))
    save_json(output_dir / "test_oracle_predictions.json", serialise(oracle_test, baseline_by_split["test"]))

    summary = {
        "method": "two-stage BART anchor plus source-switch gate",
        "method_short_name": "SSG",
        "workspace_root": str(workspace),
        "refs_root": str(captions_root),
        "refs_filename_template": args.refs_filename_template,
        "proposals_json": str(proposals_path),
        "proposal_method": proposals_payload.get("method"),
        "candidate_sources": args.candidate_sources,
        "non_bart_sources": sorted(allowed_sources),
        "max_blip_candidates": args.max_blip_candidates,
        "dev_splits": dev_splits,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "gate_feature_names": GATE_FEATURE_NAMES,
        "msrvtt_pretrain_type": pretrain_type,
        "msrvtt_pretrain_enabled": bool(bart_pretrain_rows),
        "msrvtt_pretrain_candidates": len(bart_pretrain_rows),
        "msrvtt_pretrain_metadata": pretrain_metadata,
        "pretrain_weight": args.pretrain_weight if bart_pretrain_rows else 0.0,
        "clip_video_text_features_enabled": bool(args.enable_clip_video_text_features),
        "clip_feature_roots": [str(path) for path in clip_feature_roots],
        "prior_evidence_summary_keywords": prior_evidence_keywords,
        "adaptive_fire_forest_prior_enabled": bool(args.enable_adaptive_fire_forest_prior),
        "best_config": best_config,
        "best_cv_result": best,
        "selection_constraints": {
            "min_oof_source_switches": args.min_oof_source_switches,
            "min_oof_positive_switches": args.min_oof_positive_switches,
            "max_oof_negative_switches": args.max_oof_negative_switches,
            "eligible_cv_results": len(eligible_cv_results),
            "fell_back_to_unconstrained": not bool(
                [
                    row
                    for row in cv_results
                    if row["oof_source_switches"] >= args.min_oof_source_switches
                    and row["oof_positive_switches"] >= args.min_oof_positive_switches
                    and row["oof_negative_switches"] <= args.max_oof_negative_switches
                ]
            ),
        },
        "cv_results_top10": cv_results[:10],
        "eligible_cv_results_top10": eligible_cv_results[:10],
        "gate_coefficients_by_abs_value": gate_coefficients(final_gate)[:50],
        "dev_baseline_metrics": baseline_dev_metrics,
        "dev_bart_anchor_metrics": bart_dev_metrics,
        "dev_metrics": dev_metrics,
        "dev_oracle_metrics": oracle_dev_metrics,
        "dev_cider_gain_vs_clean_baseline": dev_metrics["CIDEr"] - baseline_dev_metrics["CIDEr"],
        "dev_cider_gain_vs_bart_anchor": dev_metrics["CIDEr"] - bart_dev_metrics["CIDEr"],
        "dev_source_switches": source_switch_rows(selected_dev, dev_anchors),
        "dev_switches_vs_clean_baseline": switch_rows(selected_dev, dev_baseline),
        "dev_source_counts": source_counts(selected_dev),
        "test_baseline_metrics": baseline_test_metrics,
        "test_bart_anchor_metrics": bart_test_metrics,
        "test_metrics": test_metrics,
        "test_oracle_metrics": oracle_test_metrics,
        "test_cider_gain_vs_clean_baseline": test_metrics["CIDEr"] - baseline_test_metrics["CIDEr"],
        "test_cider_gain_vs_bart_anchor": test_metrics["CIDEr"] - bart_test_metrics["CIDEr"],
        "test_cider_gain_vs_current_reported_baseline": test_metrics["CIDEr"] - args.current_reported_cider,
        "test_source_switches": source_switch_rows(selected_test, test_anchors),
        "test_switches_vs_clean_baseline": switch_rows(selected_test, baseline_by_split["test"]),
        "test_source_counts": source_counts(selected_test),
        "test_baseline_source_counts": source_counts(baseline_test_rows),
        "test_bart_anchor_source_counts": source_counts(bart_test_rows),
        "test_oracle_source_counts": source_counts(oracle_test),
        "test_num_candidates": int(sum(len(rows) for rows in candidates_by_split["test"].values())),
        "dev_num_candidates": int(sum(len(rows) for rows in dev_candidates.values())),
        "interpretation_note": (
            "All gate hyperparameters are selected by video-grouped development "
            "cross-validation. Test references are used only for final scoring and "
            "oracle diagnostics."
        ),
    }
    save_json(output_dir / "summary.json", summary)
    write_markdown(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
