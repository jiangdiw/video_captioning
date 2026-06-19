from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


VIDEO_CAPTIONING_ROOT = Path("/Users/aglooney03/video_captioning")
OPENJDK_BIN = Path("/opt/homebrew/opt/openjdk/bin")
if OPENJDK_BIN.exists():
    os.environ["PATH"] = f"{OPENJDK_BIN}:{os.environ.get('PATH', '')}"
if str(VIDEO_CAPTIONING_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_CAPTIONING_ROOT))
if str(VIDEO_CAPTIONING_ROOT / "coco-caption") not in sys.path:
    sys.path.insert(0, str(VIDEO_CAPTIONING_ROOT / "coco-caption"))

from misc.cocoeval import COCOScorer  # type: ignore
from pycocoevalcap.cider.cider_scorer import CiderScorer, cook_refs, cook_test  # type: ignore


DEFAULT_WORKSPACE = Path("/Users/aglooney03/dattalion_transfer_workspace_local")
DEFAULT_ADAPTED_RUN = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "size_74_top5medoid_partial0_freezeNonBart_frombest120_cap20_v1"
)
DEFAULT_PROPOSALS = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "adaptive_blip_keyframe_proposals.json"
)
DEFAULT_FALLBACK_PROPOSALS = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "blip_keyframe_proposals.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "long_video_evidence_reranker_v1"
)

TOKEN_RE = re.compile(r"\b[a-zA-Z']+\b")

TERM_GROUPS: dict[str, set[str]] = {
    "ambulance": {"ambulance", "ambulances"},
    "building": {"apartment", "building", "buildings", "facility", "facilities", "house", "houses", "home", "homes"},
    "bus": {"bus", "buses"},
    "car": {"car", "cars", "vehicle", "vehicles", "van", "vans"},
    "church": {"church", "churches", "cathedral"},
    "debris": {"debris", "rubble", "wreckage", "ruins", "trash", "bricks", "fragments"},
    "damage": {"broken", "burned", "collapsed", "damaged", "demolished", "destroyed", "ruined", "shattered"},
    "fire": {"fire", "fires", "flame", "flames", "burning"},
    "firefighter": {"firefighter", "firefighters", "fireman", "firemen"},
    "horse": {"horse", "horses"},
    "hospital": {"hospital", "hospitals", "medical"},
    "people": {"crowd", "man", "men", "people", "person", "residents", "woman", "women", "workers"},
    "rescue": {"rescue", "rescuer", "rescuers", "paramedic", "paramedics", "emergency"},
    "road": {"road", "roads", "sidewalk", "street", "streets"},
    "smoke": {"smoke", "smoking"},
    "soldier": {"military", "soldier", "soldiers"},
    "truck": {"truck", "trucks"},
    "window": {"window", "windows", "glass"},
}

DAMAGE_GROUPS = {"damage", "debris", "fire", "smoke", "window"}
OBJECT_GROUPS = {
    "ambulance",
    "building",
    "bus",
    "car",
    "church",
    "firefighter",
    "horse",
    "hospital",
    "people",
    "rescue",
    "road",
    "soldier",
    "truck",
}
BAD_CAUSE_TERMS = {"earthquake", "flood", "hurricane", "tornado", "tsunami"}
GENERIC_TERMS = {"image", "photo", "picture", "screen", "closeup", "logo"}
TARGET_ACTION_TERMS = {
    "clear",
    "cleared",
    "clearing",
    "cleanup",
    "evacuated",
    "evacuation",
    "help",
    "remove",
    "removes",
    "respond",
    "walk",
    "work",
    "workers",
}

FORM_TO_GROUP = {
    form: group for group, forms in TERM_GROUPS.items() for form in forms
}


@dataclass
class Candidate:
    video_id: str
    caption: str
    references: list[str]
    source: str
    model_score: float | None = None
    model_rank: int | None = None
    proposal_rank: int | None = None
    proposal_fraction: float | None = None
    features: dict[str, float] = field(default_factory=dict)
    sentence_cider: float = 0.0


class SplitCiderScorer:
    def __init__(self, ground_truth: dict[str, list[dict[str, str]]], n: int = 4, sigma: float = 6.0) -> None:
        self.n = n
        self.sigma = sigma
        scorer = CiderScorer(n=n, sigma=sigma)
        for refs in ground_truth.values():
            scorer.cook_append(None, [row["caption"] for row in refs])
        scorer.compute_doc_freq()
        scorer.ref_len = np.log(float(len(scorer.crefs)))
        self.document_frequency = scorer.document_frequency
        self.ref_len = scorer.ref_len

    def score(self, references: list[str], candidate: str) -> float:
        def counts2vec(cnts):
            from collections import defaultdict

            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for ngram, term_freq in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                order = len(ngram) - 1
                vec[order][ngram] = float(term_freq) * (self.ref_len - df)
                norm[order] += float(vec[order][ngram] ** 2)
                if order == 1:
                    length += term_freq
            norm = [np.sqrt(item) for item in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            delta = float(length_hyp - length_ref)
            val = np.array([0.0 for _ in range(self.n)])
            for order in range(self.n):
                for ngram in vec_hyp[order]:
                    val[order] += min(vec_hyp[order][ngram], vec_ref[order][ngram]) * vec_ref[order][ngram]
                if norm_hyp[order] != 0 and norm_ref[order] != 0:
                    val[order] /= norm_hyp[order] * norm_ref[order]
                val[order] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        test = cook_test(candidate, self.n)
        refs = cook_refs(references, self.n)
        vec, norm, length = counts2vec(test)
        score = np.array([0.0 for _ in range(self.n)])
        for ref in refs:
            vec_ref, norm_ref, length_ref = counts2vec(ref)
            score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
        score_avg = np.mean(score)
        score_avg /= max(len(refs), 1)
        score_avg *= 10.0
        return float(score_avg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a train/validation-frozen long-video evidence reranker. "
            "The method re-ranks clean BART beam candidates with visual evidence "
            "from adaptive keyframe captions and can optionally admit evidence-summary captions."
        )
    )
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--adapted-run-dir", default=str(DEFAULT_ADAPTED_RUN))
    parser.add_argument("--proposals-json", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--fallback-proposals-json", default=str(DEFAULT_FALLBACK_PROPOSALS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dev-splits", default="train,val")
    parser.add_argument("--candidate-sources", choices=["bart", "bart_blip", "bart_blip_summary"], default="bart_blip_summary")
    parser.add_argument("--max-blip-candidates", type=int, default=8)
    parser.add_argument("--min-visual-weight", type=float, default=0.0)
    parser.add_argument("--min-support-weight", type=float, default=0.0)
    parser.add_argument(
        "--switch-margins",
        default="0.0",
        help="Comma-separated score margins required before overriding the model-score baseline.",
    )
    parser.add_argument("--current-reported-cider", type=float, default=0.20687050726627432)
    parser.add_argument("--seed", type=int, default=20260610)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def groups_for_text(text: str) -> set[str]:
    return {FORM_TO_GROUP[token] for token in tokenize(text) if token in FORM_TO_GROUP}


def domain_count(text: str) -> float:
    groups = groups_for_text(text)
    tokens = set(tokenize(text))
    action_hits = len(tokens & TARGET_ACTION_TERMS)
    return float(len(groups & (DAMAGE_GROUPS | OBJECT_GROUPS)) + 0.25 * action_hits)


def repetition_penalty(text: str) -> float:
    tokens = tokenize(text)
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for idx in range(1, len(tokens)) if tokens[idx] == tokens[idx - 1])
    return float(repeats)


def bad_penalty(text: str) -> float:
    tokens = set(tokenize(text))
    return float(1.5 * len(tokens & BAD_CAUSE_TERMS) + 0.5 * len(tokens & GENERIC_TERMS))


def make_ground_truth(refs: dict[str, list[str]]) -> dict[str, list[dict[str, str]]]:
    return {
        video_id: [{"caption": caption} for caption in captions]
        for video_id, captions in refs.items()
    }


def score_predictions(refs: dict[str, list[str]], preds: dict[str, str]) -> dict[str, float]:
    scorer = COCOScorer()
    ids = list(refs)
    gt = {video_id: refs[video_id] for video_id in ids}
    res = {video_id: [preds[video_id]] for video_id in ids}
    with contextlib.redirect_stdout(io.StringIO()):
        return scorer.score(gt, res, ids)


def build_target_idf(memory_refs: dict[str, list[str]]) -> dict[str, float]:
    doc_freq: Counter[str] = Counter()
    for captions in memory_refs.values():
        doc_tokens = set()
        for caption in captions:
            doc_tokens.update(tokenize(caption))
        doc_freq.update(doc_tokens)
    total_docs = max(1, len(memory_refs))
    return {
        token: math.log((1.0 + total_docs) / (1.0 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def target_lexical_score(text: str, idf: dict[str, float]) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    score = sum(idf.get(token, 0.0) for token in set(tokens))
    return float(score / math.sqrt(len(tokens)))


def load_bart_candidate_split(
    run_dir: Path,
    split: str,
    refs: dict[str, list[str]],
    scorer: SplitCiderScorer,
) -> tuple[dict[str, list[Candidate]], dict[str, Candidate]]:
    path = run_dir / f"beam_headroom_{split}_b8_nr8_lp1_nr2_cap20.json"
    payload = load_json(path)
    candidates_by_video: dict[str, list[Candidate]] = {}
    baseline_by_video: dict[str, Candidate] = {}
    for record in payload["per_video"]:
        video_id = record["video_id"]
        references = refs[video_id]
        sorted_by_model = sorted(
            record["all_candidates"],
            key=lambda row: row.get("model_score", -999.0),
            reverse=True,
        )
        seen: set[str] = set()
        rows: list[Candidate] = []
        for rank, row in enumerate(sorted_by_model):
            caption = " ".join(row["caption"].strip().split())
            if not caption or caption.lower() in seen:
                continue
            seen.add(caption.lower())
            candidate = Candidate(
                video_id=video_id,
                caption=caption,
                references=references,
                source="bart",
                model_score=float(row.get("model_score", -99.0)),
                model_rank=rank,
                sentence_cider=scorer.score(references, caption),
            )
            rows.append(candidate)
        if not rows:
            raise ValueError(f"No BART candidates for {video_id} in {path}")
        candidates_by_video[video_id] = rows
        baseline_by_video[video_id] = rows[0]
    return candidates_by_video, baseline_by_video


def proposal_frame_key(proposal: dict) -> str:
    if proposal.get("time_sec") is not None:
        return f"time:{float(proposal['time_sec']):.2f}"
    if proposal.get("frame_index") is not None:
        return f"frame:{proposal['frame_index']}"
    if proposal.get("fraction") is not None:
        return f"frac:{float(proposal['fraction']):.3f}"
    return f"rank:{proposal.get('rank', 0)}"


def evidence_profile(proposals: list[dict]) -> dict:
    group_frames: dict[str, set[str]] = defaultdict(set)
    token_counts: Counter[str] = Counter()
    usable = []
    for proposal in proposals:
        caption = " ".join(str(proposal.get("caption", "")).strip().split())
        if not caption:
            continue
        tokens = tokenize(caption)
        if set(tokens) & BAD_CAUSE_TERMS:
            continue
        frame_key = proposal_frame_key(proposal)
        groups = groups_for_text(caption)
        for group in groups:
            group_frames[group].add(frame_key)
        token_counts.update(tokens)
        usable.append(proposal)
    frame_count = len({proposal_frame_key(row) for row in usable})
    return {
        "usable_proposals": usable,
        "frame_count": frame_count,
        "group_support": {group: len(frames) for group, frames in group_frames.items()},
        "token_counts": dict(token_counts),
    }


def support(profile: dict, group: str) -> int:
    return int(profile["group_support"].get(group, 0))


def has(profile: dict, group: str, min_support: int = 1) -> bool:
    return support(profile, group) >= min_support


def visual_alignment(text: str, profile: dict) -> float:
    groups = groups_for_text(text)
    if not groups:
        return 0.0
    frame_count = max(1, int(profile.get("frame_count", 0)))
    total = 0.0
    for group in groups:
        group_weight = 1.35 if group in DAMAGE_GROUPS else 1.0
        total += group_weight * min(1.0, support(profile, group) / frame_count)
    return float(total / math.sqrt(len(groups)))


def evidence_support_score(text: str, profile: dict) -> float:
    groups = groups_for_text(text)
    if not groups:
        return 0.0
    return float(sum(min(3, support(profile, group)) for group in groups) / math.sqrt(len(groups)))


def add_summary(summaries: list[str], caption: str) -> None:
    caption = " ".join(caption.strip().split())
    if caption and caption.lower() not in {row.lower() for row in summaries}:
        summaries.append(caption)


def evidence_summary_captions(profile: dict) -> list[str]:
    summaries: list[str] = []
    damage_seen = any(has(profile, group) for group in DAMAGE_GROUPS)
    building_seen = has(profile, "building")
    people_seen = has(profile, "people")

    if has(profile, "horse"):
        add_summary(summaries, "horses are evacuated through a damaged street")
        add_summary(summaries, "people guide horses along a damaged road")
    if has(profile, "fire") or has(profile, "smoke"):
        if has(profile, "firefighter") or has(profile, "rescue"):
            add_summary(summaries, "firefighters respond to fire and smoke near damaged buildings")
        add_summary(summaries, "smoke rises from damaged buildings after a fire")
        add_summary(summaries, "fire and smoke are visible near damaged buildings")
    if has(profile, "hospital") and damage_seen:
        add_summary(summaries, "workers clear debris from a damaged hospital")
        add_summary(summaries, "a damaged medical building is shown with debris")
    if has(profile, "church") and damage_seen:
        add_summary(summaries, "a damaged church building is shown with rubble")
    if building_seen and (has(profile, "debris") or has(profile, "damage")):
        if has(profile, "rescue") or has(profile, "soldier"):
            add_summary(summaries, "responders clear debris from a damaged building")
        add_summary(summaries, "people walk through damaged buildings and rubble")
        add_summary(summaries, "damaged buildings and debris are shown")
    if has(profile, "car") and damage_seen:
        add_summary(summaries, "damaged vehicles sit near buildings and debris")
        add_summary(summaries, "vehicles are shown beside damaged buildings")
    if has(profile, "road") and damage_seen:
        if people_seen:
            add_summary(summaries, "people walk along a damaged street with debris")
        add_summary(summaries, "a damaged road is covered with debris")
    if damage_seen and not summaries:
        add_summary(summaries, "damage and debris are shown after an attack")
    return summaries


def add_visual_candidates(
    candidates_by_video: dict[str, list[Candidate]],
    refs: dict[str, list[str]],
    scorer: SplitCiderScorer,
    proposals_payload: dict,
    idf: dict[str, float],
    mode: str,
    max_blip_candidates: int,
) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    per_video = proposals_payload.get("per_video", {})
    for video_id, references in refs.items():
        proposals = per_video.get(video_id, {}).get("proposals", [])
        profile = evidence_profile(proposals)
        profiles[video_id] = profile
        rows = candidates_by_video.setdefault(video_id, [])
        seen = {candidate.caption.lower() for candidate in rows}
        if mode in {"bart_blip", "bart_blip_summary"}:
            scored_proposals = []
            for proposal in profile["usable_proposals"]:
                caption = " ".join(proposal["caption"].strip().split())
                if not caption or caption.lower() in seen:
                    continue
                score = (
                    1.15 * visual_alignment(caption, profile)
                    + 0.20 * target_lexical_score(caption, idf)
                    + 0.25 * domain_count(caption)
                    - 0.25 * float(proposal.get("rank", 0))
                    - bad_penalty(caption)
                )
                scored_proposals.append((score, proposal, caption))
            scored_proposals.sort(key=lambda item: item[0], reverse=True)
            for _, proposal, caption in scored_proposals[:max_blip_candidates]:
                seen.add(caption.lower())
                rows.append(
                    Candidate(
                        video_id=video_id,
                        caption=caption,
                        references=references,
                        source="adaptive_blip",
                        proposal_rank=int(proposal.get("rank", 99)),
                        proposal_fraction=(
                            float(proposal["fraction"])
                            if proposal.get("fraction") is not None
                            else None
                        ),
                        sentence_cider=scorer.score(references, caption),
                    )
                )
        if mode == "bart_blip_summary":
            for caption in evidence_summary_captions(profile):
                if caption.lower() in seen:
                    continue
                seen.add(caption.lower())
                rows.append(
                    Candidate(
                        video_id=video_id,
                        caption=caption,
                        references=references,
                        source="evidence_summary",
                        sentence_cider=scorer.score(references, caption),
                    )
                )
    return profiles


def compute_candidate_features(
    candidate: Candidate,
    baseline: Candidate,
    profile: dict,
    idf: dict[str, float],
) -> dict[str, float]:
    length = len(tokenize(candidate.caption))
    model_rel = (
        float(candidate.model_score - baseline.model_score)
        if candidate.model_score is not None and baseline.model_score is not None
        else -1.75
    )
    model_rank = float(candidate.model_rank if candidate.model_rank is not None else 12)
    source_bias = {
        "bart": 0.0,
        "adaptive_blip": -0.50,
        "evidence_summary": -0.35,
    }.get(candidate.source, -0.75)
    features = {
        "model_rel": model_rel,
        "model_rank_penalty": -model_rank,
        "visual_alignment": visual_alignment(candidate.caption, profile),
        "evidence_support": evidence_support_score(candidate.caption, profile),
        "target_lexical": target_lexical_score(candidate.caption, idf),
        "domain_count": domain_count(candidate.caption),
        "source_bias": source_bias,
        "is_bart": 1.0 if candidate.source == "bart" else 0.0,
        "is_visual": 1.0 if candidate.source in {"adaptive_blip", "evidence_summary"} else 0.0,
        "length_penalty": abs(length - 8.0) / 8.0,
        "repetition_penalty": repetition_penalty(candidate.caption),
        "bad_penalty": bad_penalty(candidate.caption),
    }
    candidate.features = features
    return features


def candidate_score(candidate: Candidate, weights: dict[str, float]) -> float:
    return (
        weights["model"] * candidate.features["model_rel"]
        + weights["rank"] * candidate.features["model_rank_penalty"]
        + weights["visual"] * candidate.features["visual_alignment"]
        + weights["support"] * candidate.features["evidence_support"]
        + weights["lexical"] * candidate.features["target_lexical"]
        + weights["domain"] * candidate.features["domain_count"]
        + weights["source"] * candidate.features["source_bias"]
        - weights["length"] * candidate.features["length_penalty"]
        - weights["repeat"] * candidate.features["repetition_penalty"]
        - weights["bad"] * candidate.features["bad_penalty"]
    )


def select_with_weights(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    weights: dict[str, float],
) -> list[Candidate]:
    selected: list[Candidate] = []
    for video_id in refs:
        rows = candidates_by_video[video_id]
        baseline = next((row for row in rows if row.source == "bart" and row.model_rank == 0), rows[0])
        baseline_score = candidate_score(baseline, weights)
        scored = []
        for row in rows:
            score = candidate_score(row, weights)
            baseline_tiebreak = 1.0 if row.model_rank == 0 and row.source == "bart" else 0.0
            scored.append((score, baseline_tiebreak, -float(row.model_rank if row.model_rank is not None else 99), row))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_score, _, _, best = scored[0]
        switch_margin = float(weights.get("switch_margin", 0.0))
        if best.caption != baseline.caption and best_score - baseline_score < switch_margin:
            selected.append(baseline)
        else:
            selected.append(best)
    return selected


def rows_to_predictions(rows: list[Candidate]) -> dict[str, str]:
    return {row.video_id: row.caption for row in rows}


def serialise(rows: list[Candidate], baseline_by_video: dict[str, Candidate] | None = None) -> list[dict]:
    output = []
    for row in rows:
        item = {
            "video_id": row.video_id,
            "caption": row.caption,
            "source": row.source,
            "model_score": row.model_score,
            "model_rank": row.model_rank,
            "sentence_cider": row.sentence_cider,
            "features": row.features,
            "references": row.references,
        }
        if baseline_by_video is not None:
            baseline = baseline_by_video[row.video_id]
            item["baseline_caption"] = baseline.caption
            item["baseline_sentence_cider"] = baseline.sentence_cider
            item["switched_from_baseline"] = row.caption != baseline.caption
        output.append(item)
    return output


def score_rows(rows: list[Candidate]) -> dict[str, float]:
    return score_predictions(
        {row.video_id: row.references for row in rows},
        {row.video_id: row.caption for row in rows},
    )


def mean_sentence_cider(rows: list[Candidate]) -> float:
    if not rows:
        return 0.0
    return float(np.mean([row.sentence_cider for row in rows]))


def build_weight_grid(switch_margins: str) -> list[dict[str, float]]:
    grid = []
    for model in [0.75, 1.0, 1.25, 1.5, 2.0]:
        for rank in [0.0, 0.02, 0.05]:
            for visual in [0.0, 0.5, 1.0, 1.5, 2.0]:
                for support_weight in [0.0, 0.25, 0.5]:
                    for lexical in [0.0, 0.05, 0.10, 0.20]:
                        for domain in [0.0, 0.05, 0.10, 0.20]:
                            grid.append(
                                {
                                    "model": model,
                                    "rank": rank,
                                    "visual": visual,
                                    "support": support_weight,
                                    "lexical": lexical,
                                    "domain": domain,
                                    "source": 1.0,
                                    "length": 0.05,
                                    "repeat": 0.20,
                                    "bad": 1.0,
                                    "switch_margin": 0.0,
                                }
                            )
    margins = [
        float(item.strip())
        for item in switch_margins.split(",")
        if item.strip()
    ]
    if not margins:
        raise ValueError("--switch-margins must contain at least one numeric value")
    expanded = []
    for weights in grid:
        for margin in margins:
            row = dict(weights)
            row["switch_margin"] = margin
            expanded.append(row)
    return expanded


def prepare_split(
    split: str,
    refs: dict[str, list[str]],
    run_dir: Path,
    scorer: SplitCiderScorer,
    proposals_payload: dict,
    idf: dict[str, float],
    candidate_sources: str,
    max_blip_candidates: int,
) -> tuple[dict[str, list[Candidate]], dict[str, Candidate], dict[str, dict]]:
    candidates_by_video, baseline_by_video = load_bart_candidate_split(run_dir, split, refs, scorer)
    profiles = add_visual_candidates(
        candidates_by_video=candidates_by_video,
        refs=refs,
        scorer=scorer,
        proposals_payload=proposals_payload,
        idf=idf,
        mode=candidate_sources,
        max_blip_candidates=max_blip_candidates,
    )
    for video_id, rows in candidates_by_video.items():
        baseline = baseline_by_video[video_id]
        profile = profiles[video_id]
        for row in rows:
            compute_candidate_features(row, baseline, profile, idf)
    return candidates_by_video, baseline_by_video, profiles


def candidate_pool_oracle(refs: dict[str, list[str]], candidates_by_video: dict[str, list[Candidate]]) -> list[Candidate]:
    return [max(candidates_by_video[video_id], key=lambda row: row.sentence_cider) for video_id in refs]


def source_counts(rows: list[Candidate]) -> dict[str, int]:
    counts: Counter[str] = Counter(row.source for row in rows)
    return dict(sorted(counts.items()))


def switch_rows(selected: list[Candidate], baseline_by_video: dict[str, Candidate]) -> list[dict]:
    switches = []
    for row in selected:
        baseline = baseline_by_video[row.video_id]
        if row.caption == baseline.caption:
            continue
        switches.append(
            {
                "video_id": row.video_id,
                "source": row.source,
                "baseline_caption": baseline.caption,
                "selected_caption": row.caption,
                "baseline_sentence_cider": baseline.sentence_cider,
                "selected_sentence_cider": row.sentence_cider,
                "sentence_cider_delta": row.sentence_cider - baseline.sentence_cider,
                "features": row.features,
            }
        )
    return switches


def main() -> None:
    args = parse_args()
    np.random.default_rng(args.seed)
    workspace = Path(args.workspace_root).expanduser().resolve()
    captions_root = workspace / "processed_clean" / "captions"
    adapted_run = Path(args.adapted_run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = Path(args.proposals_json).expanduser().resolve()
    if not proposals_path.exists():
        proposals_path = Path(args.fallback_proposals_json).expanduser().resolve()
    proposals_payload = load_json(proposals_path)

    refs_by_split = {
        split: load_json(captions_root / f"{split}_captions.json")
        for split in ["train", "val", "test"]
    }
    scorer_by_split = {
        split: SplitCiderScorer(make_ground_truth(refs))
        for split, refs in refs_by_split.items()
    }
    dev_splits = [item.strip() for item in args.dev_splits.split(",") if item.strip()]
    memory_refs = {}
    for split in dev_splits:
        memory_refs.update(refs_by_split[split])
    idf = build_target_idf(memory_refs)

    candidates_by_split = {}
    baseline_by_split = {}
    profiles_by_split = {}
    for split in ["train", "val", "test"]:
        candidates, baseline, profiles = prepare_split(
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
        profiles_by_split[split] = profiles

    dev_refs = {}
    dev_candidates: dict[str, list[Candidate]] = {}
    dev_baseline: dict[str, Candidate] = {}
    for split in dev_splits:
        dev_refs.update(refs_by_split[split])
        dev_candidates.update(candidates_by_split[split])
        dev_baseline.update(baseline_by_split[split])

    baseline_dev_rows = [dev_baseline[video_id] for video_id in dev_refs]
    baseline_test_rows = [baseline_by_split["test"][video_id] for video_id in refs_by_split["test"]]
    baseline_dev_metrics = score_rows(baseline_dev_rows)
    baseline_test_metrics = score_rows(baseline_test_rows)

    weight_grid = [
        weights
        for weights in build_weight_grid(args.switch_margins)
        if weights["visual"] >= args.min_visual_weight
        and weights["support"] >= args.min_support_weight
    ]
    if not weight_grid:
        raise ValueError("No weights remain after applying --min-visual-weight/--min-support-weight")

    grid_results = []
    for weights in weight_grid:
        selected = select_with_weights(dev_refs, dev_candidates, weights)
        switches = switch_rows(selected, dev_baseline)
        grid_results.append(
            {
                "weights": weights,
                "dev_mean_sentence_cider": mean_sentence_cider(selected),
                "num_dev_switches": len(switches),
                "source_counts": source_counts(selected),
            }
        )
    grid_results.sort(
        key=lambda row: (
            row["dev_mean_sentence_cider"],
            -row["num_dev_switches"],
        ),
        reverse=True,
    )
    best_weights = grid_results[0]["weights"]

    selected_dev = select_with_weights(dev_refs, dev_candidates, best_weights)
    selected_test = select_with_weights(refs_by_split["test"], candidates_by_split["test"], best_weights)
    oracle_test = candidate_pool_oracle(refs_by_split["test"], candidates_by_split["test"])
    oracle_dev = candidate_pool_oracle(dev_refs, dev_candidates)

    dev_metrics = score_rows(selected_dev)
    test_metrics = score_rows(selected_test)
    oracle_dev_metrics = score_rows(oracle_dev)
    oracle_test_metrics = score_rows(oracle_test)

    save_json(output_dir / "weight_grid_top50.json", grid_results[:50])
    save_json(output_dir / "dev_predictions.json", serialise(selected_dev, dev_baseline))
    save_json(output_dir / "test_predictions.json", serialise(selected_test, baseline_by_split["test"]))
    save_json(output_dir / "test_oracle_predictions.json", serialise(oracle_test, baseline_by_split["test"]))

    summary = {
        "method": "long-video evidence-calibrated reranker",
        "method_short_name": "LV-ECR",
        "proposals_json": str(proposals_path),
        "proposal_method": proposals_payload.get("method"),
        "candidate_sources": args.candidate_sources,
        "grid_constraints": {
            "min_visual_weight": args.min_visual_weight,
            "min_support_weight": args.min_support_weight,
            "num_weight_settings": len(weight_grid),
        },
        "rank_source_note": "Clean: BART beam order is recomputed from model generation scores; stored CIDEr order is not used.",
        "dev_splits": dev_splits,
        "best_weights": best_weights,
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
        "test_num_candidates": int(sum(len(rows) for rows in candidates_by_split["test"].values())),
        "dev_num_candidates": int(sum(len(rows) for rows in dev_candidates.values())),
        "top_grid_results": grid_results[:10],
        "interpretation_note": (
            "Weights are selected only on the requested development splits. "
            "Held-out test captions are used only for final scoring and oracle diagnostics."
        ),
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
