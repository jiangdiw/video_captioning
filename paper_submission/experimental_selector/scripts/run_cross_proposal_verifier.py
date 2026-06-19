from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_long_video_evidence_reranker import (  # type: ignore  # noqa: E402
    DEFAULT_ADAPTED_RUN,
    DEFAULT_FALLBACK_PROPOSALS,
    DEFAULT_PROPOSALS,
    DEFAULT_WORKSPACE,
    Candidate,
    DAMAGE_GROUPS,
    OBJECT_GROUPS,
    SplitCiderScorer,
    bad_penalty,
    build_target_idf,
    candidate_pool_oracle,
    groups_for_text,
    load_json,
    make_ground_truth,
    prepare_split,
    proposal_frame_key,
    repetition_penalty,
    save_json,
    score_rows,
    serialise,
    source_counts,
    switch_rows,
    tokenize,
)
from train_frozen_learned_selector import parse_csv_list, parse_float_list  # type: ignore  # noqa: E402
from train_source_switch_gate import token_jaccard  # type: ignore  # noqa: E402


DEFAULT_LARGE_PROPOSALS = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "blip_large_keyframe_proposals.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_WORKSPACE
    / "runs_clean"
    / "transfer_arch_search_20260610"
    / "cross_proposal_verifier_v1"
)

EXTRA_TERM_GROUPS: dict[str, set[str]] = {
    "forest": {"forest", "forests", "tree", "trees", "wooded", "woods"},
    "field": {"field", "fields"},
    "helmet": {"helmet", "helmets"},
    "hose": {"hose", "hoses"},
    "night": {"night", "nighttime"},
    "roof": {"roof", "roofs"},
}
HIGH_SPECIFICITY_GROUPS = {"ambulance", "church", "forest", "horse", "hospital"}
BLOCKED_TOKENS = {
    "aleppo",
    "bronx",
    "earthquake",
    "fucked",
    "gaza",
    "hurricane",
    "israeli",
    "logo",
    "naked",
    "petersburg",
    "philippines",
    "sex",
    "syria",
    "tornado",
}
BLOCKED_SUBSTRINGS = {
    "borough borough",
    "since since",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a cross-proposal visual verifier on top of a BART-only selector. "
            "The verifier admits only high-specificity non-BART switches that are "
            "supported by multiple independent keyframe proposal streams."
        )
    )
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--refs-root", default="")
    parser.add_argument("--refs-filename-template", default="{split}_captions.json")
    parser.add_argument("--adapted-run-dir", default=str(DEFAULT_ADAPTED_RUN))
    parser.add_argument("--proposals-json", default=str(DEFAULT_PROPOSALS))
    parser.add_argument("--fallback-proposals-json", default=str(DEFAULT_FALLBACK_PROPOSALS))
    parser.add_argument(
        "--verifier-proposals-jsons",
        default=",".join(
            [
                str(DEFAULT_PROPOSALS),
                str(DEFAULT_FALLBACK_PROPOSALS),
                str(DEFAULT_LARGE_PROPOSALS),
            ]
        ),
        help="Comma-separated proposal files used as independent verifier streams.",
    )
    parser.add_argument("--selector-results-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dev-splits", default="train,val")
    parser.add_argument("--candidate-sources", choices=["bart", "bart_blip", "bart_blip_summary"], default="bart_blip_summary")
    parser.add_argument("--max-blip-candidates", type=int, default=12)
    parser.add_argument("--thresholds", default="6,7,8,9,10,11")
    parser.add_argument("--current-reported-cider", type=float, default=0.20687050726627432)
    return parser.parse_args()


def score_rows_quiet(rows: list[Candidate]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        return score_rows(rows)


def normalise_caption(text: str) -> str:
    return " ".join(text.strip().lower().split())


def extended_groups_for_text(text: str) -> set[str]:
    groups = set(groups_for_text(text))
    tokens = set(tokenize(text))
    for group, forms in EXTRA_TERM_GROUPS.items():
        if tokens & forms:
            groups.add(group)
    return groups


def load_selector_predictions(selector_dir: Path, name: str) -> dict[str, dict]:
    path = selector_dir / name
    if not path.exists():
        return {}
    records = load_json(path)
    return {record["video_id"]: record for record in records}


def selector_anchor(
    candidates_by_video: dict[str, list[Candidate]],
    baseline_by_video: dict[str, Candidate],
    video_id: str,
    selector_records: dict[str, dict],
) -> Candidate:
    record = selector_records.get(video_id)
    if not record:
        return baseline_by_video[video_id]
    caption = normalise_caption(str(record.get("caption", "")))
    rows = candidates_by_video[video_id]
    for row in rows:
        if row.source == "bart" and normalise_caption(row.caption) == caption:
            return row
    if record.get("source") and record.get("source") != "bart":
        for row in rows:
            if normalise_caption(row.caption) == caption:
                return row
    return baseline_by_video[video_id]


def proposal_group_support(payload: dict, video_id: str) -> dict[str, int]:
    proposals = payload.get("per_video", {}).get(video_id, {}).get("proposals", [])
    frames_by_group: dict[str, set[str]] = defaultdict(set)
    for proposal in proposals:
        caption = " ".join(str(proposal.get("caption", "")).strip().split())
        if not caption:
            continue
        frame_key = proposal_frame_key(proposal)
        for group in extended_groups_for_text(caption):
            frames_by_group[group].add(frame_key)
    return {group: len(frames) for group, frames in frames_by_group.items()}


def build_support_cache(
    verifier_payloads: dict[str, dict],
    video_ids: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    return {
        video_id: {
            label: proposal_group_support(payload, video_id)
            for label, payload in verifier_payloads.items()
        }
        for video_id in video_ids
    }


def source_support_count(
    support_cache: dict[str, dict[str, dict[str, int]]],
    video_id: str,
    group: str,
    min_support: int = 1,
) -> int:
    return int(
        sum(
            1
            for group_support in support_cache[video_id].values()
            if group_support.get(group, 0) >= min_support
        )
    )


def total_group_support(
    support_cache: dict[str, dict[str, dict[str, int]]],
    video_id: str,
    group: str,
) -> int:
    return int(
        sum(group_support.get(group, 0) for group_support in support_cache[video_id].values())
    )


def blocked_caption(text: str) -> bool:
    lowered = text.lower()
    tokens = set(tokenize(text))
    return (
        bool(tokens & BLOCKED_TOKENS)
        or any(pattern in lowered for pattern in BLOCKED_SUBSTRINGS)
        or bad_penalty(text) > 0.0
        or repetition_penalty(text) > 2.0
    )


def verifier_score(
    candidate: Candidate,
    anchor: Candidate,
    support_cache: dict[str, dict[str, dict[str, int]]],
) -> tuple[float, str, dict[str, float]]:
    if blocked_caption(candidate.caption):
        return -1e9, "blocked", {}

    video_id = candidate.video_id
    candidate_groups = extended_groups_for_text(candidate.caption)
    anchor_groups = extended_groups_for_text(anchor.caption)
    text = candidate.caption.lower()
    reasons: list[str] = []
    score = 0.0

    def source_count(group: str, min_support: int = 1) -> int:
        return source_support_count(support_cache, video_id, group, min_support)

    def verified(group: str, min_sources: int = 2) -> bool:
        return source_count(group) >= min_sources

    if (
        candidate.source == "evidence_summary"
        and {"horse", "road"} <= candidate_groups
        and "horse" not in anchor_groups
        and verified("horse")
        and verified("road")
        and source_count("damage") >= 1
    ):
        score += 10.0
        reasons.append("horse_road")

    if (
        candidate.source == "evidence_summary"
        and "church" in candidate_groups
        and "church" not in anchor_groups
        and verified("church")
        and verified("building")
        and verified("damage")
    ):
        score += 9.0
        reasons.append("church_damage")

    if (
        candidate.source == "adaptive_blip"
        and {"fire", "forest"} <= candidate_groups
        and "forest" not in anchor_groups
        and verified("fire")
        and verified("forest")
    ):
        score += 8.0
        reasons.append("fire_forest")
        if "burns through" in text:
            score += 1.00
        if "burning through" in text:
            score += 0.70
        score -= 0.05 * len(tokenize(candidate.caption))

    if (
        candidate.source == "adaptive_blip"
        and {"building", "car", "fire"} <= candidate_groups
        and "car" not in anchor_groups
        and verified("building")
        and verified("car")
        and verified("fire")
    ):
        score += 7.0
        reasons.append("car_fire_building")
        score -= 0.35 * float(candidate.proposal_rank or 0)
        if "front of a building" in text:
            score += 0.30
        if "destroyed building" in text:
            score += 0.15

    if not reasons:
        return -1e9, "", {}

    support_total = sum(total_group_support(support_cache, video_id, group) for group in candidate_groups)
    high_specificity_support = sum(
        source_count(group)
        for group in candidate_groups
        if group in HIGH_SPECIFICITY_GROUPS
    )
    score += 0.03 * support_total
    score += 0.10 * high_specificity_support
    score += 0.02 * candidate.features.get("target_lexical", 0.0)
    score -= 0.20 * token_jaccard(candidate.caption, anchor.caption)

    components = {
        "support_total": float(support_total),
        "high_specificity_support": float(high_specificity_support),
        "token_jaccard_to_anchor": token_jaccard(candidate.caption, anchor.caption),
        "candidate_group_count": float(len(candidate_groups)),
        "anchor_group_count": float(len(anchor_groups)),
    }
    return float(score), "+".join(reasons), components


def select_with_verifier(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    anchors_by_video: dict[str, Candidate],
    support_cache: dict[str, dict[str, dict[str, int]]],
    threshold: float,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for video_id in refs:
        anchor = anchors_by_video[video_id]
        scored: list[tuple[float, Candidate]] = []
        for candidate in candidates_by_video[video_id]:
            if candidate.source == "bart":
                continue
            score, reason, components = verifier_score(candidate, anchor, support_cache)
            candidate.features["cross_proposal_verifier_score"] = score
            candidate.features["cross_proposal_verifier_reason_count"] = float(bool(reason))
            for name, value in components.items():
                candidate.features[f"cross_proposal_{name}"] = value
            scored.append((score, candidate))
        best_score, best_candidate = max(scored, key=lambda item: item[0])
        if best_score >= threshold:
            selected.append(best_candidate)
        else:
            selected.append(anchor)
    return selected


def anchor_rows(
    refs: dict[str, list[str]],
    candidates_by_video: dict[str, list[Candidate]],
    baseline_by_video: dict[str, Candidate],
    selector_records: dict[str, dict],
) -> dict[str, Candidate]:
    return {
        video_id: selector_anchor(candidates_by_video, baseline_by_video, video_id, selector_records)
        for video_id in refs
    }


def source_switch_rows(selected: list[Candidate], anchors_by_video: dict[str, Candidate]) -> list[dict]:
    switches = []
    for row in selected:
        anchor = anchors_by_video[row.video_id]
        if row.caption == anchor.caption:
            continue
        score, reason, components = verifier_score(row, anchor, source_switch_rows.support_cache)
        switches.append(
            {
                "video_id": row.video_id,
                "source": row.source,
                "anchor_caption": anchor.caption,
                "selected_caption": row.caption,
                "anchor_sentence_cider": anchor.sentence_cider,
                "selected_sentence_cider": row.sentence_cider,
                "sentence_cider_delta": row.sentence_cider - anchor.sentence_cider,
                "verifier_score": score,
                "verifier_reason": reason,
                "verifier_components": components,
                "features": row.features,
            }
        )
    return switches


source_switch_rows.support_cache = {}  # type: ignore[attr-defined]


def selected_metadata(
    rows: list[Candidate],
    baseline_by_video: dict[str, Candidate],
    anchors_by_video: dict[str, Candidate],
) -> list[dict]:
    payload = serialise(rows, baseline_by_video)
    for item in payload:
        anchor = anchors_by_video[item["video_id"]]
        item["anchor_caption"] = anchor.caption
        item["anchor_source"] = anchor.source
        item["anchor_sentence_cider"] = anchor.sentence_cider
        item["switched_from_anchor"] = item["caption"] != anchor.caption
        item["sentence_cider_delta_vs_anchor"] = item["sentence_cider"] - anchor.sentence_cider
    return payload


def write_markdown(output_dir: Path, summary: dict) -> None:
    lines = [
        "# Cross-Proposal Verifier",
        "",
        "## Method",
        "",
        (
            "The method starts from the MSR-VTT-pretrained BART selector and only "
            "allows a non-BART source switch when adaptive, base, and/or large BLIP "
            "keyframe proposal streams independently support a high-specificity "
            "visual concept."
        ),
        "",
        "## Test Results",
        "",
        "| Method | CIDEr | BLEU-4 | Source counts |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, label, source_key in [
        ("test_clean_baseline_metrics", "clean BART baseline", "test_clean_baseline_source_counts"),
        ("test_anchor_metrics", "BART selector anchor", "test_anchor_source_counts"),
        ("test_metrics", "cross-proposal verifier", "test_source_counts"),
        ("test_oracle_metrics", "candidate-pool oracle", "test_oracle_source_counts"),
    ]:
        metrics = summary[key]
        lines.append(
            f"| {label} | {metrics['CIDEr']:.4f} | {metrics.get('Bleu_4', 0.0):.4f} | `{summary[source_key]}` |"
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Best dev threshold: `{summary['best_threshold']}`",
            f"- OOF dev CIDEr: `{summary['dev_metrics']['CIDEr']:.4f}`",
            f"- Test CIDEr gain over clean baseline: `{summary['test_cider_gain_vs_clean_baseline']:.4f}`",
            f"- Test CIDEr gain over selector anchor: `{summary['test_cider_gain_vs_anchor']:.4f}`",
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

    verifier_paths = [
        Path(item).expanduser().resolve()
        for item in parse_csv_list(args.verifier_proposals_jsons, "--verifier-proposals-jsons")
    ]
    verifier_payloads = {
        path.stem.replace("_keyframe_proposals", ""): load_json(path)
        for path in verifier_paths
        if path.exists()
    }
    if not verifier_payloads:
        raise ValueError("No verifier proposal files were found.")

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

    selector_dir = Path(args.selector_results_dir).expanduser().resolve() if args.selector_results_dir.strip() else None
    dev_oof_records = load_selector_predictions(selector_dir, "dev_oof_predictions.json") if selector_dir else {}
    dev_records = load_selector_predictions(selector_dir, "dev_predictions.json") if selector_dir else {}
    test_records = load_selector_predictions(selector_dir, "test_predictions.json") if selector_dir else {}
    if not dev_oof_records and dev_records:
        dev_oof_records = dev_records

    dev_refs: dict[str, list[str]] = {}
    dev_candidates: dict[str, list[Candidate]] = {}
    dev_baseline: dict[str, Candidate] = {}
    dev_anchor_records: dict[str, dict] = {}
    for split in dev_splits:
        dev_refs.update(refs_by_split[split])
        dev_candidates.update(candidates_by_split[split])
        dev_baseline.update(baseline_by_split[split])
        dev_anchor_records.update(dev_oof_records)

    all_video_ids = list(dev_refs) + list(refs_by_split["test"])
    support_cache = build_support_cache(verifier_payloads, all_video_ids)
    source_switch_rows.support_cache = support_cache  # type: ignore[attr-defined]

    dev_anchors = anchor_rows(dev_refs, dev_candidates, dev_baseline, dev_anchor_records)
    test_anchors = anchor_rows(
        refs_by_split["test"],
        candidates_by_split["test"],
        baseline_by_split["test"],
        test_records,
    )

    thresholds = parse_float_list(args.thresholds, "--thresholds")
    threshold_results = []
    for threshold in thresholds:
        selected = select_with_verifier(
            dev_refs,
            dev_candidates,
            dev_anchors,
            support_cache,
            threshold,
        )
        switches = source_switch_rows(selected, dev_anchors)
        metrics = score_rows_quiet(selected)
        threshold_results.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "mean_sentence_cider": float(np.mean([row.sentence_cider for row in selected])),
                "source_counts": source_counts(selected),
                "source_switches": len(switches),
                "positive_switches": int(sum(1 for row in switches if row["sentence_cider_delta"] > 0.0)),
                "negative_switches": int(sum(1 for row in switches if row["sentence_cider_delta"] <= 0.0)),
            }
        )
    threshold_results.sort(
        key=lambda row: (
            row["metrics"]["CIDEr"],
            row["mean_sentence_cider"],
            row["positive_switches"],
            -row["negative_switches"],
        ),
        reverse=True,
    )
    best_threshold = float(threshold_results[0]["threshold"])

    selected_dev = select_with_verifier(dev_refs, dev_candidates, dev_anchors, support_cache, best_threshold)
    selected_test = select_with_verifier(
        refs_by_split["test"],
        candidates_by_split["test"],
        test_anchors,
        support_cache,
        best_threshold,
    )

    clean_dev_rows = [dev_baseline[video_id] for video_id in dev_refs]
    clean_test_rows = [baseline_by_split["test"][video_id] for video_id in refs_by_split["test"]]
    anchor_dev_rows = [dev_anchors[video_id] for video_id in dev_refs]
    anchor_test_rows = [test_anchors[video_id] for video_id in refs_by_split["test"]]
    oracle_dev = candidate_pool_oracle(dev_refs, dev_candidates)
    oracle_test = candidate_pool_oracle(refs_by_split["test"], candidates_by_split["test"])

    clean_dev_metrics = score_rows_quiet(clean_dev_rows)
    clean_test_metrics = score_rows_quiet(clean_test_rows)
    anchor_dev_metrics = score_rows_quiet(anchor_dev_rows)
    anchor_test_metrics = score_rows_quiet(anchor_test_rows)
    dev_metrics = score_rows_quiet(selected_dev)
    test_metrics = score_rows_quiet(selected_test)
    oracle_dev_metrics = score_rows_quiet(oracle_dev)
    oracle_test_metrics = score_rows_quiet(oracle_test)

    save_json(output_dir / "threshold_results.json", threshold_results)
    save_json(output_dir / "dev_oof_predictions.json", selected_metadata(selected_dev, dev_baseline, dev_anchors))
    save_json(output_dir / "test_predictions.json", selected_metadata(selected_test, baseline_by_split["test"], test_anchors))
    save_json(output_dir / "test_anchor_predictions.json", selected_metadata(anchor_test_rows, baseline_by_split["test"], test_anchors))
    save_json(output_dir / "test_clean_baseline_predictions.json", serialise(clean_test_rows, baseline_by_split["test"]))
    save_json(output_dir / "test_oracle_predictions.json", serialise(oracle_test, baseline_by_split["test"]))

    summary = {
        "method": "cross-proposal verified source switcher",
        "method_short_name": "CPV",
        "workspace_root": str(workspace),
        "refs_root": str(captions_root),
        "refs_filename_template": args.refs_filename_template,
        "adapted_run_dir": str(adapted_run),
        "selector_results_dir": str(selector_dir) if selector_dir else "",
        "proposals_json": str(proposals_path),
        "verifier_proposals_jsons": [str(path) for path in verifier_paths if path.exists()],
        "candidate_sources": args.candidate_sources,
        "max_blip_candidates": args.max_blip_candidates,
        "dev_splits": dev_splits,
        "thresholds": thresholds,
        "best_threshold": best_threshold,
        "threshold_results": threshold_results,
        "dev_clean_baseline_metrics": clean_dev_metrics,
        "dev_anchor_metrics": anchor_dev_metrics,
        "dev_metrics": dev_metrics,
        "dev_oracle_metrics": oracle_dev_metrics,
        "dev_cider_gain_vs_clean_baseline": dev_metrics["CIDEr"] - clean_dev_metrics["CIDEr"],
        "dev_cider_gain_vs_anchor": dev_metrics["CIDEr"] - anchor_dev_metrics["CIDEr"],
        "dev_source_counts": source_counts(selected_dev),
        "dev_anchor_source_counts": source_counts(anchor_dev_rows),
        "dev_clean_baseline_source_counts": source_counts(clean_dev_rows),
        "dev_source_switches": source_switch_rows(selected_dev, dev_anchors),
        "test_clean_baseline_metrics": clean_test_metrics,
        "test_anchor_metrics": anchor_test_metrics,
        "test_metrics": test_metrics,
        "test_oracle_metrics": oracle_test_metrics,
        "test_cider_gain_vs_clean_baseline": test_metrics["CIDEr"] - clean_test_metrics["CIDEr"],
        "test_cider_gain_vs_anchor": test_metrics["CIDEr"] - anchor_test_metrics["CIDEr"],
        "test_cider_gain_vs_current_reported_baseline": test_metrics["CIDEr"] - args.current_reported_cider,
        "test_source_counts": source_counts(selected_test),
        "test_anchor_source_counts": source_counts(anchor_test_rows),
        "test_clean_baseline_source_counts": source_counts(clean_test_rows),
        "test_oracle_source_counts": source_counts(oracle_test),
        "test_source_switches": source_switch_rows(selected_test, test_anchors),
        "test_switches_vs_clean_baseline": switch_rows(selected_test, baseline_by_split["test"]),
        "test_num_candidates": int(sum(len(rows) for rows in candidates_by_split["test"].values())),
        "dev_num_candidates": int(sum(len(rows) for rows in dev_candidates.values())),
        "interpretation_note": (
            "The threshold is selected on train/validation OOF anchors. Test "
            "references are used only for final scoring and oracle diagnostics."
        ),
    }
    save_json(output_dir / "summary.json", summary)
    write_markdown(output_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
