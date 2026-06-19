from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer


VIDEO_CAPTIONING_ROOT = Path("/Users/aglooney03/video_captioning")
OPENJDK_BIN = Path("/opt/homebrew/opt/openjdk/bin")
if OPENJDK_BIN.exists():
    os.environ["PATH"] = f"{OPENJDK_BIN}:{os.environ.get('PATH', '')}"
if str(VIDEO_CAPTIONING_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_CAPTIONING_ROOT))
COCO_ROOT = VIDEO_CAPTIONING_ROOT / "coco-caption"
if str(COCO_ROOT) not in sys.path:
    sys.path.insert(0, str(COCO_ROOT))

from misc.cocoeval import COCOScorer  # type: ignore
from pycocoevalcap.cider.cider_scorer import CiderScorer, cook_refs, cook_test  # type: ignore
from train_final_bart import (  # type: ignore
    FinalBartDataset,
    build_model,
    choose_device,
    decode_sequences,
    load_model_weights,
    make_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure beam-search oracle headroom on Dattalion.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--captions-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--fused-visual-roots", required=True, help="Comma-separated list of fused feature roots.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--split-id-key", default=None, help="Key to read from splits.json; defaults to --split.")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap on videos after split-id resolution; useful for candidate-cache smoke tests.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="If set with --max-videos, sample that many split ids deterministically instead of taking the prefix.",
    )
    parser.add_argument("--modalities", choices=["clip", "clip_dino", "clip_dino_audio"], default="clip_dino_audio")
    parser.add_argument("--architecture", choices=["stable", "experimental"], default="stable")
    parser.add_argument("--bart-model-name", default="facebook/bart-base")
    parser.add_argument("--decoder-train-mode", choices=["freeze", "full", "partial"], default="freeze")
    parser.add_argument("--encoder-d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--max-audio-tokens", type=int, default=None)
    parser.add_argument("--max-caption-len", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--num-return-sequences", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--diversity-penalty", type=float, default=0.0)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_payload_config(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "config" in payload:
        return payload["config"]
    return {}


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


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


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    captions_root = Path(args.captions_root).expanduser().resolve()
    audio_root = Path(args.audio_root).expanduser().resolve()
    fused_roots = [Path(item.strip()).expanduser().resolve() for item in args.fused_visual_roots.split(",") if item.strip()]
    output_json = Path(args.output_json).expanduser().resolve()

    checkpoint_config = load_payload_config(checkpoint_path)
    modalities = checkpoint_config.get("modalities", args.modalities)
    architecture = checkpoint_config.get("architecture", args.architecture)
    bart_model_name = checkpoint_config.get("bart_model_name", args.bart_model_name)
    decoder_train_mode = checkpoint_config.get("decoder_train_mode", args.decoder_train_mode)
    encoder_d_model = int(checkpoint_config.get("encoder_d_model", args.encoder_d_model))
    n_heads = int(checkpoint_config.get("n_heads", args.n_heads))
    gradient_checkpointing = bool(
        checkpoint_config.get("gradient_checkpointing", not args.disable_gradient_checkpointing)
    )

    class ArgsProxy:
        pass

    proxy = ArgsProxy()
    proxy.architecture = architecture
    proxy.decoder_train_mode = decoder_train_mode
    proxy.encoder_d_model = encoder_d_model
    proxy.n_heads = n_heads
    proxy.bart_model_name = bart_model_name
    proxy.disable_gradient_checkpointing = not gradient_checkpointing
    proxy.decoder_train_last_n_layers = int(checkpoint_config.get("decoder_train_last_n_layers", 2))
    proxy.max_visual_positions = int(checkpoint_config.get("max_visual_positions", 40))
    proxy.audio_summary_tokens = int(checkpoint_config.get("audio_summary_tokens", 4))
    proxy.feature_dropout = float(checkpoint_config.get("feature_dropout", 0.1))
    proxy.max_audio_tokens = (
        args.max_audio_tokens
        if args.max_audio_tokens is not None
        else checkpoint_config.get("max_audio_tokens")
    )
    proxy.transfer_adapter = checkpoint_config.get("transfer_adapter", "none")
    proxy.transfer_num_anchors = int(checkpoint_config.get("transfer_num_anchors", 0))
    proxy.transfer_num_tokens = int(checkpoint_config.get("transfer_num_tokens", 4))
    proxy.transfer_bottleneck_dim = int(checkpoint_config.get("transfer_bottleneck_dim", 64))
    proxy.transfer_dropout = float(checkpoint_config.get("transfer_dropout", 0.1))
    proxy.transfer_gate_init = float(checkpoint_config.get("transfer_gate_init", -4.0))
    proxy.transfer_residual_scale = float(checkpoint_config.get("transfer_residual_scale", 1.0))
    proxy.transfer_anchor_temperature = float(checkpoint_config.get("transfer_anchor_temperature", 0.08))

    use_dino = modalities in {"clip_dino", "clip_dino_audio"}
    use_audio = modalities == "clip_dino_audio"
    device = choose_device(args.device)

    tokenizer = BartTokenizer.from_pretrained(bart_model_name)
    model = build_model(proxy, use_dino=use_dino, use_audio=use_audio, device=device)
    load_model_weights(model, checkpoint_path, device)
    model.eval()

    split_id_key = args.split_id_key or args.split
    splits_path = captions_root.parent / "splits.json"
    if splits_path.exists():
        split_payload = json.loads(splits_path.read_text())
        split_ids = split_payload[split_id_key]
    else:
        split_caption_map = json.loads((captions_root / f"{args.split}_captions.json").read_text())
        split_ids = sorted(split_caption_map)
    if args.max_videos is not None:
        if args.sample_seed is not None and len(split_ids) > args.max_videos:
            rng = np.random.default_rng(args.sample_seed)
            split_ids = sorted(rng.choice(split_ids, size=args.max_videos, replace=False).tolist())
        else:
            split_ids = split_ids[: args.max_videos]
    dataset = FinalBartDataset(
        split=args.split,
        split_ids=split_ids,
        captions_root=captions_root,
        visual_root=Path("/does/not/matter"),
        audio_root=audio_root,
        visual_source="fused",
        fused_visual_roots=fused_roots,
        use_dino=use_dino,
        use_audio=use_audio,
        debug_n=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_caption_len),
        num_workers=0,
    )
    split_cider = SplitCiderScorer(dataset.ground_truth)

    oracle_predictions: dict[str, list[dict[str, str]]] = {}
    top1_predictions: dict[str, list[dict[str, str]]] = {}
    per_video = []

    for clips, dinos, audios, audio_mask, _, _, references_batch, video_ids, _ in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)

        generated = model.generate(
            clips,
            dinos,
            audios,
            audio_mask=audio_mask,
            max_new_tokens=args.max_caption_len,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            diversity_penalty=args.diversity_penalty,
            num_beam_groups=args.num_beam_groups,
            return_dict_in_generate=True,
            output_scores=True,
        )
        candidate_ids = generated.sequences
        candidates = decode_sequences(tokenizer, candidate_ids)
        sequence_scores = generated.sequences_scores.detach().cpu().tolist()

        for batch_index, (video_id, references) in enumerate(zip(video_ids, references_batch)):
            start = batch_index * args.num_return_sequences
            end = start + args.num_return_sequences
            video_candidates = candidates[start:end]
            video_scores = sequence_scores[start:end]

            scored_candidates = []
            for caption, model_score in zip(video_candidates, video_scores):
                cider = split_cider.score(references, caption)
                scored_candidates.append(
                    {
                        "caption": caption,
                        "model_score": float(model_score),
                        "cider": float(cider),
                    }
                )
            scored_candidates.sort(key=lambda row: row["cider"], reverse=True)
            best_oracle = scored_candidates[0]
            top1 = {
                "caption": video_candidates[0],
                "model_score": float(video_scores[0]),
                "cider": float(split_cider.score(references, video_candidates[0])),
            }

            top1_predictions[video_id] = [{"image_id": video_id, "caption": top1["caption"]}]
            oracle_predictions[video_id] = [{"image_id": video_id, "caption": best_oracle["caption"]}]
            per_video.append(
                {
                    "video_id": video_id,
                    "references": references,
                    "top1": top1,
                    "oracle": best_oracle,
                    "all_candidates": scored_candidates,
                }
            )
            if args.progress_every > 0 and len(per_video) % args.progress_every == 0:
                print(f"[{args.split}] processed {len(per_video)} / {len(dataset)} videos", flush=True)

    ids = [row["video_id"] for row in per_video]
    ground_truth = {video_id: dataset.ground_truth[video_id] for video_id in ids}
    scorer = COCOScorer()
    top1_metrics = scorer.score(ground_truth, top1_predictions, ids)
    scorer = COCOScorer()
    oracle_metrics = scorer.score(ground_truth, oracle_predictions, ids)

    payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "decode": {
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "diversity_penalty": args.diversity_penalty,
            "num_beam_groups": args.num_beam_groups,
            "max_audio_tokens": proxy.max_audio_tokens,
        },
        "top1_metrics": top1_metrics,
        "oracle_metrics": oracle_metrics,
        "mean_top1_sent_cider": float(np.mean([row["top1"]["cider"] for row in per_video])),
        "mean_oracle_sent_cider": float(np.mean([row["oracle"]["cider"] for row in per_video])),
        "per_video": per_video,
    }
    save_json(output_json, payload)
    print(json.dumps(payload["top1_metrics"], indent=2))
    print(json.dumps(payload["oracle_metrics"], indent=2))


if __name__ == "__main__":
    main()
