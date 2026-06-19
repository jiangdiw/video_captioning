from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the experimental MSR-VTT generated-beam selector pipeline. "
            "The pipeline can optionally run a fresh captioner command, then "
            "generates full MSR-VTT beam caches and trains the Dattalion selector."
        )
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-captioner-command", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--captions-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--fused-visual-roots", required=True)
    parser.add_argument("--beam-cache-dir", required=True)
    parser.add_argument("--selector-output-dir", required=True)
    parser.add_argument("--refs-root", required=True)
    parser.add_argument("--refs-filename-template", default="{split}_captions_normalized.json")
    parser.add_argument("--msrvtt-beam-splits", default="train,val")
    parser.add_argument("--selector-weight", type=float, default=0.001)
    parser.add_argument("--selector-learner", choices=["ridge", "pairwise", "extra_trees"], default="ridge")
    parser.add_argument("--beam-splits", default="train,val")
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--num-return-sequences", type=int, default=8)
    parser.add_argument("--max-caption-len", type=int, default=20)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--skip-existing-beams", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def beam_filename(split: str, num_beams: int, num_return_sequences: int, no_repeat_ngram_size: int, max_caption_len: int) -> str:
    return (
        f"beam_headroom_{split}_b{num_beams}_nr{num_return_sequences}"
        f"_lp1_nr{no_repeat_ngram_size}_cap{max_caption_len}.json"
    )


def main() -> None:
    args = parse_args()
    beam_cache_dir = Path(args.beam_cache_dir).expanduser().resolve()
    selector_output_dir = Path(args.selector_output_dir).expanduser().resolve()
    beam_cache_dir.mkdir(parents=True, exist_ok=True)
    selector_output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_captioner_command.strip():
        run(["/bin/zsh", "-lc", args.train_captioner_command], args.dry_run)

    for split in [item.strip() for item in args.beam_splits.split(",") if item.strip()]:
        output_json = beam_cache_dir / beam_filename(
            split=split,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_caption_len=args.max_caption_len,
        )
        if args.skip_existing_beams and output_json.exists():
            print(f"Skipping existing beam cache: {output_json}", flush=True)
            continue
        run(
            [
                args.python,
                str(SCRIPT_ROOT / "beam_headroom_diagnostic.py"),
                "--checkpoint",
                str(Path(args.checkpoint).expanduser().resolve()),
                "--captions-root",
                str(Path(args.captions_root).expanduser().resolve()),
                "--audio-root",
                str(Path(args.audio_root).expanduser().resolve()),
                "--fused-visual-roots",
                args.fused_visual_roots,
                "--output-json",
                str(output_json),
                "--split",
                split,
                "--num-beams",
                str(args.num_beams),
                "--num-return-sequences",
                str(args.num_return_sequences),
                "--max-caption-len",
                str(args.max_caption_len),
                "--no-repeat-ngram-size",
                str(args.no_repeat_ngram_size),
                "--batch-size",
                str(args.batch_size),
                "--progress-every",
                str(args.progress_every),
                "--device",
                args.device,
            ],
            args.dry_run,
        )

    run(
        [
            args.python,
            str(SCRIPT_ROOT / "train_frozen_learned_selector.py"),
            "--output-dir",
            str(selector_output_dir),
            "--refs-root",
            str(Path(args.refs_root).expanduser().resolve()),
            "--refs-filename-template",
            args.refs_filename_template,
            "--msrvtt-beam-run-dir",
            str(beam_cache_dir),
            "--msrvtt-beam-splits",
            args.msrvtt_beam_splits,
            "--pretrain-weight",
            str(args.selector_weight),
            "--learner",
            args.selector_learner,
        ],
        args.dry_run,
    )


if __name__ == "__main__":
    main()
