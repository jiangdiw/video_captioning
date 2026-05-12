import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train_final_bart.py"

SWEEP_CONFIGS = {
    "continue_v1": {
        "preset": "stable_continue_v1",
        "run_dir": "outputs/final_bart_full_tuned_continue_v1",
    },
    "continue_v2": {
        "preset": "stable_continue_v2",
        "run_dir": "outputs/final_bart_full_tuned_continue_v2",
    },
    "long_v1": {
        "preset": "stable_long_v1",
        "run_dir": "outputs/final_bart_full_tuned_long_v1",
    },
    "continue_v3": {
        "preset": "stable_continue_v3",
        "run_dir": "outputs/final_bart_full_tuned_continue_v3",
    },
    "long_v2": {
        "preset": "stable_long_v2",
        "run_dir": "outputs/final_bart_full_tuned_long_v2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tuned stable BART configurations without SCST.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=list(SWEEP_CONFIGS.keys()),
        default=["continue_v1", "continue_v2", "long_v1"],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def maybe_metrics_path(run_dir: Path) -> Path:
    return run_dir / "test_metrics_xe.json"


def main() -> int:
    args = parse_args()
    results = []

    for name in args.experiments:
        config = SWEEP_CONFIGS[name]
        run_dir = PROJECT_ROOT / config["run_dir"]
        metrics_path = maybe_metrics_path(run_dir)
        if args.skip_existing and metrics_path.exists():
            print(f"[skip] {name}: metrics already exist at {metrics_path}")
            try:
                results.append({"experiment": name, "run_dir": str(run_dir), **json.loads(metrics_path.read_text())})
            except Exception:
                pass
            continue

        cmd = [
            args.python,
            str(TRAIN_SCRIPT),
            "--preset",
            config["preset"],
            "--dataset-mode",
            "full",
            "--modalities",
            "clip_dino_audio",
            "--visual-source",
            "auto",
            "--skip-scst",
            "--device",
            args.device,
            "--run-dir",
            str(run_dir),
        ]
        print(f"[run] {name}: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

        if metrics_path.exists():
            results.append({"experiment": name, "run_dir": str(run_dir), **json.loads(metrics_path.read_text())})

    if results:
        results = sorted(results, key=lambda item: item.get("CIDEr", float("-inf")), reverse=True)
        summary_path = PROJECT_ROOT / "outputs" / "final_bart_tuning_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(json.dumps(results, indent=2))
        print(f"[saved] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
