import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.msrvtt import get_processed_layout, normalize_dataset_mode
from data.split_videos_simple import get_splits


def resolve_binary(env_var, names):
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        if path.exists():
            return str(path)

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    common_paths = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]
    for directory in common_paths:
        for name in names:
            candidate = Path(directory) / name
            if candidate.exists():
                return str(candidate)
    return None


def require_ffmpeg_tools():
    ffmpeg_bin = resolve_binary("FFMPEG_BINARY", ["ffmpeg"])
    ffprobe_bin = resolve_binary("FFPROBE_BINARY", ["ffprobe"])
    if not ffmpeg_bin or not ffprobe_bin:
        raise RuntimeError(
            "Missing ffmpeg/ffprobe. Install ffmpeg first, for example with "
            "`brew install ffmpeg`, or set FFMPEG_BINARY and FFPROBE_BINARY."
        )
    return ffmpeg_bin, ffprobe_bin


def has_audio_stream(video_path, ffprobe_bin):
    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "a",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout or "{}")
    return len(info.get("streams", [])) > 0


def extract_audio_for_split(files, split, wav_root, logs_dir, ffmpeg_bin, ffprobe_bin):
    out_dir = wav_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    no_audio = []
    failed = []

    for video_path in files:
        vid_id = video_path.stem
        out_wav = out_dir / f"{vid_id}.wav"
        if out_wav.exists():
            print(f"  SKIP (already done): {vid_id}")
            continue

        if not has_audio_stream(video_path, ffprobe_bin):
            print(f"  SKIP (no audio track): {vid_id}")
            no_audio.append(vid_id)
            continue

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_wav),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  OK: {vid_id}")
        except subprocess.CalledProcessError:
            print(f"  FAILED: {vid_id}")
            failed.append(vid_id)

    if no_audio:
        log_path = logs_dir / f"no_audio_{split}.txt"
        log_path.write_text("\n".join(no_audio))
        print(f"\n[{split}] {len(no_audio)} videos had no audio → {log_path}")

    if failed:
        log_path = logs_dir / f"failed_audio_{split}.txt"
        log_path.write_text("\n".join(failed))
        print(f"[{split}] {len(failed)} videos failed → {log_path}")

    print(f"[{split}] Done: {len(files) - len(no_audio) - len(failed)} extracted, {len(no_audio)} no audio, {len(failed)} failed\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-video WAV audio for MSR-VTT.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--video-dir", action="append", default=[], help="Optional additional directory to search for video*.mp4 files.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some expected videos are not present locally.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    wav_root = layout.audio_root / "wav"
    logs_dir = layout.audio_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin, ffprobe_bin = require_ffmpeg_tools()

    train_files, val_files, test_files = get_splits(
        dataset_mode=dataset_mode,
        strict=not args.allow_missing,
        extra_video_dirs=args.video_dir,
    )
    print(f"dataset_mode: {dataset_mode}")
    print(f"ffmpeg: {ffmpeg_bin}")
    print(f"ffprobe: {ffprobe_bin}")
    extract_audio_for_split(train_files, "train", wav_root, logs_dir, ffmpeg_bin, ffprobe_bin)
    extract_audio_for_split(val_files, "val", wav_root, logs_dir, ffmpeg_bin, ffprobe_bin)
    extract_audio_for_split(test_files, "test", wav_root, logs_dir, ffmpeg_bin, ffprobe_bin)


if __name__ == "__main__":
    main()
