import subprocess
import json
import sys
from pathlib import Path
from src.data.split_videos_simple import get_splits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("data")
WAV_ROOT = DATA_ROOT / "processed" / "audio" / "wav"
LOGS_DIR = DATA_ROOT / "raw" / "downsampled_2500_videos" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Check if a video has an audio track using ffprobe
# -------------------
def has_audio_stream(video_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a",   # only look at audio streams
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return len(info.get("streams", [])) > 0

# -------------------
# Extract audio for one split
# -------------------
def extract_audio_for_split(files, split):
    out_dir = WAV_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    no_audio = []      # videos skipped due to no audio track
    failed = []        # videos that failed for other reasons

    for video_path in files:
        vid_id = video_path.stem
        out_wav = out_dir / f"{vid_id}.wav"

        if out_wav.exists():
            print(f"  SKIP (already done): {vid_id}")
            continue

        # Check for audio stream first
        if not has_audio_stream(video_path):
            print(f"  SKIP (no audio track): {vid_id}")
            no_audio.append(vid_id)
            continue

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            str(out_wav)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  OK: {vid_id}")
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: {vid_id}")
            failed.append(vid_id)

    # Log skipped and failed videos
    if no_audio:
        log_path = LOGS_DIR / f"no_audio_{split}.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(no_audio))
        print(f"\n[{split}] {len(no_audio)} videos had no audio → {log_path}")

    if failed:
        log_path = LOGS_DIR / f"failed_audio_{split}.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(failed))
        print(f"[{split}] {len(failed)} videos failed → {log_path}")

    print(f"[{split}] Done: {len(files) - len(no_audio) - len(failed)} extracted, "
          f"{len(no_audio)} no audio, {len(failed)} failed\n")

# -------------------
# Main
# -------------------
def main():
    train_files, val_files, test_files = get_splits()
    extract_audio_for_split(train_files, "train")
    extract_audio_for_split(val_files,   "val")
    extract_audio_for_split(test_files,  "test")

if __name__ == "__main__":
    main()