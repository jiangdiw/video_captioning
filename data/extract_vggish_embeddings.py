import argparse
import sys
import traceback
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from data.msrvtt import get_processed_layout, normalize_dataset_mode
from data.split_videos_simple import get_splits


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64
NUM_FRAMES = 96


class VGGish(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 6 * 4, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 128),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model(weights_path):
    print("Loading VGGish model...")
    model = VGGish()
    if not weights_path.exists():
        print("  Downloading pretrained weights...")
        url = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"  Weights saved to {weights_path}")

    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    print("VGGish model ready.\n")
    return model


def wav_to_input(wav_path):
    data, sr = sf.read(str(wav_path))
    if data.ndim == 2:
        data = data.mean(axis=1)

    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=125,
        f_max=7500,
    )
    mel_spec = mel_transform(waveform)
    log_mel = torch.log(mel_spec + 1e-6).squeeze(0).T

    n_frames = log_mel.shape[0]
    n_windows = n_frames // NUM_FRAMES
    if n_windows == 0:
        log_mel = F.pad(log_mel, (0, 0, 0, NUM_FRAMES - n_frames))
        n_windows = 1

    windows = [log_mel[i * NUM_FRAMES : (i + 1) * NUM_FRAMES] for i in range(n_windows)]
    return torch.stack(windows).unsqueeze(1)


def extract_one(model, wav_path):
    x = wav_to_input(wav_path)
    with torch.no_grad():
        emb = model(x)
    return emb.numpy().astype(np.float32)


def extract_vggish_for_split(model, files, split, wav_root, out_root, logs_dir):
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    no_wav = []
    failed = []

    for video_path in files:
        vid_id = video_path.stem
        out_npy = out_dir / f"{vid_id}.npy"
        wav_path = wav_root / split / f"{vid_id}.wav"

        if out_npy.exists():
            print(f"  SKIP (exists): {vid_id}")
            continue

        if not wav_path.exists():
            print(f"  SKIP (no wav → zero placeholder): {vid_id}")
            no_wav.append(vid_id)
            np.save(out_npy, np.zeros((1, 128), dtype=np.float32))
            continue

        try:
            emb = extract_one(model, wav_path)
            np.save(out_npy, emb)
            print(f"  OK: {vid_id}  shape={emb.shape}")
        except Exception:
            print(f"  FAILED: {vid_id}")
            traceback.print_exc()
            failed.append(vid_id)
            np.save(out_npy, np.zeros((1, 128), dtype=np.float32))

    if no_wav:
        log = logs_dir / f"no_wav_{split}.txt"
        log.write_text("\n".join(no_wav))
        print(f"\n[{split}] {len(no_wav)} silent videos → {log}")

    if failed:
        log = logs_dir / f"failed_vggish_{split}.txt"
        log.write_text("\n".join(failed))
        print(f"[{split}] {len(failed)} failed → {log}")

    ok = len(files) - len(no_wav) - len(failed)
    print(f"[{split}] Done ✓  ok={ok}  silent={len(no_wav)}  failed={len(failed)}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VGGish audio embeddings for MSR-VTT.")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--video-dir", action="append", default=[], help="Optional additional directory to search for video*.mp4 files.")
    parser.add_argument("--allow-missing", action="store_true", help="Do not fail if some expected videos are not present locally.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    wav_root = layout.audio_root / "wav"
    out_root = layout.raw_audio_root
    logs_dir = layout.audio_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_path = layout.processed_root / "vggish_weights.pth"

    model = load_model(weights_path)
    train_files, val_files, test_files = get_splits(
        dataset_mode=dataset_mode,
        strict=not args.allow_missing,
        extra_video_dirs=args.video_dir,
    )
    print(f"dataset_mode: {dataset_mode}")
    extract_vggish_for_split(model, train_files, "train", wav_root, out_root, logs_dir)
    extract_vggish_for_split(model, val_files, "val", wav_root, out_root, logs_dir)
    extract_vggish_for_split(model, test_files, "test", wav_root, out_root, logs_dir)


if __name__ == "__main__":
    main()
