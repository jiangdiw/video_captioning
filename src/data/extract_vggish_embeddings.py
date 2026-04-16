# src/data/extract_vggish_embeddings.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import soundfile as sf               # replaces torchaudio.load
import numpy as np
import urllib.request
import traceback
from src.data.split_videos_simple import get_splits

# -------------------
# PATHS
# -------------------
DATA_ROOT    = Path("data")
WAV_ROOT     = DATA_ROOT / "processed" / "audio" / "wav"
VGGISH_ROOT  = DATA_ROOT / "processed" / "audio" / "vggish_embeddings"
LOGS_DIR     = DATA_ROOT / "raw" / "downsampled_2500_videos" / "logs"
WEIGHTS_PATH = DATA_ROOT / "vggish_weights.pth"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# VGGish constants
# -------------------
SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 64
NUM_FRAMES  = 96

# -------------------
# VGGish architecture
# -------------------
class VGGish(nn.Module):
    """
    Input:  (T, 1, 96, 64)
    Output: (T, 128)
    """
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
            nn.Linear(4096, 4096),         nn.ReLU(),
            nn.Linear(4096, 128),
        )

    def forward(self, x):
        x = self.features(x)         # (T, 512, 6, 4)
        x = x.view(x.size(0), -1)    # (T, 12288)
        x = self.fc(x)               # (T, 128)
        return x

# -------------------
# Load pretrained weights
# -------------------
def load_model():
    print("Loading VGGish model...")
    model = VGGish()

    if not WEIGHTS_PATH.exists():
        print("  Downloading pretrained weights (one-time only)...")
        url = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
        urllib.request.urlretrieve(url, str(WEIGHTS_PATH))
        print(f"  Weights saved to {WEIGHTS_PATH}")

    state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    print("VGGish model ready.\n")
    return model

# -------------------
# wav → log-mel windows → (T, 1, 96, 64)
# uses soundfile instead of torchaudio.load
# -------------------
def wav_to_input(wav_path):
    # Step 1: load with soundfile → numpy array
    data, sr = sf.read(str(wav_path))          # data: (samples,) or (samples, channels)

    # Step 2: convert to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)               # (samples,)

    # Step 3: convert to torch tensor
    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, samples)

    # Step 4: resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform  = resampler(waveform)

    # Step 5: log-mel spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=125,
        f_max=7500,
    )
    mel_spec = mel_transform(waveform)          # (1, 64, time_frames)
    log_mel  = torch.log(mel_spec + 1e-6)      # log scale
    log_mel  = log_mel.squeeze(0).T            # (time_frames, 64)

    # Step 6: slice into 96-frame windows
    n_frames  = log_mel.shape[0]
    n_windows = n_frames // NUM_FRAMES

    if n_windows == 0:
        # pad short audio to exactly one window
        pad     = NUM_FRAMES - n_frames
        log_mel = F.pad(log_mel, (0, 0, 0, pad))
        n_windows = 1

    windows = [log_mel[i * NUM_FRAMES: (i + 1) * NUM_FRAMES]
               for i in range(n_windows)]

    x = torch.stack(windows)    # (T, 96, 64)
    x = x.unsqueeze(1)          # (T, 1, 96, 64)
    return x

# -------------------
# Extract embedding for ONE wav → numpy (T, 128)
# -------------------
def extract_one(model, wav_path):
    x = wav_to_input(wav_path)
    with torch.no_grad():
        emb = model(x)           # (T, 128)
    return emb.numpy().astype(np.float32)

# -------------------
# Process one split
# -------------------
def extract_vggish_for_split(model, files, split):
    out_dir = VGGISH_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    no_wav = []
    failed = []

    for video_path in files:
        vid_id   = video_path.stem
        out_npy  = out_dir / f"{vid_id}.npy"
        wav_path = WAV_ROOT / split / f"{vid_id}.wav"

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
        except Exception as e:
            print(f"  FAILED: {vid_id}")
            traceback.print_exc()          # full error so we can see what's wrong
            failed.append(vid_id)
            np.save(out_npy, np.zeros((1, 128), dtype=np.float32))

    if no_wav:
        log = LOGS_DIR / f"no_wav_{split}.txt"
        log.write_text("\n".join(no_wav))
        print(f"\n[{split}] {len(no_wav)} silent videos → {log}")

    if failed:
        log = LOGS_DIR / f"failed_vggish_{split}.txt"
        log.write_text("\n".join(failed))
        print(f"[{split}] {len(failed)} failed → {log}")

    ok = len(files) - len(no_wav) - len(failed)
    print(f"[{split}] Done ✓  ok={ok}  silent={len(no_wav)}  failed={len(failed)}\n")

# -------------------
# Main
# -------------------
def main():
    model = load_model()
    train_files, val_files, test_files = get_splits()
    extract_vggish_for_split(model, train_files, "train")
    extract_vggish_for_split(model, val_files,   "val")
    extract_vggish_for_split(model, test_files,  "test")

if __name__ == "__main__":
    main()