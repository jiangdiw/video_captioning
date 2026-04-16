# debug_vggish.py  (run from project root)
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

import traceback
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from src.data.split_videos_simple import get_splits

SAMPLE_RATE = 16000
N_FFT       = 400
HOP_LENGTH  = 160
N_MELS      = 64
NUM_FRAMES  = 96

WAV_ROOT    = Path("data/processed/audio/wav")
WEIGHTS_PATH = Path("data/vggish_weights.pth")

# ---- Test 1: check weights loaded correctly ----
print("=== TEST 1: Check weights file ===")
state = torch.load(str(WEIGHTS_PATH), map_location="cpu")
print("Keys in weights file:")
for k, v in list(state.items())[:10]:   # print first 10 keys
    print(f"  {k}: {v.shape}")

# ---- Test 2: try wav_to_input on ONE file ----
print("\n=== TEST 2: Try loading one wav file ===")
_, _, test_files = get_splits()

# find first wav that exists
test_wav = None
for f in test_files:
    candidate = WAV_ROOT / "test" / f"{f.stem}.wav"
    if candidate.exists():
        test_wav = candidate
        break

print(f"Testing with: {test_wav}")

try:
    waveform, sr = torchaudio.load(str(test_wav))
    print(f"  waveform shape: {waveform.shape}  sr: {sr}")

    if sr != SAMPLE_RATE:
        waveform = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=125,
        f_max=7500,
    )
    mel_spec = mel_transform(waveform)
    log_mel  = torch.log(mel_spec + 1e-6)
    log_mel  = log_mel.squeeze(0).T
    print(f"  log_mel shape: {log_mel.shape}")

    n_frames  = log_mel.shape[0]
    n_windows = n_frames // NUM_FRAMES
    print(f"  n_frames={n_frames}  n_windows={n_windows}")

    if n_windows == 0:
        pad = NUM_FRAMES - n_frames
        log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad))
        n_windows = 1

    windows = [log_mel[i * NUM_FRAMES: (i + 1) * NUM_FRAMES] for i in range(n_windows)]
    x = torch.stack(windows).unsqueeze(1)
    print(f"  input tensor shape: {x.shape}")   # should be (T, 1, 96, 64)

except Exception:
    traceback.print_exc()