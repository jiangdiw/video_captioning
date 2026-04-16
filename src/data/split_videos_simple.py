# src/data/split_videos_simple.py
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SEED        = 42
N_VIDEOS    = 2500
N_TRAIN     = 2000
N_VAL       = 250
N_TEST      = 250
TEST_START  = 7010   # video7010 is the first test video

VIDEOS_DIR  = PROJECT_ROOT / "data" / "raw" / "downsampled_2500_videos"

# -------------------
# Helper: extract integer ID from filename
# "video7010.mp4" → 7010
# -------------------
def get_video_number(path):
    return int(path.stem.replace("video", ""))

# -------------------
# Main split function
# -------------------
def get_splits():
    all_videos = sorted(VIDEOS_DIR.glob("*.mp4"), key=get_video_number)
    print(f"Found {len(all_videos)} videos in {VIDEOS_DIR}")
    assert len(all_videos) == N_VIDEOS, \
        f"Expected {N_VIDEOS} videos, found {len(all_videos)}"

    # ── Test set: video7010 and above ──────────────────────────────
    test_files   = [v for v in all_videos if get_video_number(v) >= TEST_START]

    # ── Train + Val pool: video IDs below 7010 ─────────────────────
    trainval_files = [v for v in all_videos if get_video_number(v) < TEST_START]

    # Validate counts before proceeding
    assert len(test_files) == N_TEST, \
        f"Expected {N_TEST} test videos, got {len(test_files)}"
    assert len(trainval_files) == N_TRAIN + N_VAL, \
        f"Expected {N_TRAIN + N_VAL} train+val videos, got {len(trainval_files)}"

    # ── Shuffle then split train / val ─────────────────────────────
    random.seed(SEED)
    random.shuffle(trainval_files)

    train_files = trainval_files[:N_TRAIN]
    val_files   = trainval_files[N_TRAIN:]

    # Final checks
    assert len(train_files) == N_TRAIN, f"train size mismatch: {len(train_files)}"
    assert len(val_files)   == N_VAL,   f"val size mismatch: {len(val_files)}"
    assert len(test_files)  == N_TEST,  f"test size mismatch: {len(test_files)}"

    # Make sure there is zero overlap between splits
    train_ids = {v.stem for v in train_files}
    val_ids   = {v.stem for v in val_files}
    test_ids  = {v.stem for v in test_files}
    assert len(train_ids & val_ids)  == 0, "Overlap between train and val!"
    assert len(train_ids & test_ids) == 0, "Overlap between train and test!"
    assert len(val_ids   & test_ids) == 0, "Overlap between val and test!"

    print(f"train : {len(train_files)}")
    print(f"val   : {len(val_files)}")
    print(f"test  : {len(test_files)}  (video{TEST_START} and above)")
    print(f"No overlap between splits ✓")

    return train_files, val_files, test_files

if __name__ == "__main__":
    get_splits()