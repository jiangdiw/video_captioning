# src/training/train_captioning.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.vocabulary       import Vocabulary
from src.data.dataset          import MSRVTTDataset, collate_fn
from src.models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from src.models.captioning_model   import CaptioningModel

# -------------------
# CONFIG
# -------------------
DEBUG       = False       # True = 10 videos only to verify pipeline
DEBUG_N     = 10         # number of videos in debug mode
N_EPOCHS    = 30         # increase when running full training
BATCH_SIZE  = 8 if DEBUG else 32
LR          = 1e-3
HIDDEN_DIM  = 512
EMBED_DIM   = 256
NUM_LAYERS  = 2
DROPOUT     = 0.3

DATA_ROOT   = Path("data")
VOCAB_PATH  = DATA_ROOT / "processed/captions/vocabulary.json"
CKPT_DIR    = PROJECT_ROOT / "outputs/checkpoints"
LOG_DIR     = PROJECT_ROOT / "outputs/logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Device
# -------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}\n")

# -------------------
# Loss: cross entropy ignoring padding
# -------------------
def caption_loss(logits, captions, pad_idx):
    """
    logits   : (B, max_len-1, vocab_size)
    captions : (B, max_len)
    target   : captions[:, 1:]  = words after SOS
    """
    target = captions[:, 1:]                       # (B, max_len-1)
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        target.reshape(B * T),
        ignore_index=pad_idx
    )
    return loss

# -------------------
# One epoch
# -------------------
def run_epoch(model, loader, optimizer, pad_idx, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for clips, dinos, audios, caps, lengths, _ in loader:
            clips  = clips.to(DEVICE)
            dinos  = dinos.to(DEVICE)
            audios = audios.to(DEVICE)
            caps   = caps.to(DEVICE)

            logits = model(clips, dinos, audios, caps)
            loss   = caption_loss(logits, caps, pad_idx)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches

# -------------------
# Sample generated captions
# -------------------
def show_samples(model, loader, vocab, n=3):
    model.eval()
    clips, dinos, audios, caps, _, vid_ids = next(iter(loader))
    clips  = clips[:n].to(DEVICE)
    dinos  = dinos[:n].to(DEVICE)
    audios = audios[:n].to(DEVICE)
    caps   = caps[:n]

    generated = model.generate(clips, dinos, audios,
                                max_len=20,
                                sos_idx=vocab.sos_idx,
                                eos_idx=vocab.eos_idx)

    print("\n  Sample generations:")
    for i in range(n):
        gt  = vocab.decode(caps[i].tolist())
        gen = vocab.decode(generated[i].tolist())
        print(f"    [{vid_ids[i]}]")
        print(f"      GT  : {gt}")
        print(f"      GEN : {gen}")

# -------------------
# Train one approach
# -------------------
def train_approach(name, model, train_loader, val_loader, vocab):
    print(f"\n{'='*55}")
    print(f"Training: {name}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs     : {N_EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Debug mode : {DEBUG}")
    print(f"{'='*55}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )

    ckpt_dir = CKPT_DIR / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val  = float("inf")
    history   = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer,
                               vocab.pad_idx, train=True)
        val_loss   = run_epoch(model, val_loader, optimizer,
                               vocab.pad_idx, train=False)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:02d}/{N_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "history"     : history,
            }, ckpt_dir / "best.pth")
            print(f"    → Best saved (val={val_loss:.4f})")

    # Save training history
    with open(LOG_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Show sample outputs
    show_samples(model, val_loader, vocab)

    return history

# -------------------
# Main
# -------------------
def main():
    # Load vocabulary
    vocab = Vocabulary.load(VOCAB_PATH)

    # Datasets
    debug_n = DEBUG_N if DEBUG else None

    train_dataset = MSRVTTDataset("train", vocab, debug_n=debug_n)
    val_dataset   = MSRVTTDataset("val",   vocab, debug_n=debug_n)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True,  collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ── Approach 1: CLIP x DINOv2 cross-attn + audio concat ──────
    model1 = CaptioningModel(
        encoder    = Approach1Encoder(),
        vocab_size = len(vocab),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    history1 = train_approach(
        "approach1_clip_dino_crossattn_audio_concat",
        model1, train_loader, val_loader, vocab
    )

    # ── Approach 2: Trimodal cross-attn ───────────────────────────
    model2 = CaptioningModel(
        encoder    = Approach2Encoder(),
        vocab_size = len(vocab),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        num_layers = NUM_LAYERS,
        dropout    = DROPOUT,
    ).to(DEVICE)

    history2 = train_approach(
        "approach2_trimodal_crossattn",
        model2, train_loader, val_loader, vocab
    )

    # ── Final comparison ──────────────────────────────────────────
    best1 = min(history1["val"])
    best2 = min(history2["val"])

    print(f"\n{'='*55}")
    print(f"FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"  Approach 1 best val loss : {best1:.4f}")
    print(f"  Approach 2 best val loss : {best2:.4f}")
    print(f"  Winner: {'Approach 1' if best1 < best2 else 'Approach 2'}")
    print(f"\n  Full history saved to outputs/logs/")

if __name__ == "__main__":
    main()