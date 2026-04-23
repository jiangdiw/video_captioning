# src/training/train_end_to_end.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.vocabulary           import Vocabulary
from src.data.dataset              import MSRVTTDataset, collate_fn
from src.models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from src.models.captioning_model   import CaptioningModel

# -------------------
# CONFIG
# -------------------
DEBUG      = False
DEBUG_N    = 10
N_EPOCHS   = 100
BATCH_SIZE = 16
LR         = 1e-4
HIDDEN_DIM = 512
EMBED_DIM  = 256
DROPOUT    = 0.5
EARLY_STOP = 15     # stop if val loss does not improve for 5 epochs

DATA_ROOT  = Path("data")
VOCAB_PATH = DATA_ROOT / "processed/captions/vocabulary.json"
LOG_DIR    = PROJECT_ROOT / "outputs/logs"
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
# Early stopping
# -------------------
class EarlyStopping:
    """
    Stops training if val loss does not improve for `patience` epochs.

    Example with patience=5:
      epoch 10: val=3.50  ← best, counter resets to 0
      epoch 11: val=3.55  ← no improvement (1/5)
      epoch 12: val=3.58  ← no improvement (2/5)
      epoch 13: val=3.60  ← no improvement (3/5)
      epoch 14: val=3.62  ← no improvement (4/5)
      epoch 15: val=3.65  ← no improvement (5/5) → STOP
    """
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            # Improved
            self.best_loss = val_loss
            self.counter   = 0
        else:
            # No improvement
            self.counter += 1
            print(f"    EarlyStopping: {self.counter}/{self.patience} "
                  f"(best={self.best_loss:.4f}  current={val_loss:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True

# -------------------
# Loss: cross entropy ignoring padding
# -------------------
def caption_loss(logits, captions, pad_idx):
    """
    logits   : (B, max_len-1, vocab_size)
    captions : (B, max_len)  [<sos>, w1, w2, ..., wN, <eos>]

    input  to LSTM = captions[:, :-1] = [<sos>, w1, ..., wN]
    target of loss = captions[:, 1:]  = [w1, ..., wN, <eos>]

    cross entropy: -log(p(correct_word)) averaged over all steps
    padding tokens are ignored
    """
    target  = captions[:, 1:]    # (B, max_len-1)
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * T, V),
        target.reshape(B * T),
        ignore_index=pad_idx
    )

# -------------------
# One epoch
# -------------------
def run_epoch(model, loader, optimizer, pad_idx, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        # _ ignores raw caption strings (only needed for display)
        for clips, dinos, audios, caps, _, _ in loader:
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
# Sample generations
# -------------------
def show_samples(model, loader, vocab, approach_name, n=3):
    model.eval()
    clips, dinos, audios, caps, raw_captions, vid_ids = next(iter(loader))
    clips  = clips[:n].to(DEVICE)
    dinos  = dinos[:n].to(DEVICE)
    audios = audios[:n].to(DEVICE)

    generated = model.generate(
        clips, dinos, audios,
        max_len = 20,
        sos_idx = vocab.sos_idx,
        eos_idx = vocab.eos_idx
    )

    print(f"\n  [{approach_name}] Sample generations:")
    for i in range(n):
        gt  = raw_captions[i]                      # original string, no <unk>
        gen = vocab.decode(generated[i].tolist())  # model output
        print(f"    [{vid_ids[i]}]")
        print(f"      GT  : {gt}")
        print(f"      GEN : {gen}")

# -------------------
# Train one approach
# -------------------
def train_approach(name, model, train_loader, val_loader, vocab):
    ckpt_dir = PROJECT_ROOT / "outputs/checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*55}")
    print(f"Training: {name}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  Max epochs    : {N_EPOCHS}")
    print(f"  Early stop    : {EARLY_STOP} epochs patience")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR            : {LR}")
    print(f"  Dropout       : {DROPOUT}")
    print(f"{'='*55}\n")

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS
    )
    early_stopper = EarlyStopping(patience=EARLY_STOP, min_delta=1e-4)

    best_val = float("inf")
    history  = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer,
                               vocab.pad_idx, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer,
                               vocab.pad_idx, train=False)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:03d}/{N_EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "history"     : history,
                "approach"    : name,
            }, ckpt_dir / "best.pth")
            print(f"    → Best saved (val={val_loss:.4f})")

        # Save latest every epoch (safe to resume)
        torch.save({
            "epoch"           : epoch,
            "model_state"     : model.state_dict(),
            "optimizer_state" : optimizer.state_dict(),
            "val_loss"        : val_loss,
            "history"         : history,
        }, ckpt_dir / "latest.pth")

        # Show samples every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            show_samples(model, val_loader, vocab, name)

        # Early stopping check
        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            print(f"  Val loss did not improve for {EARLY_STOP} epochs")
            break

    # Save history
    with open(LOG_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    actual_epochs = len(history["train"])
    print(f"\n[{name}]")
    print(f"  Ran       : {actual_epochs}/{N_EPOCHS} epochs")
    print(f"  Best val  : {best_val:.4f}")

    return history, best_val

# -------------------
# Main
# -------------------
def main():
    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab)}\n")

    debug_n = DEBUG_N if DEBUG else None

    # Datasets shared between both approaches
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
        dropout    = DROPOUT,
    ).to(DEVICE)

    history1, best_val1 = train_approach(
        "approach1_clip_dino_crossattn_audio_concat",
        model1, train_loader, val_loader, vocab
    )

    # ── Approach 2: Trimodal sequential cross-attn ────────────────
    model2 = CaptioningModel(
        encoder    = Approach2Encoder(),
        vocab_size = len(vocab),
        embed_dim  = EMBED_DIM,
        hidden_dim = HIDDEN_DIM,
        dropout    = DROPOUT,
    ).to(DEVICE)

    history2, best_val2 = train_approach(
        "approach2_trimodal_crossattn",
        model2, train_loader, val_loader, vocab
    )

    # ── Final comparison ──────────────────────────────────────────
    winner = "Approach 1" if best_val1 < best_val2 else "Approach 2"
    diff   = abs(best_val1 - best_val2)

    print(f"\n{'='*55}")
    print(f"FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"  Approach 1 (CLIP x DINOv2 + audio concat)")
    print(f"    Epochs run    : {len(history1['train'])}")
    print(f"    Best val loss : {best_val1:.4f}")
    print(f"  Approach 2 (Trimodal cross-attention)")
    print(f"    Epochs run    : {len(history2['train'])}")
    print(f"    Best val loss : {best_val2:.4f}")
    print(f"\n  Winner     : {winner}")
    print(f"  Difference : {diff:.4f}")

    # Save comparison summary
    summary = {
        "approach1" : {
            "name"          : "CLIP x DINOv2 cross-attn + audio concat",
            "epochs_run"    : len(history1["train"]),
            "best_val_loss" : best_val1,
            "history"       : history1,
        },
        "approach2" : {
            "name"          : "Trimodal sequential cross-attention",
            "epochs_run"    : len(history2["train"]),
            "best_val_loss" : best_val2,
            "history"       : history2,
        },
        "winner"     : winner,
        "difference" : diff,
    }
    with open(LOG_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {LOG_DIR}/comparison_summary.json")


if __name__ == "__main__":
    main()