# src/training/check_attention.py
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import json
from torch.utils.data import DataLoader

from src.data.vocabulary           import Vocabulary
from src.data.dataset              import MSRVTTDataset, collate_fn
from src.models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from src.models.captioning_model   import CaptioningModel

# -------------------
# CONFIG
# -------------------
N_SAMPLES  = 5     # how many videos to check attention for
N_WORDS    = 10    # how many word steps to show per video

DATA_ROOT  = Path("data")
VOCAB_PATH = DATA_ROOT / "processed/captions/vocabulary.json"

CKPT_PATHS = {
    "approach1" : PROJECT_ROOT / "outputs/checkpoints"
                  / "approach1_clip_dino_crossattn_audio_concat"
                  / "best.pth",
    "approach2" : PROJECT_ROOT / "outputs/checkpoints"
                  / "approach2_trimodal_crossattn"
                  / "best.pth",
}

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
# Load model from checkpoint
# -------------------
def load_model(approach, vocab):
    ckpt_path = CKPT_PATHS[approach]

    if not ckpt_path.exists():
        print(f"  ERROR: checkpoint not found at {ckpt_path}")
        print(f"  Run train_end_to_end.py first to generate checkpoints.")
        return None, None

    if approach == "approach1":
        encoder = Approach1Encoder()
    else:
        encoder = Approach2Encoder()

    model = CaptioningModel(
        encoder    = encoder,
        vocab_size = len(vocab),
        embed_dim  = 256,
        hidden_dim = 512,
        dropout    = 0.5,
    ).to(DEVICE)

    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded {approach} checkpoint")
    print(f"    Saved at epoch : {ckpt['epoch']}")
    print(f"    Val loss       : {ckpt['val_loss']:.4f}")

    return model, ckpt


# -------------------
# Check attention weights for one video
# -------------------
def check_one_video(model, clip, dino, audio,
                    raw_caption, vid_id, vocab):
    """
    Runs the decoder step by step and records attention
    weights at each word generation step.

    Returns a list of dicts, one per word step:
      {
        word       : predicted word string
        attn_min   : minimum attention weight across 40 frames
        attn_max   : maximum attention weight across 40 frames
        attn_std   : std of attention weights (key metric)
                     ~0.001 = collapsed (uniform)
                     >0.02  = working (focused)
        top_frame  : which frame got highest attention
        weights    : full (40,) numpy array
      }
    """
    clip  = clip.unsqueeze(0).to(DEVICE)    # (1, 40, 512)
    dino  = dino.unsqueeze(0).to(DEVICE)    # (1, 40, 768)
    audio = audio.unsqueeze(0).to(DEVICE)   # (1,  T, 128)

    with torch.no_grad():
        # Encode
        seq, pooled = model.encoder(clip, dino, audio)
        init_h      = torch.tanh(model.init_proj(pooled))

        # Initialize LSTM states
        h = init_h
        c = torch.tanh(model.decoder.init_c(init_h))

        # Start with <sos>
        token = torch.tensor(
            [vocab.sos_idx], dtype=torch.long, device=DEVICE)

        steps = []
        for step in range(N_WORDS):
            embed              = model.decoder.embedding(token)
            context, attn_w    = model.decoder.attention(h, seq)
            lstm_in            = torch.cat([embed, context], dim=-1)
            h, c               = model.decoder.lstm_cell(lstm_in, (h, c))
            logit              = model.decoder.fc_out(h)
            token              = logit.argmax(dim=-1)

            word    = vocab.idx2word.get(token.item(), "<unk>")
            weights = attn_w.squeeze(0).cpu().numpy()   # (40,)

            steps.append({
                "step"      : step + 1,
                "word"      : word,
                "attn_min"  : float(weights.min()),
                "attn_max"  : float(weights.max()),
                "attn_std"  : float(weights.std()),
                "top_frame" : int(weights.argmax()),
                "weights"   : weights,
            })

            if word == "<eos>":
                break

    return steps


# -------------------
# Interpret attention std
# -------------------
def interpret_std(std):
    if std < 0.005:
        return "COLLAPSED  ← uniform, not attending"
    elif std < 0.015:
        return "WEAK       ← mild focus"
    elif std < 0.030:
        return "MODERATE   ← some focus"
    else:
        return "STRONG     ✓ clearly attending to specific frames"


# -------------------
# Print attention report for one video
# -------------------
def print_attention_report(steps, raw_caption, vid_id):
    gen_caption = " ".join(
        s["word"] for s in steps
        if s["word"] not in ["<sos>", "<eos>", "<pad>"]
    )

    print(f"\n  Video   : {vid_id}")
    print(f"  GT      : {raw_caption}")
    print(f"  GEN     : {gen_caption}")
    print(f"  {'─'*70}")
    print(f"  {'Step':>4}  {'Word':<15}  {'Min':>6}  {'Max':>6}  "
          f"{'Std':>6}  {'TopFrame':>9}  Status")
    print(f"  {'─'*70}")

    for s in steps:
        status = interpret_std(s["attn_std"])
        print(f"  {s['step']:>4}  {s['word']:<15}  "
              f"{s['attn_min']:>6.3f}  "
              f"{s['attn_max']:>6.3f}  "
              f"{s['attn_std']:>6.3f}  "
              f"{s['top_frame']:>9}  "
              f"{status}")


# -------------------
# Run full attention check for one approach
# -------------------
def run_attention_check(approach, model, loader, vocab):
    print(f"\n{'='*70}")
    print(f"ATTENTION CHECK: {approach}")
    print(f"{'='*70}")

    all_stds    = []
    all_results = []

    clips, dinos, audios, caps, raw_captions, vid_ids = next(iter(loader))

    for i in range(min(N_SAMPLES, len(vid_ids))):
        steps = check_one_video(
            model,
            clips[i], dinos[i], audios[i],
            raw_captions[i], vid_ids[i],
            vocab
        )

        print_attention_report(steps, raw_captions[i], vid_ids[i])

        stds = [s["attn_std"] for s in steps]
        all_stds.extend(stds)

        all_results.append({
            "vid_id"      : vid_ids[i],
            "gt"          : raw_captions[i],
            "gen"         : " ".join(
                s["word"] for s in steps
                if s["word"] not in ["<sos>", "<eos>", "<pad>"]
            ),
            "mean_std"    : float(np.mean(stds)),
            "steps"       : steps,
        })

    # -------------------
    # Summary statistics
    # -------------------
    overall_std = float(np.mean(all_stds))

    print(f"\n  {'─'*70}")
    print(f"  SUMMARY for {approach}")
    print(f"  {'─'*70}")
    print(f"  Average attention std across all steps : {overall_std:.4f}")
    print(f"  Overall status : {interpret_std(overall_std)}")
    print()

    if overall_std < 0.005:
        print(f"  DIAGNOSIS: Attention is COLLAPSED")
        print(f"  The decoder is ignoring video content entirely.")
        print(f"  Recommended fixes:")
        print(f"    1. Add scheduled sampling (teacher_forcing_ratio decay)")
        print(f"    2. Add label smoothing (smoothing=0.1)")
        print(f"    3. Train for more epochs with patience=15")
        print(f"    4. Lower learning rate to 5e-5")
    elif overall_std < 0.015:
        print(f"  DIAGNOSIS: Attention is WEAK but present")
        print(f"  The decoder is partially using video content.")
        print(f"  Recommended fixes:")
        print(f"    1. Add scheduled sampling to reduce exposure bias")
        print(f"    2. Train for more epochs")
    else:
        print(f"  DIAGNOSIS: Attention is WORKING")
        print(f"  The decoder is actively focusing on specific frames.")
        print(f"  If captions are still poor, the bottleneck is")
        print(f"  likely the decoder capacity or dataset size.")

    return all_results, overall_std


# -------------------
# Save results to JSON
# -------------------
def save_results(results_dict):
    out_path = PROJECT_ROOT / "outputs/logs/attention_check_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove numpy arrays before saving (not JSON serializable)
    clean = {}
    for approach, (results, overall_std) in results_dict.items():
        clean[approach] = {
            "overall_std"    : overall_std,
            "overall_status" : interpret_std(overall_std),
            "videos"         : [
                {k: v for k, v in r.items() if k != "steps"}
                for r in results
            ]
        }

    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Full results saved to {out_path}")


# -------------------
# Main
# -------------------
def main():
    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Vocabulary size: {len(vocab)}")

    # Use val set for checking
    val_dataset = MSRVTTDataset("val", vocab, debug_n=None)
    val_loader  = DataLoader(
        val_dataset, batch_size=N_SAMPLES,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    results_dict = {}

    for approach in ["approach1", "approach2"]:
        print(f"\nLoading {approach}...")
        model, ckpt = load_model(approach, vocab)

        if model is None:
            print(f"  Skipping {approach} — checkpoint not found")
            continue

        results, overall_std = run_attention_check(
            approach, model, val_loader, vocab)

        results_dict[approach] = (results, overall_std)

    # -------------------
    # Final comparison
    # -------------------
    if len(results_dict) == 2:
        std1 = results_dict["approach1"][1]
        std2 = results_dict["approach2"][1]

        print(f"\n{'='*70}")
        print(f"FINAL ATTENTION COMPARISON")
        print(f"{'='*70}")
        print(f"  Approach 1  avg attention std : {std1:.4f}  "
              f"{interpret_std(std1)}")
        print(f"  Approach 2  avg attention std : {std2:.4f}  "
              f"{interpret_std(std2)}")

        better = "approach1" if std1 > std2 else "approach2"
        print(f"\n  Better attention : {better}")
        print(f"  (higher std = more focused = better)")

    save_results(results_dict)


if __name__ == "__main__":
    main()