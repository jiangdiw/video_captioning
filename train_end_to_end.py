# src/training/train_end_to_end.py
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.vocabulary           import Vocabulary
from data.dataset              import (
    MSRVTTDataset,
    PrecomputedDataset,
    SequencePrecomputedDataset,
    collate_fn,
    collate_fn_precomputed,
    collate_fn_sequence_precomputed,
)
from data.msrvtt import get_processed_layout, normalize_dataset_mode
from models.multimodal_encoder import Approach1Encoder, Approach2Encoder
from models.captioning_model   import (
    CaptioningModel,
    PrecomputedCaptioningModel,
    SequencePrecomputedCaptioningModel,
)
from misc.cocoeval import COCOScorer, suppress_stdout_stderr

# -------------------
# CONFIG
# -------------------
LOG_DIR    = PROJECT_ROOT / "outputs/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT = PROJECT_ROOT / "outputs/results"
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs/checkpoints"

MAX_GENERATION_LEN = 20
SAVE_EVERY = 10

HYPERPARAMETER_PRESETS = {
    "stable": {
        "epochs": 150,
        "raw_batch_size": 8,
        "precomputed_batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "hidden_dim": 512,
        "embed_dim": 256,
        "dropout": 0.45,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
    },
    "best_guess": {
        "epochs": 150,
        "raw_batch_size": 8,
        "precomputed_batch_size": 32,
        "lr": 2.5e-4,
        "weight_decay": 1e-4,
        "hidden_dim": 640,
        "embed_dim": 320,
        "dropout": 0.35,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
    },
    "capacity": {
        "epochs": 180,
        "raw_batch_size": 6,
        "precomputed_batch_size": 24,
        "lr": 2e-4,
        "weight_decay": 5e-5,
        "hidden_dim": 768,
        "embed_dim": 384,
        "dropout": 0.30,
        "label_smoothing": 0.05,
        "grad_clip": 0.75,
    },
}
DEFAULT_PRESET = "best_guess"

def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_run_config(args, use_precomputed):
    preset = HYPERPARAMETER_PRESETS[args.preset].copy()
    preset["batch_size"] = (
        preset["precomputed_batch_size"] if use_precomputed else preset["raw_batch_size"]
    )
    if args.epochs is not None:
        preset["epochs"] = args.epochs
    if args.batch_size is not None:
        preset["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        preset["lr"] = args.learning_rate
    if args.hidden_dim is not None:
        preset["hidden_dim"] = args.hidden_dim
    if args.embed_dim is not None:
        preset["embed_dim"] = args.embed_dim
    if args.dropout is not None:
        preset["dropout"] = args.dropout
    if args.weight_decay is not None:
        preset["weight_decay"] = args.weight_decay
    if args.label_smoothing is not None:
        preset["label_smoothing"] = args.label_smoothing
    if args.grad_clip is not None:
        preset["grad_clip"] = args.grad_clip
    return preset

# -------------------
# Loss: cross entropy ignoring padding
# -------------------
def caption_loss(logits, captions, pad_idx, label_smoothing=0.0):
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
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )

# -------------------
# One epoch
# -------------------
def run_epoch(model, loader, optimizer, pad_idx, config, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        # _ ignores raw caption strings (only needed for display)
        for clips, dinos, audios, caps, _, _ in loader:
            clips  = clips.to(device)
            dinos  = dinos.to(device)
            audios = audios.to(device)
            caps   = caps.to(device)

            logits = model(clips, dinos, audios, caps)
            loss   = caption_loss(
                logits, caps, pad_idx, label_smoothing=config["label_smoothing"]
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / n_batches


def run_epoch_precomputed(model, loader, optimizer, pad_idx, config, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for embeddings, caps, _ in loader:
            embeddings = embeddings.to(device)
            caps = caps.to(device)

            logits = model(embeddings, caps)
            loss = caption_loss(
                logits, caps, pad_idx, label_smoothing=config["label_smoothing"]
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def run_epoch_sequence_precomputed(model, loader, optimizer, pad_idx, config, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for seq_embeddings, pooled_embeddings, caps, _ in loader:
            seq_embeddings = seq_embeddings.to(device)
            pooled_embeddings = pooled_embeddings.to(device)
            caps = caps.to(device)

            logits = model(seq_embeddings, pooled_embeddings, caps)
            loss = caption_loss(
                logits, caps, pad_idx, label_smoothing=config["label_smoothing"]
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def build_coco_ground_truth(captions):
    return {
        video_id: [
            {"image_id": video_id, "cap_id": idx, "caption": caption}
            for idx, caption in enumerate(video_captions)
        ]
        for video_id, video_captions in captions.items()
    }


def serialize_scores(scores):
    return {key: float(value) for key, value in scores.items()}


def evaluate_cider(model, loader, dataset, vocab, device, generation_max_len, feature_mode):
    model.eval()
    samples = {}
    scorer = COCOScorer()
    gts = build_coco_ground_truth(dataset.captions)

    with torch.no_grad():
        for batch in loader:
            if feature_mode == "raw":
                clips, dinos, audios, _, _, video_ids = batch
                clips = clips.to(device)
                dinos = dinos.to(device)
                audios = audios.to(device)
                seq_preds = model.generate(
                    clips,
                    dinos,
                    audios,
                    max_len=generation_max_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )
            elif feature_mode == "precomputed_sequence":
                seq_embeddings, pooled_embeddings, _, video_ids = batch
                seq_embeddings = seq_embeddings.to(device)
                pooled_embeddings = pooled_embeddings.to(device)
                seq_preds = model.generate(
                    seq_embeddings,
                    pooled_embeddings,
                    max_len=generation_max_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )
            else:
                embeddings, _, video_ids = batch
                embeddings = embeddings.to(device)
                seq_preds = model.generate(
                    embeddings,
                    max_len=generation_max_len,
                    sos_idx=vocab.sos_idx,
                    eos_idx=vocab.eos_idx,
                )

            for index, seq in enumerate(seq_preds):
                video_id = video_ids[index]
                samples[video_id] = [
                    {"image_id": video_id, "caption": vocab.decode(seq.tolist())}
                ]

    with suppress_stdout_stderr():
        scores = scorer.score(gts, samples, samples.keys())
    return serialize_scores(scores)

# -------------------
# Sample generations
# -------------------
def show_samples(model, loader, vocab, approach_name, device, n=3):
    model.eval()
    clips, dinos, audios, caps, raw_captions, vid_ids = next(iter(loader))
    sample_n = min(n, len(vid_ids))
    clips  = clips[:sample_n].to(device)
    dinos  = dinos[:sample_n].to(device)
    audios = audios[:sample_n].to(device)

    generated = model.generate(
        clips, dinos, audios,
        max_len = MAX_GENERATION_LEN,
        sos_idx = vocab.sos_idx,
        eos_idx = vocab.eos_idx
    )

    print(f"\n  [{approach_name}] Sample generations:")
    for i in range(sample_n):
        gt  = raw_captions[i]                      # original string, no <unk>
        gen = vocab.decode(generated[i].tolist())  # model output
        print(f"    [{vid_ids[i]}]")
        print(f"      GT  : {gt}")
        print(f"      GEN : {gen}")


def show_samples_precomputed(model, loader, vocab, approach_name, device, n=3):
    model.eval()
    embeddings, caps, vid_ids = next(iter(loader))
    embeddings = embeddings[:n].to(device)
    generated = model.generate(
        embeddings,
        max_len=MAX_GENERATION_LEN,
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
    )

    print(f"\n  [{approach_name}] Sample generations:")
    for i in range(min(n, len(vid_ids))):
        gt = vocab.decode(caps[i].tolist())
        gen = vocab.decode(generated[i].tolist())
        print(f"    [{vid_ids[i]}]")
        print(f"      GT  : {gt}")
        print(f"      GEN : {gen}")


def show_samples_sequence_precomputed(model, loader, vocab, approach_name, device, n=3):
    model.eval()
    seq_embeddings, pooled_embeddings, caps, vid_ids = next(iter(loader))
    seq_embeddings = seq_embeddings[:n].to(device)
    pooled_embeddings = pooled_embeddings[:n].to(device)
    generated = model.generate(
        seq_embeddings,
        pooled_embeddings,
        max_len=MAX_GENERATION_LEN,
        sos_idx=vocab.sos_idx,
        eos_idx=vocab.eos_idx,
    )

    print(f"\n  [{approach_name}] Sample generations:")
    for i in range(min(n, len(vid_ids))):
        gt = vocab.decode(caps[i].tolist())
        gen = vocab.decode(generated[i].tolist())
        print(f"    [{vid_ids[i]}]")
        print(f"      GT  : {gt}")
        print(f"      GEN : {gen}")

# -------------------
# Train one approach
# -------------------
def train_approach(name, model, train_loader, val_loader, vocab, config, device, layout):
    ckpt_dir = CHECKPOINT_ROOT / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt_path = save_opt_info(ckpt_dir, name, config, "raw", name.split("_")[0], vocab, layout)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*55}")
    print(f"Training: {name}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  Max epochs    : {config['epochs']}")
    print(f"  Batch size    : {config['batch_size']}")
    print(f"  LR            : {config['lr']}")
    print(f"  Weight decay  : {config['weight_decay']}")
    print(f"  Dropout       : {config['dropout']}")
    print(f"  Label smooth  : {config['label_smoothing']}")
    print(f"  Saved config  : {opt_path}")
    print(f"{'='*55}\n")

    optimizer     = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_val = float("inf")
    best_cider = float("-inf")
    history  = {"train": [], "val": [], "val_metrics": []}

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, optimizer,
                               vocab.pad_idx, config, device, train=True)
        val_loss   = run_epoch(model, val_loader, optimizer,
                               vocab.pad_idx, config, device, train=False)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:03d}/{config['epochs']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}")

        val_scores = evaluate_cider(
            model,
            val_loader,
            val_loader.dataset,
            vocab,
            device,
            MAX_GENERATION_LEN,
            feature_mode="raw",
        )
        history["val_metrics"].append(val_scores)
        val_cider = val_scores.get("CIDEr", float("-inf"))
        print(f"    CIDEr={val_cider:.4f}  Bleu_4={val_scores.get('Bleu_4', 0.0):.4f}")

        # Save best loss checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "val_cider"   : val_cider,
                "history"     : history,
                "approach"    : name,
                "framework"   : "end_to_end",
                "feature_mode": "raw",
                "variant"     : name.split("_")[0],
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best_loss.pth")
            print(f"    → Best loss saved (val={val_loss:.4f})")

        if val_cider > best_cider:
            best_cider = val_cider
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_loss"    : val_loss,
                "val_cider"   : val_cider,
                "history"     : history,
                "approach"    : name,
                "framework"   : "end_to_end",
                "feature_mode": "raw",
                "variant"     : name.split("_")[0],
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best.pth")
            print(f"    → Best CIDEr saved (CIDEr={val_cider:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "raw",
                "variant": name.split("_")[0],
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / f"model_{epoch}.pth")

        # Save latest every epoch (safe to resume)
        torch.save({
            "epoch"           : epoch,
            "model_state"     : model.state_dict(),
            "optimizer_state" : optimizer.state_dict(),
            "val_loss"        : val_loss,
            "val_cider"       : val_cider,
            "history"         : history,
            "approach"        : name,
            "framework"       : "end_to_end",
            "feature_mode"    : "raw",
            "variant"         : name.split("_")[0],
            "opt_info_path"   : str(opt_path),
            "config"          : config,
        }, ckpt_dir / "latest.pth")

        # Show samples every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            show_samples(model, val_loader, vocab, name, device)

    # Save history
    with open(LOG_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    actual_epochs = len(history["train"])
    print(f"\n[{name}]")
    print(f"  Ran       : {actual_epochs}/{config['epochs']} epochs")
    print(f"  Best val  : {best_val:.4f}")
    print(f"  Best CIDEr: {best_cider:.4f}")

    return history, best_val, best_cider


def train_approach_precomputed(name, model, train_loader, val_loader, vocab, config, device, layout, input_dim):
    ckpt_dir = CHECKPOINT_ROOT / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    variant = "approach1" if input_dim == 640 else "approach2"
    opt_path = save_opt_info(ckpt_dir, name, config, "precomputed", variant, vocab, layout, input_dim=input_dim)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"Training: {name}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  Max epochs    : {config['epochs']}")
    print(f"  Batch size    : {config['batch_size']}")
    print(f"  LR            : {config['lr']}")
    print(f"  Weight decay  : {config['weight_decay']}")
    print(f"  Dropout       : {config['dropout']}")
    print(f"  Label smooth  : {config['label_smoothing']}")
    print(f"  Saved config  : {opt_path}")
    print(f"{'='*55}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best_val = float("inf")
    best_cider = float("-inf")
    history = {"train": [], "val": [], "val_metrics": []}

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch_precomputed(model, train_loader, optimizer, vocab.pad_idx, config, device, train=True)
        val_loss = run_epoch_precomputed(model, val_loader, optimizer, vocab.pad_idx, config, device, train=False)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:03d}/{config['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}")

        val_scores = evaluate_cider(
            model,
            val_loader,
            val_loader.dataset,
            vocab,
            device,
            MAX_GENERATION_LEN,
            feature_mode="precomputed",
        )
        history["val_metrics"].append(val_scores)
        val_cider = val_scores.get("CIDEr", float("-inf"))
        print(f"    CIDEr={val_cider:.4f}  Bleu_4={val_scores.get('Bleu_4', 0.0):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed",
                "variant": variant,
                "input_dim": input_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best_loss.pth")
            print(f"    → Best loss saved (val={val_loss:.4f})")

        if val_cider > best_cider:
            best_cider = val_cider
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed",
                "variant": variant,
                "input_dim": input_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best.pth")
            print(f"    → Best CIDEr saved (CIDEr={val_cider:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed",
                "variant": variant,
                "input_dim": input_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / f"model_{epoch}.pth")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_cider": val_cider,
            "history": history,
            "approach": name,
            "framework": "end_to_end",
            "feature_mode": "precomputed",
            "variant": variant,
            "input_dim": input_dim,
            "opt_info_path": str(opt_path),
            "config": config,
        }, ckpt_dir / "latest.pth")

        if epoch % 10 == 0 or epoch == 1:
            show_samples_precomputed(model, val_loader, vocab, name, device)

    with open(LOG_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{name}]")
    print(f"  Ran       : {len(history['train'])}/{config['epochs']} epochs")
    print(f"  Best val  : {best_val:.4f}")
    print(f"  Best CIDEr: {best_cider:.4f}")
    return history, best_val, best_cider


def train_approach_precomputed_sequence(name, model, train_loader, val_loader, vocab, config, device, layout, input_dim, pooled_dim):
    ckpt_dir = CHECKPOINT_ROOT / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    variant = "approach1" if pooled_dim == 640 else "approach2"
    opt_path = save_opt_info(
        ckpt_dir,
        name,
        config,
        "precomputed_sequence",
        variant,
        vocab,
        layout,
        input_dim=input_dim,
        pooled_input_dim=pooled_dim,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*55}")
    print(f"Training: {name}")
    print(f"  Parameters    : {total_params:,}")
    print(f"  Max epochs    : {config['epochs']}")
    print(f"  Batch size    : {config['batch_size']}")
    print(f"  LR            : {config['lr']}")
    print(f"  Weight decay  : {config['weight_decay']}")
    print(f"  Dropout       : {config['dropout']}")
    print(f"  Label smooth  : {config['label_smoothing']}")
    print(f"  Saved config  : {opt_path}")
    print(f"{'='*55}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best_val = float("inf")
    best_cider = float("-inf")
    history = {"train": [], "val": [], "val_metrics": []}

    for epoch in range(1, config["epochs"] + 1):
        train_loss = run_epoch_sequence_precomputed(
            model, train_loader, optimizer, vocab.pad_idx, config, device, train=True
        )
        val_loss = run_epoch_sequence_precomputed(
            model, val_loader, optimizer, vocab.pad_idx, config, device, train=False
        )
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:03d}/{config['epochs']}  train={train_loss:.4f}  val={val_loss:.4f}")

        val_scores = evaluate_cider(
            model,
            val_loader,
            val_loader.dataset,
            vocab,
            device,
            MAX_GENERATION_LEN,
            feature_mode="precomputed_sequence",
        )
        history["val_metrics"].append(val_scores)
        val_cider = val_scores.get("CIDEr", float("-inf"))
        print(f"    CIDEr={val_cider:.4f}  Bleu_4={val_scores.get('Bleu_4', 0.0):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed_sequence",
                "variant": variant,
                "input_dim": input_dim,
                "pooled_input_dim": pooled_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best_loss.pth")
            print(f"    → Best loss saved (val={val_loss:.4f})")

        if val_cider > best_cider:
            best_cider = val_cider
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed_sequence",
                "variant": variant,
                "input_dim": input_dim,
                "pooled_input_dim": pooled_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / "best.pth")
            print(f"    → Best CIDEr saved (CIDEr={val_cider:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "val_cider": val_cider,
                "history": history,
                "approach": name,
                "framework": "end_to_end",
                "feature_mode": "precomputed_sequence",
                "variant": variant,
                "input_dim": input_dim,
                "pooled_input_dim": pooled_dim,
                "opt_info_path": str(opt_path),
                "config": config,
            }, ckpt_dir / f"model_{epoch}.pth")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_cider": val_cider,
            "history": history,
            "approach": name,
            "framework": "end_to_end",
            "feature_mode": "precomputed_sequence",
            "variant": variant,
            "input_dim": input_dim,
            "pooled_input_dim": pooled_dim,
            "opt_info_path": str(opt_path),
            "config": config,
        }, ckpt_dir / "latest.pth")

        if epoch % 10 == 0 or epoch == 1:
            show_samples_sequence_precomputed(model, val_loader, vocab, name, device)

    with open(LOG_DIR / f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{name}]")
    print(f"  Ran       : {len(history['train'])}/{config['epochs']} epochs")
    print(f"  Best val  : {best_val:.4f}")
    print(f"  Best CIDEr: {best_cider:.4f}")
    return history, best_val, best_cider


def has_raw_modalities(layout):
    return layout.raw_clip_root.exists() and layout.raw_dino_root.exists() and layout.raw_audio_root.exists()


def has_precomputed_modalities(layout):
    return layout.approach1_root.exists() and layout.approach2_root.exists()


def has_precomputed_sequence_modalities(layout):
    return layout.approach1_seq_root.exists() and layout.approach2_seq_root.exists()


def save_opt_info(ckpt_dir, name, config, feature_mode, variant, vocab, layout, input_dim=None, pooled_input_dim=None):
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)
    opt = {
        "framework": "end_to_end",
        "dataset_mode": layout.mode,
        "processed_root": str(layout.processed_root),
        "feature_mode": feature_mode,
        "variant": variant,
        "model_name": name,
        "vocab_path": str(layout.vocab_path),
        "train_caption_json": str(layout.captions_root / "train_captions.json"),
        "val_caption_json": str(layout.captions_root / "val_captions.json"),
        "test_caption_json": str(layout.captions_root / "test_captions.json"),
        "results_path": str(results_dir),
        "checkpoint_dir": str(ckpt_dir),
        "default_saved_model": str(ckpt_dir / "best.pth"),
        "embed_dim": config["embed_dim"],
        "hidden_dim": config["hidden_dim"],
        "dropout": config["dropout"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "learning_rate": config["lr"],
        "weight_decay": config["weight_decay"],
        "label_smoothing": config["label_smoothing"],
        "grad_clip": config["grad_clip"],
        "generation_max_len": MAX_GENERATION_LEN,
        "save_every": SAVE_EVERY,
        "vocab_size": len(vocab),
        "preset": config.get("preset_name", DEFAULT_PRESET),
    }
    if feature_mode == "precomputed":
        opt["input_dim"] = input_dim
        opt["precomputed_dir"] = str(
            layout.approach1_root if variant == "approach1" else layout.approach2_root
        )
    elif feature_mode == "precomputed_sequence":
        opt["input_dim"] = input_dim
        opt["pooled_input_dim"] = pooled_input_dim
        opt["precomputed_dir"] = str(
            layout.approach1_seq_root if variant == "approach1" else layout.approach2_seq_root
        )
    else:
        opt["raw_clip_dir"] = str(layout.raw_clip_root)
        opt["raw_dino_dir"] = str(layout.raw_dino_root)
        opt["raw_audio_dir"] = str(layout.raw_audio_root)

    opt_path = ckpt_dir / "opt_info.json"
    with open(opt_path, "w") as f:
        json.dump(opt, f, indent=2)
    return opt_path

# -------------------
# Main
# -------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="subset")
    parser.add_argument("--approach", choices=["approach1", "approach2", "both"], default="approach1")
    parser.add_argument("--preset", choices=sorted(HYPERPARAMETER_PRESETS.keys()), default=DEFAULT_PRESET)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-n", type=int, default=10)
    return parser.parse_args()


def main(args):
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = get_processed_layout(dataset_mode)
    vocab = Vocabulary.load(layout.vocab_path)
    print(f"Vocabulary size: {len(vocab)}\n")

    device = resolve_device(args.device)
    print(f"Using device: {device}\n")

    use_raw = has_raw_modalities(layout)
    use_precomputed_sequence = has_precomputed_sequence_modalities(layout)
    use_precomputed = has_precomputed_modalities(layout)
    if not use_raw and not use_precomputed_sequence and not use_precomputed:
        raise FileNotFoundError(
            f"Could not find raw CLIP/DINO/audio splits, sequence-precomputed multimodal embeddings, or pooled precomputed embeddings under {layout.processed_root}."
        )

    if use_raw:
        feature_mode = "raw"
    elif use_precomputed_sequence:
        feature_mode = "precomputed_sequence"
    else:
        feature_mode = "precomputed"

    config = build_run_config(args, use_precomputed=(feature_mode != "raw"))
    config["preset_name"] = args.preset
    debug_n = args.debug_n if args.debug else None

    print(f"Training preset: {args.preset}")
    print(f"Dataset mode: {dataset_mode}")
    print(f"Approach selection: {args.approach}")
    print(f"Processed root: {layout.processed_root}")
    print(f"Feature mode: {feature_mode}")
    print(f"Active config: {json.dumps(config, indent=2)}\n")

    def run_name(base_name):
        suffixes = []
        if dataset_mode == "full":
            suffixes.append("full")
        if args.run_suffix:
            suffixes.append(args.run_suffix)
        return "_".join([base_name] + suffixes) if suffixes else base_name

    selected_approaches = ["approach1", "approach2"] if args.approach == "both" else [args.approach]
    trained = {}

    if use_raw:
        print("Using raw CLIP/DINO/audio embeddings.\n")
        train_dataset = MSRVTTDataset("train", vocab, debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root)
        val_dataset   = MSRVTTDataset("val",   vocab, debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root)

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"],
            shuffle=True,  collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=0
        )

        if "approach1" in selected_approaches:
            model1 = CaptioningModel(
                encoder=Approach1Encoder(),
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history1, best_val1, best_cider1 = train_approach(
                run_name("approach1_clip_dino_crossattn_audio_concat"), model1, train_loader, val_loader, vocab, config, device, layout
            )
            trained["approach1"] = {
                "name": "CLIP x DINOv2 cross-attn + audio concat",
                "epochs_run": len(history1["train"]),
                "best_val_loss": best_val1,
                "best_cider": best_cider1,
                "history": history1,
            }

        if "approach2" in selected_approaches:
            model2 = CaptioningModel(
                encoder=Approach2Encoder(),
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history2, best_val2, best_cider2 = train_approach(
                run_name("approach2_trimodal_crossattn"), model2, train_loader, val_loader, vocab, config, device, layout
            )
            trained["approach2"] = {
                "name": "Trimodal sequential cross-attention",
                "epochs_run": len(history2["train"]),
                "best_val_loss": best_val2,
                "best_cider": best_cider2,
                "history": history2,
            }
    elif use_precomputed_sequence:
        print("Using sequence-preserving precomputed multimodal embeddings.\n")
        if "approach1" in selected_approaches:
            train_loader1 = DataLoader(
                SequencePrecomputedDataset("train", vocab, variant="approach1", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_sequence_precomputed,
                num_workers=0,
            )
            val_loader1 = DataLoader(
                SequencePrecomputedDataset("val", vocab, variant="approach1", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_sequence_precomputed,
                num_workers=0,
            )
            model1 = SequencePrecomputedCaptioningModel(
                seq_input_dim=512,
                pooled_input_dim=640,
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history1, best_val1, best_cider1 = train_approach_precomputed_sequence(
                run_name("approach1_clip_dino_crossattn_audio_concat"),
                model1, train_loader1, val_loader1, vocab, config, device, layout,
                input_dim=512, pooled_dim=640
            )
            trained["approach1"] = {
                "name": "CLIP x DINOv2 cross-attn + audio concat",
                "epochs_run": len(history1["train"]),
                "best_val_loss": best_val1,
                "best_cider": best_cider1,
                "history": history1,
            }

        if "approach2" in selected_approaches:
            train_loader2 = DataLoader(
                SequencePrecomputedDataset("train", vocab, variant="approach2", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_sequence_precomputed,
                num_workers=0,
            )
            val_loader2 = DataLoader(
                SequencePrecomputedDataset("val", vocab, variant="approach2", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_sequence_precomputed,
                num_workers=0,
            )
            model2 = SequencePrecomputedCaptioningModel(
                seq_input_dim=512,
                pooled_input_dim=512,
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history2, best_val2, best_cider2 = train_approach_precomputed_sequence(
                run_name("approach2_trimodal_crossattn"),
                model2, train_loader2, val_loader2, vocab, config, device, layout,
                input_dim=512, pooled_dim=512
            )
            trained["approach2"] = {
                "name": "Trimodal sequential cross-attention",
                "epochs_run": len(history2["train"]),
                "best_val_loss": best_val2,
                "best_cider": best_cider2,
                "history": history2,
            }
    else:
        print("Raw CLIP/DINO splits not found. Falling back to precomputed multimodal embeddings.\n")
        if "approach1" in selected_approaches:
            train_loader1 = DataLoader(
                PrecomputedDataset("train", vocab, variant="approach1", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_precomputed,
                num_workers=0,
            )
            val_loader1 = DataLoader(
                PrecomputedDataset("val", vocab, variant="approach1", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_precomputed,
                num_workers=0,
            )
            model1 = PrecomputedCaptioningModel(
                input_dim=640,
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history1, best_val1, best_cider1 = train_approach_precomputed(
                run_name("approach1_clip_dino_crossattn_audio_concat"),
                model1, train_loader1, val_loader1, vocab, config, device, layout, input_dim=640
            )
            trained["approach1"] = {
                "name": "CLIP x DINOv2 cross-attn + audio concat",
                "epochs_run": len(history1["train"]),
                "best_val_loss": best_val1,
                "best_cider": best_cider1,
                "history": history1,
            }

        if "approach2" in selected_approaches:
            train_loader2 = DataLoader(
                PrecomputedDataset("train", vocab, variant="approach2", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=collate_fn_precomputed,
                num_workers=0,
            )
            val_loader2 = DataLoader(
                PrecomputedDataset("val", vocab, variant="approach2", debug_n=debug_n, dataset_mode=dataset_mode, processed_root=layout.processed_root),
                batch_size=config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn_precomputed,
                num_workers=0,
            )
            model2 = PrecomputedCaptioningModel(
                input_dim=512,
                vocab_size=len(vocab),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
            ).to(device)
            history2, best_val2, best_cider2 = train_approach_precomputed(
                run_name("approach2_trimodal_crossattn"),
                model2, train_loader2, val_loader2, vocab, config, device, layout, input_dim=512
            )
            trained["approach2"] = {
                "name": "Trimodal sequential cross-attention",
                "epochs_run": len(history2["train"]),
                "best_val_loss": best_val2,
                "best_cider": best_cider2,
                "history": history2,
            }

    print(f"\n{'='*55}")
    if len(trained) == 1:
        approach_key = next(iter(trained))
        record = trained[approach_key]
        print("TRAINING SUMMARY")
        print(f"{'='*55}")
        print(f"  Approach      : {approach_key}")
        print(f"  Name          : {record['name']}")
        print(f"  Epochs run    : {record['epochs_run']}")
        print(f"  Best val loss : {record['best_val_loss']:.4f}")
        print(f"  Best CIDEr    : {record['best_cider']:.4f}")
    else:
        winner = "Approach 1" if trained["approach1"]["best_cider"] > trained["approach2"]["best_cider"] else "Approach 2"
        diff = abs(trained["approach1"]["best_cider"] - trained["approach2"]["best_cider"])
        print("FINAL COMPARISON")
        print(f"{'='*55}")
        print(f"  Approach 1 (CLIP x DINOv2 + audio concat)")
        print(f"    Epochs run    : {trained['approach1']['epochs_run']}")
        print(f"    Best val loss : {trained['approach1']['best_val_loss']:.4f}")
        print(f"    Best CIDEr    : {trained['approach1']['best_cider']:.4f}")
        print(f"  Approach 2 (Trimodal cross-attention)")
        print(f"    Epochs run    : {trained['approach2']['epochs_run']}")
        print(f"    Best val loss : {trained['approach2']['best_val_loss']:.4f}")
        print(f"    Best CIDEr    : {trained['approach2']['best_cider']:.4f}")
        print(f"\n  Winner     : {winner}")
        print(f"  CIDEr gap   : {diff:.4f}")

    # Save comparison summary
    summary = {
        "dataset_mode": dataset_mode,
        "processed_root": str(layout.processed_root),
        "approach_selection": args.approach,
        "trained": trained,
    }
    comparison_suffixes = []
    if dataset_mode == "full":
        comparison_suffixes.append("full")
    if args.run_suffix:
        comparison_suffixes.append(args.run_suffix)
    comparison_name = (
        f"comparison_summary_{'_'.join(comparison_suffixes)}.json"
        if comparison_suffixes
        else "comparison_summary.json"
    )
    with open(LOG_DIR / comparison_name, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {LOG_DIR}/{comparison_name}")


if __name__ == "__main__":
    main(parse_args())
