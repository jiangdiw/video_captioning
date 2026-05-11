import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.msrvtt import normalize_dataset_mode
from misc.cocoeval import COCOScorer
from models.flexible_bart_captioning_model import FlexibleBartCaptioningModel


CLIP_DIM = 512
DINO_DIM = 768
AUDIO_DIM = 128


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata_path(resource_root: Path, dataset_mode: str) -> Path:
    dataset_root = resource_root / "dataset" / "MSR-VTT"
    if dataset_mode == "subset":
        return dataset_root / "downsampled_2500.json"
    return dataset_root / "train_val_videodatainfo.json"


def processed_root(resource_root: Path, dataset_mode: str) -> Path:
    return resource_root / "data" / ("processed" if dataset_mode == "subset" else "processed_full")


def load_split_video_ids(resource_root: Path, dataset_mode: str) -> dict[str, list[str]]:
    raw = json.loads(metadata_path(resource_root, dataset_mode).read_text())
    split_map = {"train": [], "val": [], "test": []}
    for video in raw.get("videos", []):
        split = video.get("split", "")
        if split == "validate":
            split = "val"
        if split in split_map:
            split_map[split].append(video["video_id"])
    for split in split_map:
        split_map[split] = sorted(
            split_map[split],
            key=lambda video_id: int(video_id.replace("video", "")),
        )
    return split_map


def load_split_video_ids_from_captions(captions_root: Path) -> dict[str, list[str]]:
    split_map: dict[str, list[str]] = {}
    for split in ["train", "val", "test"]:
        payload = json.loads((captions_root / f"{split}_captions.json").read_text())
        split_map[split] = sorted(
            payload.keys(),
            key=lambda video_id: int(video_id.replace("video", "")),
        )
    return split_map


def parse_modalities(modality_name: str) -> tuple[bool, bool, bool]:
    mapping = {
        "clip": (True, False, False),
        "clip_dino": (True, True, False),
        "clip_dino_audio": (True, True, True),
    }
    if modality_name not in mapping:
        raise ValueError(f"Unsupported modality configuration: {modality_name}")
    return mapping[modality_name]


@dataclass
class ExperimentLayout:
    resource_root: Path
    dataset_mode: str
    run_dir: Path
    captions_root: Path
    visual_root: Path
    audio_root: Path
    fused_visual_roots: list[Path]


def resolve_layout(args: argparse.Namespace) -> ExperimentLayout:
    resource_root = Path(args.resource_root).expanduser().resolve()
    base_processed_root = processed_root(resource_root, args.dataset_mode)
    captions_root = (
        Path(args.captions_root).expanduser().resolve()
        if args.captions_root
        else base_processed_root / "captions"
    )
    visual_root = (
        Path(args.visual_root).expanduser().resolve()
        if args.visual_root
        else base_processed_root / "visual"
    )
    audio_root = (
        Path(args.audio_root).expanduser().resolve()
        if args.audio_root
        else base_processed_root / "audio" / "vggish_embeddings"
    )
    fused_visual_roots = []
    if args.fused_visual_roots:
        for raw in args.fused_visual_roots.split(","):
            raw = raw.strip()
            if not raw:
                continue
            fused_visual_roots.append(Path(raw).expanduser().resolve())
    return ExperimentLayout(
        resource_root=resource_root,
        dataset_mode=args.dataset_mode,
        run_dir=Path(args.run_dir).expanduser().resolve(),
        captions_root=captions_root,
        visual_root=visual_root,
        audio_root=audio_root,
        fused_visual_roots=fused_visual_roots,
    )


class BartAblationDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_ids: list[str],
        captions_root: Path,
        visual_root: Path,
        audio_root: Path,
        use_dino: bool,
        use_audio: bool,
        fused_visual_roots: list[Path] | None = None,
        debug_n: int | None = None,
    ) -> None:
        self.split = split
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.fused_visual_roots = fused_visual_roots or []
        self.clip_dir = visual_root / "clip_embedding" / split
        self.dino_dir = visual_root / "Dinov2_embedding" / split
        self.audio_dir = audio_root / split
        self.captions = json.loads((captions_root / f"{split}_captions.json").read_text())

        self.video_ids: list[str] = []
        for vid in split_ids:
            if vid not in self.captions:
                continue
            if self.fused_visual_roots:
                if self._find_fused_path(vid) is None:
                    continue
            else:
                if not (self.clip_dir / f"{vid}.npy").exists():
                    continue
                if self.use_dino and not (self.dino_dir / f"{vid}.npy").exists():
                    continue
            if self.use_audio and not (self.audio_dir / f"{vid}.npy").exists():
                continue
            self.video_ids.append(vid)

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]

        print(f"[{split}] {len(self.video_ids)} usable videos")

    def __len__(self) -> int:
        return len(self.video_ids)

    def _find_fused_path(self, vid: str) -> Path | None:
        for root in self.fused_visual_roots:
            candidate = root / f"{vid}.npy"
            if candidate.exists():
                return candidate
        return None

    def __getitem__(self, idx: int):
        vid = self.video_ids[idx]
        if self.fused_visual_roots:
            fused_path = self._find_fused_path(vid)
            if fused_path is None:
                raise FileNotFoundError(f"Missing fused visual feature for {vid}")
            fused = torch.tensor(np.load(fused_path), dtype=torch.float32)
            clip = fused[:, :CLIP_DIM]
            if self.use_dino:
                dino = fused[:, CLIP_DIM : CLIP_DIM + DINO_DIM]
            else:
                dino = torch.zeros(clip.shape[0], DINO_DIM, dtype=torch.float32)
        else:
            clip = torch.tensor(np.load(self.clip_dir / f"{vid}.npy"), dtype=torch.float32)
            if self.use_dino:
                dino = torch.tensor(np.load(self.dino_dir / f"{vid}.npy"), dtype=torch.float32)
            else:
                dino = torch.zeros(clip.shape[0], DINO_DIM, dtype=torch.float32)
        if self.use_audio:
            audio = torch.tensor(np.load(self.audio_dir / f"{vid}.npy"), dtype=torch.float32)
        else:
            audio = torch.zeros(1, AUDIO_DIM, dtype=torch.float32)
        caption = random.choice(self.captions[vid])
        return clip, dino, audio, caption, vid


def make_collate_fn(tokenizer: BartTokenizer, max_caption_len: int):
    def collate(batch):
        clips, dinos, audios, captions, video_ids = zip(*batch)
        clips = torch.stack(clips)
        dinos = torch.stack(dinos)

        max_audio_steps = max(audio.shape[0] for audio in audios)
        audio_padded = torch.zeros(len(audios), max_audio_steps, AUDIO_DIM)
        audio_mask = torch.zeros(len(audios), max_audio_steps, dtype=torch.long)
        for index, audio in enumerate(audios):
            audio_padded[index, : audio.shape[0]] = audio
            audio_mask[index, : audio.shape[0]] = 1

        encoded = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_caption_len,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return clips, dinos, audio_padded, audio_mask, labels, list(captions), list(video_ids)

    return collate


def train_one_epoch(model, loader, optimizer, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for clips, dinos, audios, audio_mask, labels, _, _ in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(clips, dinos, audios, audio_mask=audio_mask, labels=labels)
        out.loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += out.loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for clips, dinos, audios, audio_mask, labels, _, _ in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        labels = labels.to(device)
        out = model(clips, dinos, audios, audio_mask=audio_mask, labels=labels)
        total_loss += out.loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def build_ground_truth(captions_path: Path) -> dict[str, list[dict[str, str]]]:
    captions = json.loads(captions_path.read_text())
    return {
        video_id: [{"image_id": video_id, "caption": caption} for caption in caption_list]
        for video_id, caption_list in captions.items()
    }


@torch.no_grad()
def generate_predictions(
    model,
    loader,
    tokenizer,
    device,
    num_beams: int,
    max_new_tokens: int,
) -> dict[str, list[dict[str, str]]]:
    model.eval()
    predictions: dict[str, list[dict[str, str]]] = {}
    for clips, dinos, audios, audio_mask, _, _, video_ids in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        token_ids = model.generate(
            clips,
            dinos,
            audios,
            audio_mask=audio_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        for row, video_id in enumerate(video_ids):
            caption = tokenizer.decode(token_ids[row], skip_special_tokens=True).strip()
            predictions[video_id] = [{"image_id": video_id, "caption": caption}]
    return predictions


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate one BART ablation experiment.")
    parser.add_argument("--resource-root", required=True, help="Root containing data/processed* and dataset/MSR-VTT")
    parser.add_argument("--dataset-mode", choices=["subset", "full"], required=True)
    parser.add_argument("--modalities", choices=["clip", "clip_dino", "clip_dino_audio"], required=True)
    parser.add_argument("--freeze-decoder", action="store_true")
    parser.add_argument("--visual-root", default=None, help="Root containing clip_embedding/ and Dinov2_embedding/")
    parser.add_argument(
        "--fused-visual-roots",
        default=None,
        help="Comma-separated flat directories containing TASCC fused video*.npy files. When set, these are used instead of split clip/dino directories.",
    )
    parser.add_argument("--audio-root", default=None, help="Root containing split audio npy files")
    parser.add_argument("--captions-root", default=None, help="Root containing train/val/test_captions.json")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-caption-len", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-if-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.dataset_mode = normalize_dataset_mode(args.dataset_mode)
    layout = resolve_layout(args)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = layout.run_dir / "metrics.json"
    if args.skip_if_complete and metrics_path.exists():
        print(f"Skipping completed run: {layout.run_dir}")
        return 0

    set_seed(args.seed)
    device = choose_device(args.device)
    use_clip, use_dino, use_audio = parse_modalities(args.modalities)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    split_ids = load_split_video_ids_from_captions(layout.captions_root)
    train_ds = BartAblationDataset(
        split="train",
        split_ids=split_ids["train"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        use_dino=use_dino,
        use_audio=use_audio,
        fused_visual_roots=layout.fused_visual_roots,
        debug_n=args.debug_n,
    )
    val_ds = BartAblationDataset(
        split="val",
        split_ids=split_ids["val"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        use_dino=use_dino,
        use_audio=use_audio,
        fused_visual_roots=layout.fused_visual_roots,
        debug_n=args.debug_n,
    )
    test_ds = BartAblationDataset(
        split="test",
        split_ids=split_ids["test"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        use_dino=use_dino,
        use_audio=use_audio,
        fused_visual_roots=layout.fused_visual_roots,
        debug_n=args.debug_n,
    )

    collate_fn = make_collate_fn(tokenizer, args.max_caption_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = FlexibleBartCaptioningModel(
        use_dino=use_dino,
        use_audio=use_audio,
        freeze_decoder=args.freeze_decoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "resource_root": str(layout.resource_root),
        "dataset_mode": args.dataset_mode,
        "modalities": args.modalities,
        "freeze_decoder": args.freeze_decoder,
        "visual_root": str(layout.visual_root),
        "fused_visual_roots": [str(path) for path in layout.fused_visual_roots],
        "audio_root": str(layout.audio_root),
        "captions_root": str(layout.captions_root),
        "run_dir": str(layout.run_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "num_beams": args.num_beams,
        "max_caption_len": args.max_caption_len,
        "device": str(device),
        "seed": args.seed,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }
    save_json(layout.run_dir / "config.json", config)

    best_val = math.inf
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        val_loss = evaluate_loss(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")
        save_json(layout.run_dir / "history.json", history)
        latest_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "val_loss": val_loss,
            "config": config,
        }
        torch.save(latest_payload, layout.run_dir / "latest.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(latest_payload, layout.run_dir / "best.pt")

    best_ckpt = torch.load(layout.run_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    predictions = generate_predictions(
        model=model,
        loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        num_beams=args.num_beams,
        max_new_tokens=args.max_caption_len,
    )
    ground_truth = build_ground_truth(layout.captions_root / "test_captions.json")
    test_ids = list(predictions.keys())
    metrics = COCOScorer().score(ground_truth, predictions, test_ids)
    metrics["best_val_loss"] = float(best_ckpt["val_loss"])
    metrics["best_epoch"] = int(best_ckpt["epoch"])

    save_json(layout.run_dir / "predictions.json", predictions)
    save_json(layout.run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
