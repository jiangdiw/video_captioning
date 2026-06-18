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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, get_cosine_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPENJDK_BIN = Path("/opt/homebrew/opt/openjdk/bin")
if OPENJDK_BIN.exists():
    os.environ["PATH"] = f"{OPENJDK_BIN}:{os.environ.get('PATH', '')}"

COCO_ROOT = PROJECT_ROOT / "coco-caption"
if str(COCO_ROOT) not in sys.path:
    sys.path.insert(0, str(COCO_ROOT))

from pycocoevalcap.cider.cider_scorer import CiderScorer

from data.msrvtt import get_processed_layout, get_split_video_ids_from_captions, normalize_dataset_mode, video_number
from misc.cocoeval import COCOScorer, simple_tokenize
from models.final_bart_captioning_model import FinalBartCaptioningModel
from models.flexible_bart_captioning_model import FlexibleBartCaptioningModel


CLIP_DIM = 512
DINO_DIM = 768
AUDIO_DIM = 128

TUNING_PRESETS: dict[str, dict[str, object]] = {
    # Fresh long run with slower LR decay and beam-4 validation/test.
    "stable_long_v1": {
        "architecture": "stable",
        "decoder_train_mode": "freeze",
        "skip_scst": True,
        "xe_epochs": 60,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "eval_batch_size": 1,
        "encoder_lr": 8e-5,
        "bart_lr": 2e-5,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.06,
        "val_num_beams": 4,
        "test_num_beams": 4,
        "val_length_penalty": 1.0,
        "test_length_penalty": 1.0,
        "val_no_repeat_ngram_size": 0,
        "test_no_repeat_ngram_size": 0,
    },
    # Warm-start continuation from the current best stable XE checkpoint.
    "stable_continue_v1": {
        "architecture": "stable",
        "decoder_train_mode": "freeze",
        "skip_scst": True,
        "xe_epochs": 24,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "eval_batch_size": 1,
        "encoder_lr": 5e-5,
        "bart_lr": 1e-5,
        "weight_decay": 5e-3,
        "warmup_ratio": 0.02,
        "val_num_beams": 4,
        "test_num_beams": 4,
        "val_length_penalty": 1.0,
        "test_length_penalty": 1.0,
        "val_no_repeat_ngram_size": 0,
        "test_no_repeat_ngram_size": 0,
        "init_from_checkpoint": "/Users/aglooney03/video_captioning/outputs/final_bart_full_stable_v1/best_cider_xe.pt",
    },
    # Slightly less conservative continuation in case the lower-LR run stalls.
    "stable_continue_v2": {
        "architecture": "stable",
        "decoder_train_mode": "freeze",
        "skip_scst": True,
        "xe_epochs": 24,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "eval_batch_size": 1,
        "encoder_lr": 7e-5,
        "bart_lr": 1.5e-5,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.02,
        "val_num_beams": 4,
        "test_num_beams": 4,
        "val_length_penalty": 1.0,
        "test_length_penalty": 1.0,
        "val_no_repeat_ngram_size": 0,
        "test_no_repeat_ngram_size": 0,
        "init_from_checkpoint": "/Users/aglooney03/video_captioning/outputs/final_bart_full_stable_v1/best_cider_xe.pt",
    },
    # Higher-time continuation from the current best tuned checkpoint.
    "stable_continue_v3": {
        "architecture": "stable",
        "decoder_train_mode": "freeze",
        "skip_scst": True,
        "xe_epochs": 32,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "eval_batch_size": 1,
        "encoder_lr": 3e-5,
        "bart_lr": 5e-6,
        "weight_decay": 5e-3,
        "warmup_ratio": 0.02,
        "val_num_beams": 5,
        "test_num_beams": 5,
        "val_length_penalty": 1.0,
        "test_length_penalty": 1.0,
        "val_no_repeat_ngram_size": 0,
        "test_no_repeat_ngram_size": 0,
        "init_from_checkpoint": "/Users/aglooney03/video_captioning/outputs/final_bart_full_tuned_continue_v1/best_cider_xe.pt",
    },
    # Long fresh run with more time budget on the same stable architecture.
    "stable_long_v2": {
        "architecture": "stable",
        "decoder_train_mode": "freeze",
        "skip_scst": True,
        "xe_epochs": 80,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "eval_batch_size": 1,
        "encoder_lr": 7e-5,
        "bart_lr": 2e-5,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.08,
        "val_num_beams": 5,
        "test_num_beams": 5,
        "val_length_penalty": 1.0,
        "test_length_penalty": 1.0,
        "val_no_repeat_ngram_size": 0,
        "test_no_repeat_ngram_size": 0,
    },
}


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


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def apply_tuning_preset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    if args.preset == "none":
        return args
    preset = TUNING_PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    return args


def parse_modalities(name: str) -> tuple[bool, bool, bool]:
    mapping = {
        "clip": (True, False, False),
        "clip_dino": (True, True, False),
        "clip_dino_audio": (True, True, True),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported modalities setting: {name}")
    return mapping[name]


def dedupe_existing_dirs(candidates: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    existing: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def discover_fused_visual_roots(resource_root: Path) -> list[Path]:
    candidates = [
        resource_root / "features" / "tascc_fused",
        resource_root / "datas" / "feats" / "tascc_fused",
        PROJECT_ROOT / "features" / "tascc_fused",
        PROJECT_ROOT / "datas" / "feats" / "tascc_fused",
        Path("/Users/aglooney03/Video-Summarization/features/tascc_fused"),
        Path("/Users/aglooney03/Video-Summarization/datas/feats/tascc_fused"),
        Path(
            "/Users/aglooney03/Library/CloudStorage/GoogleDrive-aidanlooney@g.harvard.edu/My Drive/"
            "undergrad-research/video_captioning_project/Video-Summarization/features/tascc_fused"
        ),
        Path(
            "/Users/aglooney03/Library/CloudStorage/GoogleDrive-aidanlooney@g.harvard.edu/My Drive/"
            "undergrad-research/video_captioning_project/Video-Summarization/datas/feats/tascc_fused"
        ),
    ]
    return dedupe_existing_dirs(candidates)


@dataclass
class FinalLayout:
    resource_root: Path
    dataset_mode: str
    run_dir: Path
    captions_root: Path
    visual_root: Path
    audio_root: Path
    visual_source: str
    fused_visual_roots: list[Path]


def resolve_layout(args: argparse.Namespace) -> FinalLayout:
    dataset_mode = normalize_dataset_mode(args.dataset_mode)
    resource_root = Path(args.resource_root).expanduser().resolve()
    processed = get_processed_layout(dataset_mode)

    captions_root = Path(args.captions_root).expanduser().resolve() if args.captions_root else processed.captions_root.resolve()
    visual_root = Path(args.visual_root).expanduser().resolve() if args.visual_root else processed.visual_root.resolve()
    audio_root = Path(args.audio_root).expanduser().resolve() if args.audio_root else processed.raw_audio_root.resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()

    fused_visual_roots: list[Path] = []
    if args.fused_visual_roots:
        fused_visual_roots = dedupe_existing_dirs([Path(raw.strip()) for raw in args.fused_visual_roots.split(",") if raw.strip()])
    elif args.visual_source != "split":
        fused_visual_roots = discover_fused_visual_roots(resource_root)

    visual_source = args.visual_source
    if visual_source == "auto":
        visual_source = "fused" if fused_visual_roots else "split"
    if visual_source == "fused" and not fused_visual_roots:
        raise FileNotFoundError("visual_source=fused but no TASCC fused roots were found")

    return FinalLayout(
        resource_root=resource_root,
        dataset_mode=dataset_mode,
        run_dir=run_dir,
        captions_root=captions_root,
        visual_root=visual_root,
        audio_root=audio_root,
        visual_source=visual_source,
        fused_visual_roots=fused_visual_roots,
    )


class FinalBartDataset(Dataset):
    def __init__(
        self,
        split: str,
        split_ids: list[str],
        captions_root: Path,
        visual_root: Path,
        audio_root: Path,
        visual_source: str,
        fused_visual_roots: list[Path],
        use_dino: bool,
        use_audio: bool,
        expand_captions: bool = False,
        debug_n: int | None = None,
    ) -> None:
        self.split = split
        self.visual_source = visual_source
        self.fused_visual_roots = fused_visual_roots
        self.use_dino = use_dino
        self.use_audio = use_audio
        self.expand_captions = expand_captions

        self.visual_root = visual_root
        self.audio_root = audio_root
        self.clip_dir = visual_root / "clip_embedding" / split
        self.dino_dir = visual_root / "Dinov2_embedding" / split
        self.audio_dir = audio_root / split
        self.caption_map = json.loads((captions_root / f"{split}_captions.json").read_text())
        self.anchor_target_map: dict[str, list[float]] | None = None

        self.video_ids: list[str] = []
        self.examples: list[tuple[str, str]] = []
        for video_id in split_ids:
            if video_id not in self.caption_map:
                continue
            if self.visual_source == "fused":
                if self._find_fused_path(video_id) is None:
                    continue
            else:
                if self._find_split_feature_path(
                    base_dir=self.visual_root / "clip_embedding",
                    split=self.split,
                    video_id=video_id,
                ) is None:
                    continue
                if self.use_dino and self._find_split_feature_path(
                    base_dir=self.visual_root / "Dinov2_embedding",
                    split=self.split,
                    video_id=video_id,
                ) is None:
                    continue
            if self.use_audio and self._find_split_feature_path(
                base_dir=self.audio_root,
                split=self.split,
                video_id=video_id,
            ) is None:
                continue
            self.video_ids.append(video_id)
            if self.expand_captions:
                self.examples.extend((video_id, caption) for caption in self.caption_map[video_id])

        if debug_n is not None:
            self.video_ids = self.video_ids[:debug_n]
            if self.expand_captions:
                keep = set(self.video_ids)
                self.examples = [example for example in self.examples if example[0] in keep]

        self.ground_truth = {
            video_id: [{"image_id": video_id, "caption": caption} for caption in self.caption_map[video_id]]
            for video_id in self.video_ids
        }
        if self.expand_captions:
            print(f"[{split}] {len(self.video_ids)} usable videos / {len(self.examples)} caption instances")
        else:
            print(f"[{split}] {len(self.video_ids)} usable videos")

    def _find_fused_path(self, video_id: str) -> Path | None:
        for root in self.fused_visual_roots:
            candidate = root / f"{video_id}.npy"
            if candidate.exists():
                return candidate
        return None

    def _find_split_feature_path(self, base_dir: Path, split: str, video_id: str) -> Path | None:
        preferred = base_dir / split / f"{video_id}.npy"
        if preferred.exists():
            return preferred

        direct = base_dir / f"{video_id}.npy"
        if direct.exists():
            return direct

        for candidate_split in ("train", "val", "test"):
            if candidate_split == split:
                continue
            candidate = base_dir / candidate_split / f"{video_id}.npy"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        if self.expand_captions:
            return len(self.examples)
        return len(self.video_ids)

    def __getitem__(self, idx: int):
        if self.expand_captions:
            video_id, caption = self.examples[idx]
        else:
            video_id = self.video_ids[idx]
            caption = random.choice(self.caption_map[video_id])
        if self.visual_source == "fused":
            fused_path = self._find_fused_path(video_id)
            if fused_path is None:
                raise FileNotFoundError(f"Missing fused feature for {video_id}")
            fused = np.load(fused_path)
            clip = torch.tensor(fused[:, :CLIP_DIM], dtype=torch.float32)
            if self.use_dino:
                dino = torch.tensor(fused[:, CLIP_DIM : CLIP_DIM + DINO_DIM], dtype=torch.float32)
            else:
                dino = torch.zeros(clip.shape[0], DINO_DIM, dtype=torch.float32)
        else:
            clip_path = self._find_split_feature_path(
                base_dir=self.visual_root / "clip_embedding",
                split=self.split,
                video_id=video_id,
            )
            if clip_path is None:
                raise FileNotFoundError(f"Missing CLIP feature for {video_id}")
            clip = torch.tensor(np.load(clip_path), dtype=torch.float32)
            if self.use_dino:
                dino_path = self._find_split_feature_path(
                    base_dir=self.visual_root / "Dinov2_embedding",
                    split=self.split,
                    video_id=video_id,
                )
                if dino_path is None:
                    raise FileNotFoundError(f"Missing DINO feature for {video_id}")
                dino = torch.tensor(np.load(dino_path), dtype=torch.float32)
            else:
                dino = torch.zeros(clip.shape[0], DINO_DIM, dtype=torch.float32)

        if self.use_audio:
            audio_path = self._find_split_feature_path(
                base_dir=self.audio_root,
                split=self.split,
                video_id=video_id,
            )
            if audio_path is None:
                raise FileNotFoundError(f"Missing audio feature for {video_id}")
            audio = torch.tensor(np.load(audio_path), dtype=torch.float32)
        else:
            audio = torch.zeros(1, AUDIO_DIM, dtype=torch.float32)

        references = self.caption_map[video_id]
        if self.anchor_target_map is not None and video_id in self.anchor_target_map:
            anchor_target = torch.tensor(self.anchor_target_map[video_id], dtype=torch.float32)
        else:
            anchor_target = torch.empty(0, dtype=torch.float32)
        return clip, dino, audio, caption, references, video_id, anchor_target

    def set_anchor_target_map(self, anchor_target_map: dict[str, list[float]] | None) -> None:
        self.anchor_target_map = anchor_target_map


def make_collate_fn(tokenizer: BartTokenizer, max_caption_len: int):
    def collate(batch):
        clips, dinos, audios, captions, references, video_ids, anchor_targets = zip(*batch)
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
        if all(target.numel() > 0 for target in anchor_targets):
            anchor_target_tensor = torch.stack(anchor_targets)
        else:
            anchor_target_tensor = None
        return (
            clips,
            dinos,
            audio_padded,
            audio_mask,
            labels,
            anchor_target_tensor,
            list(references),
            list(video_ids),
            list(captions),
        )

    return collate


def build_optimizer(model: nn.Module, encoder_lr: float, bart_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay_terms = ("bias", "LayerNorm.weight", "layer_norm.weight", "layernorm_embedding.weight")
    grouped: dict[tuple[float, float], list[torch.nn.Parameter]] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_bart = name.startswith("bart.")
        lr = bart_lr if is_bart else encoder_lr
        use_weight_decay = not any(term in name for term in no_decay_terms)
        wd = weight_decay if use_weight_decay else 0.0
        grouped.setdefault((lr, wd), []).append(parameter)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in grouped.items()
    ]
    return torch.optim.AdamW(param_groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_ratio: float,
):
    total_steps = max(steps_per_epoch * epochs, 0)
    if total_steps <= 0:
        return None
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def train_one_epoch_xe(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    grad_clip: float,
    grad_accum_steps: int,
    transfer_anchor_aux_weight: float = 0.0,
):
    model.train()
    total_loss = 0.0
    total_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step_index, (clips, dinos, audios, audio_mask, labels, anchor_targets, _, _, _) in enumerate(loader, start=1):
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        labels = labels.to(device)
        if anchor_targets is not None:
            anchor_targets = anchor_targets.to(device)

        out = model(
            clips,
            dinos,
            audios,
            audio_mask=audio_mask,
            labels=labels,
            transfer_anchor_targets=anchor_targets,
        )
        batch_loss = out.loss
        aux_loss = getattr(getattr(model, "transfer_adapter", None), "last_aux_loss", None)
        if transfer_anchor_aux_weight > 0 and aux_loss is not None:
            batch_loss = batch_loss + transfer_anchor_aux_weight * aux_loss
        loss = batch_loss / max(grad_accum_steps, 1)
        loss.backward()
        if step_index % grad_accum_steps == 0 or step_index == len(loader):
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
        total_loss += batch_loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for clips, dinos, audios, audio_mask, labels, anchor_targets, _, _, _ in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        labels = labels.to(device)
        if anchor_targets is not None:
            anchor_targets = anchor_targets.to(device)
        out = model(
            clips,
            dinos,
            audios,
            audio_mask=audio_mask,
            labels=labels,
            transfer_anchor_targets=anchor_targets,
        )
        total_loss += out.loss.item()
        total_batches += 1
    return total_loss / max(total_batches, 1)


def decode_sequences(tokenizer: BartTokenizer, token_ids: torch.Tensor) -> list[str]:
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in token_ids]


@torch.no_grad()
def generate_predictions(
    model,
    loader,
    tokenizer,
    device,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, list[dict[str, str]]]:
    model.eval()
    predictions: dict[str, list[dict[str, str]]] = {}
    for clips, dinos, audios, audio_mask, _, anchor_targets, _, video_ids, _ in loader:
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        if anchor_targets is not None:
            anchor_targets = anchor_targets.to(device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
        }
        if no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        generated = model.generate(
            clips,
            dinos,
            audios,
            audio_mask=audio_mask,
            transfer_anchor_targets=anchor_targets,
            **generate_kwargs,
        )
        captions = decode_sequences(tokenizer, generated)
        for video_id, caption in zip(video_ids, captions):
            predictions[video_id] = [{"image_id": video_id, "caption": caption}]
    return predictions


def evaluate_metrics(
    model,
    loader,
    tokenizer,
    device,
    num_beams: int,
    max_new_tokens: int,
    length_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
):
    predictions = generate_predictions(
        model,
        loader,
        tokenizer,
        device,
        num_beams,
        max_new_tokens,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    ground_truth = {video_id: loader.dataset.ground_truth[video_id] for video_id in predictions}
    ids = list(predictions.keys())
    metrics = COCOScorer().score(ground_truth, predictions, ids)
    return metrics, predictions


def load_model_weights(model: nn.Module, checkpoint_path: Path, device: torch.device, strict: bool = True) -> dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if strict:
        model.load_state_dict(state)
        return payload if isinstance(payload, dict) else {"model": state}

    model_state = model.state_dict()
    filtered = {
        name: value
        for name, value in state.items()
        if name in model_state and model_state[name].shape == value.shape
    }
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    print(
        "Partial checkpoint load:"
        f" loaded={len(filtered)} missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
    )
    return payload if isinstance(payload, dict) else {"model": state}


@torch.no_grad()
def embed_captions_with_bart(
    model: nn.Module,
    tokenizer: BartTokenizer,
    captions: list[str],
    max_caption_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    embedding_layer = model.bart.model.shared
    special_token_ids = {
        token_id
        for token_id in (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
        if token_id is not None
    }

    for start in range(0, len(captions), batch_size):
        batch = captions[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_caption_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).bool()
        valid_mask = attention_mask.clone()
        for token_id in special_token_ids:
            valid_mask &= input_ids.ne(token_id)
        fallback_mask = attention_mask & input_ids.ne(tokenizer.pad_token_id)
        empty_rows = ~valid_mask.any(dim=1)
        if empty_rows.any():
            valid_mask[empty_rows] = fallback_mask[empty_rows]

        token_embeddings = embedding_layer(input_ids)
        weights = valid_mask.to(token_embeddings.dtype).unsqueeze(-1)
        pooled = (token_embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        embeddings.append(F.normalize(pooled.detach().cpu(), dim=-1))

    return torch.cat(embeddings, dim=0)


def cosine_kmeans(
    embeddings: torch.Tensor,
    num_clusters: int,
    num_iters: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D tensor")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot cluster an empty caption set")

    embeddings = F.normalize(embeddings.float(), dim=-1)
    n_items = embeddings.shape[0]
    chosen = [0]
    min_distance = 1.0 - (embeddings @ embeddings[0].unsqueeze(1)).squeeze(1)
    for _ in range(1, min(num_clusters, n_items)):
        next_index = int(torch.argmax(min_distance).item())
        chosen.append(next_index)
        next_distance = 1.0 - (embeddings @ embeddings[next_index].unsqueeze(1)).squeeze(1)
        min_distance = torch.minimum(min_distance, next_distance)
    while len(chosen) < num_clusters:
        chosen.append(chosen[len(chosen) % n_items])

    centroids = F.normalize(embeddings[torch.tensor(chosen[:num_clusters])].clone(), dim=-1)
    labels = torch.zeros(n_items, dtype=torch.long)
    for _ in range(max(num_iters, 1)):
        labels = torch.argmax(embeddings @ centroids.T, dim=1)
        next_centroids = centroids.clone()
        for cluster_index in range(num_clusters):
            member_mask = labels.eq(cluster_index)
            if member_mask.any():
                next_centroids[cluster_index] = F.normalize(embeddings[member_mask].mean(dim=0), dim=0)
        if torch.allclose(next_centroids, centroids, atol=1e-6):
            centroids = next_centroids
            break
        centroids = next_centroids
    labels = torch.argmax(embeddings @ centroids.T, dim=1)
    return centroids, labels


def build_transfer_anchor_bank(
    model: nn.Module,
    tokenizer: BartTokenizer,
    train_dataset: FinalBartDataset,
    num_anchors: int,
    max_caption_len: int,
    device: torch.device,
    num_iters: int,
    smoothing: float,
) -> tuple[torch.Tensor, dict[str, list[float]], dict]:
    caption_records: list[tuple[str, str]] = []
    for video_id in train_dataset.video_ids:
        for caption in train_dataset.caption_map[video_id]:
            caption_records.append((video_id, caption))
    if not caption_records:
        raise ValueError("Cannot build transfer anchor bank: no training captions are available")

    captions = [caption for _, caption in caption_records]
    caption_embeddings = embed_captions_with_bart(
        model=model,
        tokenizer=tokenizer,
        captions=captions,
        max_caption_len=max_caption_len,
        device=device,
    )
    anchor_embeddings, labels = cosine_kmeans(caption_embeddings, num_anchors, num_iters)

    target_map: dict[str, list[float]] = {}
    label_offset = 0
    for video_id in train_dataset.video_ids:
        captions_for_video = train_dataset.caption_map[video_id]
        count = torch.full((num_anchors,), float(max(smoothing, 0.0)), dtype=torch.float32)
        for _ in captions_for_video:
            count[int(labels[label_offset].item())] += 1.0
            label_offset += 1
        target_map[video_id] = (count / count.sum().clamp_min(1e-8)).tolist()

    cluster_counts = torch.bincount(labels, minlength=num_anchors).tolist()
    summary = {
        "source_split": "train",
        "num_train_videos": len(train_dataset.video_ids),
        "num_train_captions": len(caption_records),
        "num_anchors": num_anchors,
        "num_iters": num_iters,
        "smoothing": smoothing,
        "cluster_counts": [int(value) for value in cluster_counts],
    }
    return anchor_embeddings, target_map, summary


def build_checkpoint_payload(
    stage: str,
    epoch: int,
    model,
    optimizer,
    scheduler,
    config: dict,
    history,
    val_loss: float | None = None,
    val_metrics: dict | None = None,
):
    return {
        "stage": stage,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "history": history,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
    }


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def sample_sequences_with_log_probs(
    model,
    clips: torch.Tensor,
    dinos: torch.Tensor | None,
    audios: torch.Tensor | None,
    audio_mask: torch.Tensor | None,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    pad_token_id: int,
    eos_token_id: int,
    transfer_anchor_targets: torch.Tensor | None = None,
):
    encoder_hidden_states, attention_mask = model._encode(
        clips,
        dinos,
        audios,
        audio_mask,
        transfer_anchor_targets=transfer_anchor_targets,
    )
    batch_size = clips.shape[0]
    decoder_input_ids = torch.full(
        (batch_size, 1),
        model.bart.config.decoder_start_token_id,
        dtype=torch.long,
        device=clips.device,
    )
    finished = torch.zeros(batch_size, dtype=torch.bool, device=clips.device)
    step_log_probs: list[torch.Tensor] = []

    for _ in range(max_new_tokens):
        out = model.bart(
            attention_mask=attention_mask,
            encoder_outputs=(encoder_hidden_states,),
            decoder_input_ids=decoder_input_ids,
        )
        next_logits = out.logits[:, -1, :]
        next_logits = next_logits / max(temperature, 1e-6)
        filtered_logits = top_p_filter(next_logits, top_p)
        log_probs = torch.log_softmax(filtered_logits, dim=-1)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        token_log_prob = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)

        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)
        token_log_prob = torch.where(finished, torch.zeros_like(token_log_prob), token_log_prob)
        decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(1)], dim=1)
        step_log_probs.append(token_log_prob)

        finished = finished | next_tokens.eq(eos_token_id)
        if bool(finished.all()):
            break

    if step_log_probs:
        sequence_log_probs = torch.stack(step_log_probs, dim=1).sum(dim=1)
    else:
        sequence_log_probs = torch.zeros(batch_size, device=clips.device)
    return decoder_input_ids, sequence_log_probs


def cider_scores_from_batch(references: list[list[str]], hypotheses: list[str], video_ids: list[str]) -> np.ndarray:
    ground_truth = {video_id: refs for video_id, refs in zip(video_ids, references)}
    predictions = {video_id: [caption] for video_id, caption in zip(video_ids, hypotheses)}
    tokenized_gt = simple_tokenize(ground_truth)
    tokenized_pred = simple_tokenize(predictions)
    scorer = CiderScorer(n=4, sigma=6.0)
    for video_id in video_ids:
        scorer += (tokenized_pred[video_id][0], tokenized_gt[video_id])
    scorer.compute_doc_freq()
    scorer.ref_len = np.log(float(len(scorer.crefs)))
    scores = scorer.compute_cider()
    return np.asarray(scores, dtype=np.float32)


def train_one_epoch_scst(
    model,
    loader,
    tokenizer,
    optimizer,
    scheduler,
    device,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
    grad_clip: float,
    xe_weight: float,
    grad_accum_steps: int,
    transfer_anchor_aux_weight: float = 0.0,
):
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    total_sample_cider = 0.0
    total_greedy_cider = 0.0
    total_batches = 0
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    optimizer.zero_grad(set_to_none=True)
    for step_index, (clips, dinos, audios, audio_mask, labels, anchor_targets, references, video_ids, _) in enumerate(loader, start=1):
        clips = clips.to(device)
        dinos = dinos.to(device)
        audios = audios.to(device)
        audio_mask = audio_mask.to(device)
        labels = labels.to(device)
        if anchor_targets is not None:
            anchor_targets = anchor_targets.to(device)

        model.eval()
        sampled_ids, sampled_log_probs = sample_sequences_with_log_probs(
            model=model,
            clips=clips,
            dinos=dinos,
            audios=audios,
            audio_mask=audio_mask,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            transfer_anchor_targets=anchor_targets,
        )
        sampled_captions = decode_sequences(tokenizer, sampled_ids)

        with torch.no_grad():
            greedy_ids = model.generate(
                clips,
                dinos,
                audios,
                audio_mask=audio_mask,
                transfer_anchor_targets=anchor_targets,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        greedy_captions = decode_sequences(tokenizer, greedy_ids)
        model.train()

        sampled_rewards = cider_scores_from_batch(references, sampled_captions, video_ids)
        greedy_rewards = cider_scores_from_batch(references, greedy_captions, video_ids)
        reward_advantage = torch.tensor(sampled_rewards - greedy_rewards, dtype=sampled_log_probs.dtype, device=device)

        scst_loss = -(reward_advantage * sampled_log_probs).mean()
        loss = scst_loss
        if xe_weight > 0:
            xe_out = model(
                clips,
                dinos,
                audios,
                audio_mask=audio_mask,
                labels=labels,
                transfer_anchor_targets=anchor_targets,
            )
            loss = loss + xe_weight * xe_out.loss
            aux_loss = getattr(getattr(model, "transfer_adapter", None), "last_aux_loss", None)
            if transfer_anchor_aux_weight > 0 and aux_loss is not None:
                loss = loss + transfer_anchor_aux_weight * aux_loss

        scaled_loss = loss / max(grad_accum_steps, 1)
        scaled_loss.backward()
        if step_index % grad_accum_steps == 0 or step_index == len(loader):
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item())
        total_reward += float(reward_advantage.mean().item())
        total_sample_cider += float(sampled_rewards.mean())
        total_greedy_cider += float(greedy_rewards.mean())
        total_batches += 1

    divisor = max(total_batches, 1)
    return {
        "loss": total_loss / divisor,
        "reward": total_reward / divisor,
        "sample_cider": total_sample_cider / divisor,
        "greedy_cider": total_greedy_cider / divisor,
    }


def build_model(args: argparse.Namespace, use_dino: bool, use_audio: bool, device: torch.device):
    architecture = args.architecture.lower()
    if architecture == "stable":
        if args.decoder_train_mode not in {"freeze", "full", "partial"}:
            raise ValueError(
                "Stable architecture only supports decoder_train_mode=freeze, partial, or full."
            )
        model = FlexibleBartCaptioningModel(
            use_dino=use_dino,
            use_audio=use_audio,
            encoder_d_model=args.encoder_d_model,
            n_heads=args.n_heads,
            bart_model_name=args.bart_model_name,
            decoder_train_mode=args.decoder_train_mode,
            decoder_train_last_n_layers=args.decoder_train_last_n_layers,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
            max_audio_tokens=args.max_audio_tokens,
            transfer_adapter=args.transfer_adapter,
            transfer_num_anchors=args.transfer_num_anchors,
            transfer_num_tokens=args.transfer_num_tokens,
            transfer_bottleneck_dim=args.transfer_bottleneck_dim,
            transfer_dropout=args.transfer_dropout,
            transfer_gate_init=args.transfer_gate_init,
            transfer_residual_scale=args.transfer_residual_scale,
            transfer_anchor_temperature=args.transfer_anchor_temperature,
        )
    elif architecture == "experimental":
        model = FinalBartCaptioningModel(
            use_dino=use_dino,
            use_audio=use_audio,
            encoder_d_model=args.encoder_d_model,
            n_heads=args.n_heads,
            bart_model_name=args.bart_model_name,
            decoder_train_mode=args.decoder_train_mode,
            decoder_train_last_n_layers=args.decoder_train_last_n_layers,
            max_visual_positions=args.max_visual_positions,
            audio_summary_tokens=args.audio_summary_tokens,
            feature_dropout=args.feature_dropout,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
            transfer_adapter=args.transfer_adapter,
            transfer_num_anchors=args.transfer_num_anchors,
            transfer_num_tokens=args.transfer_num_tokens,
            transfer_bottleneck_dim=args.transfer_bottleneck_dim,
            transfer_dropout=args.transfer_dropout,
            transfer_gate_init=args.transfer_gate_init,
            transfer_residual_scale=args.transfer_residual_scale,
            transfer_anchor_temperature=args.transfer_anchor_temperature,
        )
    else:
        raise ValueError(f"Unsupported architecture: {args.architecture}")
    return model.to(device)


def freeze_non_bart_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("bart."):
            parameter.requires_grad = False


def freeze_for_transfer_adapter(
    model: nn.Module,
    train_lm_head: bool = False,
    train_visual_projection: bool = False,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if "transfer_adapter" in name:
            parameter.requires_grad = True
        if train_visual_projection and (
            name.startswith("proj.")
            or name.startswith("input_norm.")
            or name.startswith("modality_type_embeddings.")
        ):
            parameter.requires_grad = True

    if train_lm_head and hasattr(model, "bart"):
        model.bart.model.shared.weight.requires_grad = True
        for parameter in model.bart.lm_head.parameters():
            parameter.requires_grad = True


def print_layout(layout: FinalLayout) -> None:
    print("Resolved training layout:")
    print(f"  dataset_mode   : {layout.dataset_mode}")
    print(f"  captions_root  : {layout.captions_root}")
    print(f"  visual_root    : {layout.visual_root}")
    print(f"  audio_root     : {layout.audio_root}")
    print(f"  visual_source  : {layout.visual_source}")
    if layout.fused_visual_roots:
        for index, root in enumerate(layout.fused_visual_roots, start=1):
            print(f"  fused_root[{index}] : {root}")


def parse_args():
    parser = argparse.ArgumentParser(description="Final BART training path with val-CIDEr checkpointing and SCST.")
    parser.add_argument("--preset", choices=["none", *TUNING_PRESETS.keys()], default="none")
    parser.add_argument("--resource-root", default=str(PROJECT_ROOT))
    parser.add_argument("--dataset-mode", choices=["subset", "full"], default="full")
    parser.add_argument("--modalities", choices=["clip", "clip_dino", "clip_dino_audio"], default="clip_dino_audio")
    parser.add_argument("--architecture", choices=["stable", "experimental"], default="stable")
    parser.add_argument("--visual-source", choices=["auto", "split", "fused"], default="auto")
    parser.add_argument("--captions-root", default=None)
    parser.add_argument("--visual-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--fused-visual-roots", default=None)
    parser.add_argument("--run-dir", default="outputs/final_bart_full")
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--bart-model-name", default="facebook/bart-base")
    parser.add_argument("--decoder-train-mode", choices=["full", "freeze", "partial"], default="freeze")
    parser.add_argument("--decoder-train-last-n-layers", type=int, default=2)
    parser.add_argument("--encoder-d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--max-visual-positions", type=int, default=40)
    parser.add_argument("--audio-summary-tokens", type=int, default=4)
    parser.add_argument("--feature-dropout", type=float, default=0.1)
    parser.add_argument("--max-audio-tokens", type=int, default=None)
    parser.add_argument("--transfer-adapter", choices=["none", "target_residual_prefix", "target_anchor_prefix"], default="none")
    parser.add_argument("--transfer-num-anchors", type=int, default=12)
    parser.add_argument("--transfer-num-tokens", type=int, default=4)
    parser.add_argument("--transfer-bottleneck-dim", type=int, default=64)
    parser.add_argument("--transfer-dropout", type=float, default=0.1)
    parser.add_argument("--transfer-gate-init", type=float, default=-4.0)
    parser.add_argument("--transfer-residual-scale", type=float, default=1.0)
    parser.add_argument("--transfer-anchor-temperature", type=float, default=0.08)
    parser.add_argument("--transfer-anchor-smoothing", type=float, default=0.02)
    parser.add_argument("--transfer-anchor-iters", type=int, default=25)
    parser.add_argument("--transfer-anchor-aux-weight", type=float, default=0.0)
    parser.add_argument("--transfer-train-only", action="store_true")
    parser.add_argument("--transfer-train-lm-head", action="store_true")
    parser.add_argument("--transfer-train-visual-proj", action="store_true")
    parser.add_argument("--xe-epochs", type=int, default=20)
    parser.add_argument("--scst-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--scst-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--scst-grad-accum-steps", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--bart-lr", type=float, default=2e-5)
    parser.add_argument("--scst-encoder-lr", type=float, default=2e-5)
    parser.add_argument("--scst-bart-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--scst-warmup-ratio", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scst-grad-clip", type=float, default=1.0)
    parser.add_argument("--max-caption-len", type=int, default=40)
    parser.add_argument("--val-num-beams", type=int, default=4)
    parser.add_argument("--test-num-beams", type=int, default=4)
    parser.add_argument("--val-length-penalty", type=float, default=1.0)
    parser.add_argument("--test-length-penalty", type=float, default=1.0)
    parser.add_argument("--val-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--test-no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--scst-top-p", type=float, default=0.9)
    parser.add_argument("--scst-temperature", type=float, default=1.0)
    parser.add_argument("--scst-xe-weight", type=float, default=0.05)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-scst", action="store_true")
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--expand-train-captions", action="store_true")
    parser.add_argument("--expand-val-loss-captions", action="store_true")
    parser.add_argument("--allow-partial-init", action="store_true")
    parser.add_argument("--freeze-non-bart", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    return apply_tuning_preset(args, parser)


def main() -> int:
    args = parse_args()
    layout = resolve_layout(args)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    print_layout(layout)

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Device: {device}")

    use_clip, use_dino, use_audio = parse_modalities(args.modalities)
    tokenizer = BartTokenizer.from_pretrained(args.bart_model_name)
    split_ids = get_split_video_ids_from_captions(layout.captions_root)

    train_ds = FinalBartDataset(
        split="train",
        split_ids=split_ids["train"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        visual_source=layout.visual_source,
        fused_visual_roots=layout.fused_visual_roots,
        use_dino=use_dino,
        use_audio=use_audio,
        expand_captions=args.expand_train_captions,
        debug_n=args.debug_n,
    )
    val_loss_ds = FinalBartDataset(
        split="val",
        split_ids=split_ids["val"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        visual_source=layout.visual_source,
        fused_visual_roots=layout.fused_visual_roots,
        use_dino=use_dino,
        use_audio=use_audio,
        expand_captions=args.expand_val_loss_captions,
        debug_n=args.debug_n,
    )
    val_ds = FinalBartDataset(
        split="val",
        split_ids=split_ids["val"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        visual_source=layout.visual_source,
        fused_visual_roots=layout.fused_visual_roots,
        use_dino=use_dino,
        use_audio=use_audio,
        debug_n=args.debug_n,
    )
    test_ds = FinalBartDataset(
        split="test",
        split_ids=split_ids["test"],
        captions_root=layout.captions_root,
        visual_root=layout.visual_root,
        audio_root=layout.audio_root,
        visual_source=layout.visual_source,
        fused_visual_roots=layout.fused_visual_roots,
        use_dino=use_dino,
        use_audio=use_audio,
        debug_n=args.debug_n,
    )

    collate_fn = make_collate_fn(tokenizer, args.max_caption_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loss_loader = DataLoader(val_loss_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    scst_train_loader = DataLoader(train_ds, batch_size=args.scst_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = build_model(args, use_dino=use_dino, use_audio=use_audio, device=device)
    init_payload = None
    if args.init_from_checkpoint:
        checkpoint_path = Path(args.init_from_checkpoint).expanduser().resolve()
        print(f"Loading initial weights from: {checkpoint_path}")
        init_payload = load_model_weights(model, checkpoint_path, device, strict=not args.allow_partial_init)
    transfer_anchor_summary = None
    if args.transfer_adapter == "target_anchor_prefix":
        if args.transfer_num_anchors <= 0:
            raise ValueError("--transfer-num-anchors must be positive for target_anchor_prefix")
        anchor_embeddings, anchor_target_map, transfer_anchor_summary = build_transfer_anchor_bank(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            num_anchors=args.transfer_num_anchors,
            max_caption_len=args.max_caption_len,
            device=device,
            num_iters=args.transfer_anchor_iters,
            smoothing=args.transfer_anchor_smoothing,
        )
        model.set_transfer_anchor_bank(anchor_embeddings.to(device))
        train_ds.set_anchor_target_map(anchor_target_map)
        save_json(layout.run_dir / "transfer_anchor_summary.json", transfer_anchor_summary)
        print(
            "Transfer anchors:"
            f" captions={transfer_anchor_summary['num_train_captions']} "
            f"anchors={transfer_anchor_summary['num_anchors']} "
            f"cluster_counts={transfer_anchor_summary['cluster_counts']}"
        )
    if args.transfer_train_only:
        if args.transfer_adapter == "none":
            raise ValueError("--transfer-train-only requires --transfer-adapter to be enabled")
        if args.freeze_non_bart:
            raise ValueError("--transfer-train-only cannot be combined with --freeze-non-bart")
        freeze_for_transfer_adapter(
            model,
            train_lm_head=args.transfer_train_lm_head,
            train_visual_projection=args.transfer_train_visual_proj,
        )
    if args.freeze_non_bart:
        freeze_non_bart_parameters(model)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    config = {
        "resource_root": str(layout.resource_root),
        "preset": args.preset,
        "dataset_mode": layout.dataset_mode,
        "run_dir": str(layout.run_dir),
        "captions_root": str(layout.captions_root),
        "visual_root": str(layout.visual_root),
        "audio_root": str(layout.audio_root),
        "visual_source": layout.visual_source,
        "fused_visual_roots": [str(path) for path in layout.fused_visual_roots],
        "modalities": args.modalities,
        "architecture": args.architecture,
        "init_from_checkpoint": str(Path(args.init_from_checkpoint).expanduser().resolve()) if args.init_from_checkpoint else None,
        "bart_model_name": args.bart_model_name,
        "decoder_train_mode": args.decoder_train_mode,
        "decoder_train_last_n_layers": args.decoder_train_last_n_layers,
        "encoder_d_model": args.encoder_d_model,
        "n_heads": args.n_heads,
        "max_visual_positions": args.max_visual_positions,
        "audio_summary_tokens": args.audio_summary_tokens,
        "feature_dropout": args.feature_dropout,
        "max_audio_tokens": args.max_audio_tokens,
        "transfer_adapter": args.transfer_adapter,
        "transfer_num_anchors": args.transfer_num_anchors,
        "transfer_num_tokens": args.transfer_num_tokens,
        "transfer_bottleneck_dim": args.transfer_bottleneck_dim,
        "transfer_dropout": args.transfer_dropout,
        "transfer_gate_init": args.transfer_gate_init,
        "transfer_residual_scale": args.transfer_residual_scale,
        "transfer_anchor_temperature": args.transfer_anchor_temperature,
        "transfer_anchor_smoothing": args.transfer_anchor_smoothing,
        "transfer_anchor_iters": args.transfer_anchor_iters,
        "transfer_anchor_aux_weight": args.transfer_anchor_aux_weight,
        "transfer_train_only": args.transfer_train_only,
        "transfer_train_lm_head": args.transfer_train_lm_head,
        "transfer_train_visual_proj": args.transfer_train_visual_proj,
        "transfer_anchor_summary": transfer_anchor_summary,
        "xe_epochs": args.xe_epochs,
        "scst_epochs": args.scst_epochs,
        "batch_size": args.batch_size,
        "scst_batch_size": args.scst_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "scst_grad_accum_steps": args.scst_grad_accum_steps,
        "encoder_lr": args.encoder_lr,
        "bart_lr": args.bart_lr,
        "scst_encoder_lr": args.scst_encoder_lr,
        "scst_bart_lr": args.scst_bart_lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "scst_warmup_ratio": args.scst_warmup_ratio,
        "grad_clip": args.grad_clip,
        "scst_grad_clip": args.scst_grad_clip,
        "max_caption_len": args.max_caption_len,
        "val_num_beams": args.val_num_beams,
        "test_num_beams": args.test_num_beams,
        "val_length_penalty": args.val_length_penalty,
        "test_length_penalty": args.test_length_penalty,
        "val_no_repeat_ngram_size": args.val_no_repeat_ngram_size,
        "test_no_repeat_ngram_size": args.test_no_repeat_ngram_size,
        "scst_top_p": args.scst_top_p,
        "scst_temperature": args.scst_temperature,
        "scst_xe_weight": args.scst_xe_weight,
        "gradient_checkpointing": not args.disable_gradient_checkpointing,
        "seed": args.seed,
        "device": str(device),
        "expand_train_captions": args.expand_train_captions,
        "expand_val_loss_captions": args.expand_val_loss_captions,
        "freeze_non_bart": args.freeze_non_bart,
        "train_size": len(train_ds),
        "val_loss_size": len(val_loss_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "init_stage": init_payload.get("stage") if init_payload else None,
        "init_epoch": init_payload.get("epoch") if init_payload else None,
    }
    save_json(layout.run_dir / "config.json", config)

    xe_optimizer = build_optimizer(model, args.encoder_lr, args.bart_lr, args.weight_decay)
    xe_scheduler = build_scheduler(
        xe_optimizer,
        math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)),
        args.xe_epochs,
        args.warmup_ratio,
    )

    history_xe: list[dict] = []
    best_val_loss = float("inf")
    best_val_cider = float("-inf")
    for epoch in range(1, args.xe_epochs + 1):
        train_loss = train_one_epoch_xe(
            model,
            train_loader,
            xe_optimizer,
            xe_scheduler,
            device,
            args.grad_clip,
            args.grad_accum_steps,
            transfer_anchor_aux_weight=args.transfer_anchor_aux_weight,
        )
        val_loss = evaluate_loss(model, val_loss_loader, device)
        val_metrics, _ = evaluate_metrics(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            num_beams=args.val_num_beams,
            max_new_tokens=args.max_caption_len,
            length_penalty=args.val_length_penalty,
            no_repeat_ngram_size=args.val_no_repeat_ngram_size,
        )
        val_cider = float(val_metrics.get("CIDEr", float("-inf")))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_metrics": val_metrics,
        }
        history_xe.append(record)
        save_json(layout.run_dir / "history_xe.json", history_xe)

        print(
            f"[XE] Epoch {epoch:03d}/{args.xe_epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} CIDEr={val_cider:.4f} "
            f"Bleu_4={val_metrics.get('Bleu_4', 0.0):.4f}"
        )

        latest_payload = build_checkpoint_payload(
            stage="xe",
            epoch=epoch,
            model=model,
            optimizer=xe_optimizer,
            scheduler=xe_scheduler,
            config=config,
            history=history_xe,
            val_loss=val_loss,
            val_metrics=val_metrics,
        )
        torch.save(latest_payload, layout.run_dir / "latest_xe.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(latest_payload, layout.run_dir / "best_loss_xe.pt")
        if val_cider > best_val_cider:
            best_val_cider = val_cider
            torch.save(latest_payload, layout.run_dir / "best_cider_xe.pt")

    def evaluate_named_checkpoint(checkpoint_name: str) -> tuple[dict, dict, dict]:
        checkpoint_path = layout.run_dir / checkpoint_name
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        metrics, predictions = evaluate_metrics(
            model=model,
            loader=test_loader,
            tokenizer=tokenizer,
            device=device,
            num_beams=args.test_num_beams,
            max_new_tokens=args.max_caption_len,
            length_penalty=args.test_length_penalty,
            no_repeat_ngram_size=args.test_no_repeat_ngram_size,
        )
        stem = checkpoint_name.replace(".pt", "")
        save_json(layout.run_dir / f"test_metrics_{stem}.json", metrics)
        save_json(layout.run_dir / f"test_predictions_{stem}.json", predictions)
        return payload, metrics, predictions

    best_loss_xe_payload, xe_test_metrics_best_loss, xe_test_predictions_best_loss = evaluate_named_checkpoint("best_loss_xe.pt")
    best_xe_payload, xe_test_metrics_best_cider, xe_test_predictions_best_cider = evaluate_named_checkpoint("best_cider_xe.pt")

    # Preserve the historical artifact names for downstream scripts.
    xe_test_metrics = xe_test_metrics_best_cider
    xe_test_predictions = xe_test_predictions_best_cider
    save_json(layout.run_dir / "test_metrics_xe.json", xe_test_metrics)
    save_json(layout.run_dir / "test_predictions_xe.json", xe_test_predictions)
    print(
        f"[XE] Test CIDEr={xe_test_metrics.get('CIDEr', 0.0):.4f} "
        f"Bleu_4={xe_test_metrics.get('Bleu_4', 0.0):.4f}"
    )

    final_summary = {
        "xe": {
            "best_epoch": int(best_xe_payload["epoch"]),
            "best_val_loss": float(best_xe_payload.get("val_loss", best_val_loss)),
            "best_val_cider": float(best_xe_payload.get("val_metrics", {}).get("CIDEr", best_val_cider)),
            "test_metrics": xe_test_metrics,
            "best_loss_epoch": int(best_loss_xe_payload["epoch"]),
            "test_metrics_best_loss": xe_test_metrics_best_loss,
            "test_metrics_best_cider": xe_test_metrics_best_cider,
        }
    }

    if not args.skip_scst and args.scst_epochs > 0:
        model.load_state_dict(best_xe_payload["model"])
        scst_optimizer = build_optimizer(model, args.scst_encoder_lr, args.scst_bart_lr, args.weight_decay)
        scst_scheduler = build_scheduler(
            scst_optimizer,
            math.ceil(len(scst_train_loader) / max(args.scst_grad_accum_steps, 1)),
            args.scst_epochs,
            args.scst_warmup_ratio,
        )

        history_scst: list[dict] = []
        best_scst_cider = float("-inf")
        for epoch in range(1, args.scst_epochs + 1):
            scst_stats = train_one_epoch_scst(
                model=model,
                loader=scst_train_loader,
                tokenizer=tokenizer,
                optimizer=scst_optimizer,
                scheduler=scst_scheduler,
                device=device,
                max_new_tokens=args.max_caption_len,
                top_p=args.scst_top_p,
                temperature=args.scst_temperature,
                grad_clip=args.scst_grad_clip,
                xe_weight=args.scst_xe_weight,
                grad_accum_steps=args.scst_grad_accum_steps,
                transfer_anchor_aux_weight=args.transfer_anchor_aux_weight,
            )
            val_loss = evaluate_loss(model, val_loss_loader, device)
            val_metrics, _ = evaluate_metrics(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                num_beams=args.val_num_beams,
                max_new_tokens=args.max_caption_len,
                length_penalty=args.val_length_penalty,
                no_repeat_ngram_size=args.val_no_repeat_ngram_size,
            )
            val_cider = float(val_metrics.get("CIDEr", float("-inf")))
            record = {
                "epoch": epoch,
                "train_scst": scst_stats,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
            }
            history_scst.append(record)
            save_json(layout.run_dir / "history_scst.json", history_scst)

            print(
                f"[SCST] Epoch {epoch:03d}/{args.scst_epochs} "
                f"loss={scst_stats['loss']:.4f} reward={scst_stats['reward']:.4f} "
                f"sampleCIDEr={scst_stats['sample_cider']:.4f} baselineCIDEr={scst_stats['greedy_cider']:.4f} "
                f"valCIDEr={val_cider:.4f}"
            )

            latest_payload = build_checkpoint_payload(
                stage="scst",
                epoch=epoch,
                model=model,
                optimizer=scst_optimizer,
                scheduler=scst_scheduler,
                config=config,
                history=history_scst,
                val_loss=val_loss,
                val_metrics=val_metrics,
            )
            torch.save(latest_payload, layout.run_dir / "latest_scst.pt")
            if val_cider > best_scst_cider:
                best_scst_cider = val_cider
                torch.save(latest_payload, layout.run_dir / "best_scst.pt")

        best_scst_path = layout.run_dir / "best_scst.pt"
        if best_scst_path.exists():
            best_scst_payload = torch.load(best_scst_path, map_location=device, weights_only=False)
            model.load_state_dict(best_scst_payload["model"])
            scst_test_metrics, scst_test_predictions = evaluate_metrics(
                model=model,
                loader=test_loader,
                tokenizer=tokenizer,
                device=device,
                num_beams=args.test_num_beams,
                max_new_tokens=args.max_caption_len,
                length_penalty=args.test_length_penalty,
                no_repeat_ngram_size=args.test_no_repeat_ngram_size,
            )
            save_json(layout.run_dir / "test_metrics_scst.json", scst_test_metrics)
            save_json(layout.run_dir / "test_predictions_scst.json", scst_test_predictions)
            print(
                f"[SCST] Test CIDEr={scst_test_metrics.get('CIDEr', 0.0):.4f} "
                f"Bleu_4={scst_test_metrics.get('Bleu_4', 0.0):.4f}"
            )
            final_summary["scst"] = {
                "best_epoch": int(best_scst_payload["epoch"]),
                "best_val_cider": float(best_scst_payload.get("val_metrics", {}).get("CIDEr", best_scst_cider)),
                "test_metrics": scst_test_metrics,
            }

    save_json(layout.run_dir / "summary.json", final_summary)
    print(json.dumps(final_summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
