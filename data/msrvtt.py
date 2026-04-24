import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = PROJECT_ROOT / "dataset" / "MSR-VTT"


SUBSET_ALIASES = {"subset", "2500", "downsampled", "downsampled_2500"}
FULL_ALIASES = {"full", "full_msrvtt", "msrvtt"}


@dataclass(frozen=True)
class ProcessedLayout:
    mode: str
    processed_root: Path
    captions_root: Path
    visual_root: Path
    audio_root: Path
    multimodal_root: Path
    vocab_path: Path
    raw_clip_root: Path
    raw_dino_root: Path
    raw_audio_root: Path
    approach1_root: Path
    approach2_root: Path
    approach1_seq_root: Path
    approach2_seq_root: Path


def normalize_dataset_mode(mode: str | None) -> str:
    value = (mode or "subset").strip().lower()
    if value in SUBSET_ALIASES:
        return "subset"
    if value in FULL_ALIASES:
        return "full"
    raise ValueError(f"Unsupported dataset mode: {mode}")


def get_processed_layout(mode: str | None = None) -> ProcessedLayout:
    normalized = normalize_dataset_mode(mode)
    processed_root = DATA_ROOT / "processed" if normalized == "subset" else DATA_ROOT / "processed_full"
    captions_root = processed_root / "captions"
    visual_root = processed_root / "visual"
    audio_root = processed_root / "audio"
    multimodal_root = processed_root / "multimodal"
    return ProcessedLayout(
        mode=normalized,
        processed_root=processed_root,
        captions_root=captions_root,
        visual_root=visual_root,
        audio_root=audio_root,
        multimodal_root=multimodal_root,
        vocab_path=captions_root / "vocabulary.json",
        raw_clip_root=visual_root / "clip_embedding",
        raw_dino_root=visual_root / "Dinov2_embedding",
        raw_audio_root=audio_root / "vggish_embeddings",
        approach1_root=multimodal_root / "approach1_visual_cross_attn_audio_concat",
        approach2_root=multimodal_root / "approach2_trimodal_cross_attn",
        approach1_seq_root=multimodal_root / "approach1_visual_cross_attn_audio_concat_sequence",
        approach2_seq_root=multimodal_root / "approach2_trimodal_cross_attn_sequence",
    )


def get_metadata_path(mode: str | None = None) -> Path:
    normalized = normalize_dataset_mode(mode)
    if normalized == "subset":
        path = DATASET_ROOT / "downsampled_2500.json"
    else:
        path = DATASET_ROOT / "train_val_videodatainfo.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata json for {normalized}: {path}")
    return path


def get_eval_metadata_path(mode: str | None = None) -> Path:
    normalized = normalize_dataset_mode(mode)
    if normalized == "full":
        path = DATASET_ROOT / "test_videodatainfo.json"
    else:
        path = DATASET_ROOT / "downsampled_2500.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation metadata json for {normalized}: {path}")
    return path


def load_metadata(mode: str | None = None) -> dict:
    return json.loads(get_metadata_path(mode).read_text())


def canonical_split_name(split: str) -> str:
    if split == "validate":
        return "val"
    return split


def get_split_video_ids(mode: str | None = None) -> dict[str, list[str]]:
    data = load_metadata(mode)
    split_map = {"train": [], "val": [], "test": []}
    for video in data.get("videos", []):
        split = canonical_split_name(video.get("split", ""))
        if split not in split_map:
            continue
        split_map[split].append(video["video_id"])
    for split in split_map:
        split_map[split] = sorted(split_map[split], key=video_number)
    return split_map


def video_number(video_id: str) -> int:
    return int(video_id.replace("video", ""))


def get_video_dir_candidates(mode: str | None = None) -> list[Path]:
    normalized = normalize_dataset_mode(mode)
    if normalized == "subset":
        return [
            DATA_ROOT / "raw" / "downsampled_2500_videos",
            DATASET_ROOT / "downsampled_2500_videos",
        ]
    return [
        DATASET_ROOT / "full_dataset",
        DATASET_ROOT / "full_dataset" / "TrainValVideo",
        DATASET_ROOT / "TrainValVideo",
        DATASET_ROOT / "TestVideo",
        Path("/Users/aglooney03/Downloads/TestVideo 2"),
        Path("/Users/aglooney03/Downloads/TestVideo"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/full_dataset"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/TrainValVideo"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/TestVideo"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/full_dataset"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/TrainValVideo"),
        Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/TestVideo"),
    ]


def _iter_existing_dirs(dirs: Iterable[Path]) -> list[Path]:
    seen = set()
    existing = []
    for directory in dirs:
        directory = Path(directory)
        if directory.exists():
            resolved = directory.resolve()
            if resolved not in seen:
                seen.add(resolved)
                existing.append(directory)
    return existing


def build_video_index(mode: str | None = None, extra_video_dirs: Iterable[str | Path] | None = None) -> tuple[dict[str, Path], list[Path]]:
    candidate_dirs = get_video_dir_candidates(mode)
    if extra_video_dirs:
        candidate_dirs = list(extra_video_dirs) + candidate_dirs
    existing_dirs = _iter_existing_dirs(candidate_dirs)
    if not existing_dirs:
        raise FileNotFoundError(
            f"Could not find any video directories for dataset mode {normalize_dataset_mode(mode)}. "
            f"Checked: {', '.join(str(p) for p in candidate_dirs)}"
        )

    index: dict[str, Path] = {}
    for directory in existing_dirs:
        for path in directory.rglob("video*.mp4"):
            index.setdefault(path.stem, path)
    return index, existing_dirs


def get_splits(
    dataset_mode: str | None = None,
    strict: bool = True,
    extra_video_dirs: Iterable[str | Path] | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    split_ids = get_split_video_ids(dataset_mode)
    index, existing_dirs = build_video_index(dataset_mode, extra_video_dirs=extra_video_dirs)

    missing: dict[str, list[str]] = {}
    resolved: dict[str, list[Path]] = {}
    for split, video_ids in split_ids.items():
        found = [index[video_id] for video_id in video_ids if video_id in index]
        absent = [video_id for video_id in video_ids if video_id not in index]
        resolved[split] = sorted(found, key=lambda path: video_number(path.stem))
        if absent:
            missing[split] = absent

    if strict and missing:
        summary = ", ".join(f"{split}={len(ids)}" for split, ids in missing.items())
        raise FileNotFoundError(
            f"Missing videos for dataset mode {normalize_dataset_mode(dataset_mode)} ({summary}). "
            f"Video directories searched: {', '.join(str(p) for p in existing_dirs)}"
        )

    return resolved["train"], resolved["val"], resolved["test"]
