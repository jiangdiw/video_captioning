#!/usr/bin/env python3
"""Run TASCC on one video and write the selected keyframe images to a directory."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "novel_keyframe_extractor.py").exists() else SCRIPT_DIR.parent
EXTRACTOR_PATH = PROJECT_ROOT / "novel_keyframe_extractor.py"

DEFAULT_VIDEO_DIR_CANDIDATES = [
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/TrainValVideo"),
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v2/AY25-1/MA498/MSR-VTT/TestVideo"),
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/TrainValVideo"),
    Path("/Volumes/Seagate Portable Drive/west_point_backup_v3/AY25-1/MA498/MSR-VTT/TestVideo"),
    Path("dataset/MSR-VTT/full_dataset/TrainValVideo"),
    Path("dataset/MSR-VTT/full_dataset/TestVideo"),
    Path("dataset/MSR-VTT/TrainValVideo"),
    Path("dataset/MSR-VTT/TestVideo"),
]

DEFAULT_OUTPUT_ROOT = Path("datas/keyframe_visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write TASCC-selected keyframes for a single video into a directory."
    )
    parser.add_argument(
        "video_id",
        nargs="?",
        default=None,
        help="Video id such as video123 or video123.mp4.",
    )
    parser.add_argument("--video", type=Path, default=None, help="Absolute or relative path to a specific video file.")
    parser.add_argument(
        "--video-id",
        dest="video_id_flag",
        type=str,
        default=None,
        help="Video id such as video123 or video123.mp4.",
    )
    parser.add_argument("--video-dir", type=Path, default=None, help="Directory to search when using --video-id.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where keyframe images and metadata will be written.",
    )
    parser.add_argument("--num-keyframes", type=int, default=40)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adaptive-budget", action="store_true")
    parser.add_argument("--save-raw-stage", action="store_true")
    parser.add_argument("--thumb-size", type=int, default=224)
    parser.add_argument("--overview-panels", type=int, default=8)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=PROJECT_ROOT / "prompt_banks" / "generic_caption_prompts.txt",
    )
    return parser.parse_args()


def normalize_video_id(video_id: str) -> str:
    stem = video_id.strip()
    if stem.endswith(".mp4"):
        return stem[:-4]
    return stem


def resolve_video_path(args: argparse.Namespace) -> Path:
    if args.video is not None:
        video_path = args.video.expanduser()
        if video_path.exists():
            return video_path
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_id = args.video_id_flag or args.video_id
    if not video_id:
        raise ValueError("Pass either --video or --video-id.")

    video_stem = normalize_video_id(video_id)
    candidate_dirs = []
    if args.video_dir is not None:
        candidate_dirs.append(args.video_dir.expanduser())
    candidate_dirs.extend(DEFAULT_VIDEO_DIR_CANDIDATES)

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        candidate = directory / f"{video_stem}.mp4"
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidate_dirs)
    raise FileNotFoundError(f"Could not resolve {video_stem}.mp4 in:\n{searched}")


def resolve_output_dir(args: argparse.Namespace, video_path: Path) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser()
    return DEFAULT_OUTPUT_ROOT / video_path.stem


def evenly_spaced_indices(count: int, target: int) -> list[int]:
    if count <= 0 or target <= 0:
        return []
    if target >= count:
        return list(range(count))
    step = (count - 1) / max(target - 1, 1)
    indices = sorted({int(round(i * step)) for i in range(target)})
    cursor = 0
    while len(indices) < target:
        if cursor not in indices:
            indices.append(cursor)
        cursor += 1
    return sorted(indices[:target])


def build_horizontal_overview(output_dir: Path, overview_panels: int) -> str:
    metadata_path = output_dir / "selection_metadata.json"
    if not metadata_path.exists():
        return ""

    payload = json.loads(metadata_path.read_text())
    selected_frames = payload.get("selected_frames", [])
    if not selected_frames:
        return ""

    total_frames = len(selected_frames)
    chosen_indices = evenly_spaced_indices(total_frames, min(overview_panels, total_frames))

    tile_w = 184
    tile_h = 104
    gap = 20
    pad_x = 28
    header_h = 54
    label_h = 26
    timeline_h = 88
    footer_h = 22
    width = pad_x * 2 + len(chosen_indices) * tile_w + max(len(chosen_indices) - 1, 0) * gap
    height = header_h + tile_h + label_h + timeline_h + footer_h

    canvas = Image.new("RGB", (width, height), color=(250, 248, 243))
    draw = ImageDraw.Draw(canvas)

    title = f"TASCC Keyframe Summary  |  {total_frames} selected frames"
    duration = selected_frames[-1]["timestamp_sec"]
    subtitle = f"{selected_frames[0]['timestamp_sec']:.1f}s to {duration:.1f}s"
    draw.text((pad_x, 18), title, fill=(24, 24, 24))
    draw.text((pad_x, 34), subtitle, fill=(92, 92, 92))

    tile_centers: list[tuple[int, int, int]] = []
    for panel_pos, frame_idx in enumerate(chosen_indices):
        frame_info = selected_frames[frame_idx]
        frame_path = Path(frame_info["saved_path"])
        if not frame_path.exists():
            continue
        image = Image.open(frame_path).convert("RGB")
        image = ImageOps.fit(image, (tile_w, tile_h), method=Image.Resampling.LANCZOS)

        x = pad_x + panel_pos * (tile_w + gap)
        y = header_h
        canvas.paste(image, (x, y))
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=10, outline=(210, 206, 196), width=1)

        label = f"#{frame_idx + 1}  {frame_info['timestamp_sec']:.1f}s"
        draw.text((x + 6, y + tile_h + 6), label, fill=(34, 34, 34))
        tile_centers.append((frame_idx, x + tile_w // 2, y + tile_h))

    timeline_y = header_h + tile_h + label_h + 30
    timeline_x0 = pad_x + 10
    timeline_x1 = width - pad_x - 10
    draw.line((timeline_x0, timeline_y, timeline_x1, timeline_y), fill=(190, 184, 170), width=2)

    dot_radius = 3
    accent_radius = 6
    for index in range(total_frames):
        ratio = index / max(total_frames - 1, 1)
        x = int(round(timeline_x0 + ratio * (timeline_x1 - timeline_x0)))
        fill = (72, 72, 72)
        radius = dot_radius
        if index in chosen_indices:
            fill = (202, 94, 52)
            radius = accent_radius
        draw.ellipse((x - radius, timeline_y - radius, x + radius, timeline_y + radius), fill=fill)

    for frame_idx, center_x, tile_bottom in tile_centers:
        ratio = frame_idx / max(total_frames - 1, 1)
        dot_x = int(round(timeline_x0 + ratio * (timeline_x1 - timeline_x0)))
        draw.line((center_x, tile_bottom + 8, dot_x, timeline_y - accent_radius - 3), fill=(202, 94, 52), width=2)

    draw.text((timeline_x0 - 4, timeline_y + 14), "start", fill=(110, 110, 110))
    end_label = "end"
    end_box = draw.textbbox((0, 0), end_label)
    end_w = end_box[2] - end_box[0]
    draw.text((timeline_x1 - end_w + 4, timeline_y + 14), end_label, fill=(110, 110, 110))

    note = "Dots mark all selected keyframes; thumbnails show representative points across the 40-frame sample."
    draw.text((pad_x, height - footer_h), note, fill=(92, 92, 92))

    horizontal_path = output_dir / "timeline_overview.jpg"
    canvas.save(horizontal_path, quality=95)
    canvas.save(output_dir / "contact_sheet.jpg", quality=95)
    return str(horizontal_path)


def main() -> int:
    args = parse_args()
    video_path = resolve_video_path(args)
    output_dir = resolve_output_dir(args, video_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(EXTRACTOR_PATH),
        "--video",
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--num-keyframes",
        str(args.num_keyframes),
        "--sample-fps",
        str(args.sample_fps),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--batch-size",
        str(args.batch_size),
        "--thumb-size",
        str(args.thumb_size),
        "--save-contact-sheet",
    ]
    if args.adaptive_budget:
        command.append("--adaptive-budget")
    if args.save_raw_stage:
        command.append("--save-raw-stage")
    if args.prompt_file is not None:
        command.extend(["--prompt-file", str(args.prompt_file)])

    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    subprocess.run(command, check=True)

    original_contact_sheet = output_dir / "contact_sheet.jpg"
    if original_contact_sheet.exists():
        original_contact_sheet.replace(output_dir / "contact_sheet_grid.jpg")
    overview_path = build_horizontal_overview(output_dir, args.overview_panels)
    if overview_path:
        print(f"Horizontal overview: {overview_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
