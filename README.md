# Video-Summarization

This repo contains a TASCC-based video captioning pipeline for MSR-VTT. The current training stack uses:

- TASCC key-frame extraction to produce fixed `(40, 1280)` fused visual features
- split CLIP `(40, 512)` and DINOv2 `(40, 768)` visual streams
- VGGish audio embeddings
- a multimodal encoder
- an LSTM caption decoder

The default end-to-end training path now trains **approach 1 only** unless you explicitly ask for approach 2 or both.

## Model Variants

### Approach 1
- CLIP and DINOv2 are fused with cross-attention
- audio is concatenated into the multimodal representation
- this is the default training target

### Approach 2
- trimodal sequential cross-attention across CLIP, DINOv2, and audio
- available for comparison runs

## Dataset Modes

The repo supports two distinct dataset modes.

### `subset`
- Uses the downsampled `2500`-video selection
- Processed outputs live under [data/processed](/Users/aglooney03/Video-Summarization/data/processed)

### `full`
- Uses full MSR-VTT metadata:
  - train: `6513`
  - val: `497`
  - test: `2990`
- Processed outputs live under [data/processed_full](/Users/aglooney03/Video-Summarization/data/processed_full)

The split definitions are metadata-driven through [data/msrvtt.py](/Users/aglooney03/Video-Summarization/data/msrvtt.py). The code no longer relies on hard-coded ID thresholds for the new pipeline.

## Repository Assumptions

Expected metadata files:

- [dataset/MSR-VTT/downsampled_2500.json](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/downsampled_2500.json)
- [dataset/MSR-VTT/train_val_videodatainfo.json](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/train_val_videodatainfo.json)
- [dataset/MSR-VTT/test_videodatainfo.json](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/test_videodatainfo.json)

Expected video locations that the code can resolve automatically:

- [dataset/MSR-VTT/downsampled_2500_videos](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/downsampled_2500_videos)
- [dataset/MSR-VTT/full_dataset](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/full_dataset)
- [dataset/MSR-VTT/TrainValVideo](/Users/aglooney03/Video-Summarization/dataset/MSR-VTT/TrainValVideo)
- [/Users/aglooney03/Downloads/TestVideo 2](/Users/aglooney03/Downloads/TestVideo%202)

## Prerequisites

Python dependencies:

```bash
pip install -r requirements-tascc-macos.txt
pip install -r requirements-keyframe.txt
```

Audio extraction depends on `ffmpeg` and `ffprobe`:

```bash
brew install ffmpeg
```

Verify:

```bash
which ffmpeg
which ffprobe
```

## Full Pipeline

Run all commands from [Video-Summarization](/Users/aglooney03/Video-Summarization).

### 1. Build per-split captions and vocabulary

Subset:

```bash
python3 data/extract_captions.py --dataset-mode subset
python3 data/build_vocab.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_captions.py --dataset-mode full
python3 data/build_vocab.py --dataset-mode full
```

### 2. Generate TASCC fused features

If the fused TASCC features do not already exist, generate them first.

Subset example:

```bash
python3 prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/downsampled_2500_videos \
  --video-list-json dataset/MSR-VTT/downsampled_2500.json \
  --output-dir features/tascc_fused \
  --timestamps-dir features/tascc_timestamps \
  --manifest-path features/tascc_manifest_subset.json
```

Full train/val example:

```bash
python3 prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/full_dataset \
  --video-list-json dataset/MSR-VTT/train_val_videodatainfo.json \
  --output-dir datas/feats/tascc_fused \
  --timestamps-dir datas/feats/tascc_timestamps \
  --manifest-path datas/feats/tascc_manifest_trainval_full.json
```

Full test example:

```bash
python3 prepro_tascc_feats.py \
  --video-dir "/Users/aglooney03/Downloads/TestVideo 2" \
  --video-list-json dataset/MSR-VTT/test_videodatainfo.json \
  --output-dir datas/feats/tascc_fused \
  --timestamps-dir datas/feats/tascc_timestamps \
  --manifest-path datas/feats/tascc_manifest_test_full.json
```

### 3. Split TASCC fused features into CLIP and DINOv2 streams

Subset:

```bash
python3 data/split_visual_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/split_visual_embeddings.py \
  --dataset-mode full \
  --fused-dir /Users/aglooney03/Video-Summarization/features/tascc_fused \
  --fused-dir /Users/aglooney03/Video-Summarization/datas/feats/tascc_fused
```

Use multiple `--fused-dir` flags when train/val and test fused features live in different folders.

### 4. Extract WAV audio

Subset:

```bash
python3 data/extract_audio_wav.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_audio_wav.py --dataset-mode full
```

### 5. Extract VGGish audio embeddings

Subset:

```bash
python3 data/extract_vggish_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_vggish_embeddings.py --dataset-mode full
```

### 6. Build multimodal encoder outputs

This is optional if you are training directly from raw CLIP/DINO/audio streams, but it enables the fallback precomputed modes.

Subset:

```bash
python3 data/extract_multimodal_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_multimodal_embeddings.py --dataset-mode full
```

### 7. Train

Default: approach 1 only.

Subset:

```bash
python3 train_end_to_end.py --dataset-mode subset --preset best_guess
```

Full:

```bash
python3 train_end_to_end.py --dataset-mode full --preset best_guess --run-suffix raw_seq_v1
```

Train approach 2 only:

```bash
python3 train_end_to_end.py --dataset-mode full --approach approach2 --preset best_guess
```

Train both for a comparison run:

```bash
python3 train_end_to_end.py --dataset-mode full --approach both --preset best_guess
```

### 8. Evaluate

Subset example:

```bash
python3 eval.py \
  --recover_opt /Users/aglooney03/Video-Summarization/outputs/checkpoints/approach1_clip_dino_crossattn_audio_concat/opt_info.json
```

Full example:

```bash
python3 eval.py \
  --recover_opt /Users/aglooney03/Video-Summarization/outputs/checkpoints/approach1_clip_dino_crossattn_audio_concat_full_raw_seq-v1/opt_info.json
```

`eval.py` now reads the checkpoint metadata to reconstruct:

- dataset mode
- processed root
- feature mode
- correct caption ground-truth path

## Notes

- A normal `train_end_to_end.py` run now trains **approach 1 only** by default.
- `eval.py` supports both processed caption JSONs and raw MSR-VTT metadata JSONs for ground truth.
- The new pipeline stores subset and full artifacts separately, so running one mode does not overwrite the other.

## Recommended Default Commands

If you only want the main path:

Subset:

```bash
python3 train_end_to_end.py --dataset-mode subset
```

Full:

```bash
python3 train_end_to_end.py --dataset-mode full --run-suffix raw_seq_v1
```
