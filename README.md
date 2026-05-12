# Video Captioning Pipeline

This repository contains the current MSR-VTT video captioning workflow used in this project. The repo supports:

- TASCC key-frame extraction
- CLIP + DINOv2 visual features
- VGGish audio features
- multimodal fusion
- BART-based caption generation
- ablation runs
- category-level analysis

The recommended path for the final model is:

1. build captions and vocabulary
2. extract TASCC fused features
3. split TASCC fused features into CLIP and DINOv2 streams
4. extract WAV and VGGish audio features
5. train the stable BART model with `train_final_bart.py`
6. analyze category-level performance with `analyze_test_by_category.py`

If you want a notebook that runs the pipeline by calling the existing scripts, use:

- [pipeline_runner.ipynb](./pipeline_runner.ipynb)

It is an orchestration notebook only. It does not reimplement the pipeline logic.

## Repository Layout

### Core pipeline entrypoints

- [prepro_tascc_feats.py](./prepro_tascc_feats.py)
  - Runs TASCC key-frame extraction and writes fused `(40, 1280)` frame features.
  - Also writes selected timestamp CSVs and manifest files.

- [novel_keyframe_extractor.py](./novel_keyframe_extractor.py)
  - Implements TASCC itself:
    - candidate frame sampling
    - CLIP and DINOv2 embedding
    - semantic scoring
    - scene detection
    - greedy frame selection
    - compression to a fixed 40-frame budget

- [visualize_tascc_keyframes.py](./visualize_tascc_keyframes.py)
  - Runs TASCC on a single video and writes visualization artifacts such as the horizontal timeline overview.

- [train_final_bart.py](./train_final_bart.py)
  - Recommended final training entrypoint.
  - Trains the stable BART-based captioning model on either the subset or full dataset.
  - Supports warm-start continuation, validation-CIDEr checkpointing, and controlled tuning presets.

- [run_final_stable_tuning.py](./run_final_stable_tuning.py)
  - Convenience wrapper for launching multiple stable final-model tuning runs.
  - Useful when you want to compare continuation and long-run configurations without rewriting commands.

- [analyze_test_by_category.py](./analyze_test_by_category.py)
  - Computes test-set metrics broken down by MSR-VTT category.
  - Also produces category plots and average train-duration diagnostics.

- [run_ablation_suite.py](./run_ablation_suite.py)
  - Orchestrates the BART ablation study workflow.
  - Runs decoder freeze-vs-trainable comparisons, modality ablations, and TASCC-vs-histogram comparisons.

- [run_bart_experiment.py](./run_bart_experiment.py)
  - Lower-level BART experiment runner used by the ablation suite.
  - Handles one experiment configuration at a time and writes metrics/checkpoints for that run.

- [build_visual_ablation_features.py](./build_visual_ablation_features.py)
  - Builds or reorganizes visual feature stores needed for ablation experiments.

### Data preparation scripts

- [data/extract_captions.py](./data/extract_captions.py)
  - Builds processed caption JSONs for `train`, `val`, and `test`.

- [data/build_vocab.py](./data/build_vocab.py)
  - Builds the vocabulary used by the non-BART captioning paths and some processed-data flows.

- [data/split_visual_embeddings.py](./data/split_visual_embeddings.py)
  - Splits TASCC fused features into:
    - CLIP embeddings under `clip_embedding/`
    - DINOv2 embeddings under `Dinov2_embedding/`

- [data/extract_audio_wav.py](./data/extract_audio_wav.py)
  - Extracts WAV audio from videos.
  - Requires `ffmpeg` and `ffprobe`.

- [data/extract_vggish_embeddings.py](./data/extract_vggish_embeddings.py)
  - Converts extracted WAV files into VGGish embeddings.

- [data/extract_multimodal_embeddings.py](./data/extract_multimodal_embeddings.py)
  - Builds cached multimodal feature representations used by older or auxiliary training paths.

- [data/msrvtt.py](./data/msrvtt.py)
  - Central dataset utility module.
  - Resolves:
    - subset vs full mode
    - processed artifact roots
    - metadata paths
    - split IDs from caption files
    - video-number sorting helpers

- [data/dataset.py](./data/dataset.py)
  - Dataset utilities used by training and generation scripts.

- [data/vocabulary.py](./data/vocabulary.py)
  - Vocabulary helper logic for the older captioning stack.

### Model code

- [models/flexible_bart_captioning_model.py](./models/flexible_bart_captioning_model.py)
  - Current stable BART captioning model.
  - Uses the external multimodal encoder and a pretrained BART decoder stack.
  - In stable freeze mode, BART is kept fixed and training focuses on the external encoder/projection path.

- [models/final_bart_captioning_model.py](./models/final_bart_captioning_model.py)
  - Experimental final-model variant with a more aggressive architecture.
  - Kept in the repo for controlled experimentation, but not the default recommended path.

- [models/multimodal_encoder_40embedding.py](./models/multimodal_encoder_40embedding.py)
  - Main multimodal encoder for the BART path.
  - Handles CLIP, DINOv2, and audio fusion at the 40-frame sequence level.

- [models/bart_captioning_model.py](./models/bart_captioning_model.py)
  - Original BART-based captioning model with audio.

- [models/bart_captioning_model_no_audio.py](./models/bart_captioning_model_no_audio.py)
  - BART-based captioning model variant without audio.

- [models/captioning_model.py](./models/captioning_model.py)
  - Older LSTM-based captioning model code.

- [models/attention_lstm_decoder.py](./models/attention_lstm_decoder.py)
  - Attention LSTM decoder implementation for the older stack.

- [models/multimodal_encoder.py](./models/multimodal_encoder.py)
  - Older multimodal encoder implementation used by the legacy path.

### Legacy training and evaluation scripts

- [train_end_to_end.py](./train_end_to_end.py)
  - Older end-to-end trainer for the pre-BART multimodal captioning pipeline.

- [eval.py](./eval.py)
  - Evaluation entrypoint for the older checkpoint format and legacy training path.

- [train_bart.py](./train_bart.py)
- [train_bart_frozen_decoder.py](./train_bart_frozen_decoder.py)
- [train_bart_no_audio.py](./train_bart_no_audio.py)
- [train_bart_no_audio_frozen_decoder.py](./train_bart_no_audio_frozen_decoder.py)
  - Narrow training wrappers for specific BART variants.
  - These remain useful for targeted experiments, but `train_final_bart.py` is the cleaner recommended entrypoint.

- [generate_bart.py](./generate_bart.py)
- [generate_bart_no_audio.py](./generate_bart_no_audio.py)
  - Inference/generation scripts for older BART experiment flows.

### Evaluation utilities

- [misc/cocoeval.py](./misc/cocoeval.py)
  - Metric wrapper for BLEU, CIDEr, ROUGE, and METEOR.
  - Includes fallbacks for environments where Java-backed scorers are unavailable.

- [coco-caption](./coco-caption)
  - Bundled COCO caption metric package used by the evaluation code.

### Supporting assets

- [prompt_banks/generic_caption_prompts.txt](./prompt_banks/generic_caption_prompts.txt)
  - Default prompt bank used by TASCC prompt-alignment scoring.

- [requirements-pipeline.txt](./requirements-pipeline.txt)
  - Python dependencies for the training/evaluation pipeline.

- [requirements-keyframe.txt](./requirements-keyframe.txt)
  - Python dependencies for TASCC and key-frame visualization.

- [requirements-tascc-macos.txt](./requirements-tascc-macos.txt)
  - Minimal macOS-specific requirements file kept from the earlier setup.

- [PIPELINE_TRANSFER_FILES.md](./PIPELINE_TRANSFER_FILES.md)
  - Narrow manifest of files required when moving only the core pipeline into another repo.

## Dataset Modes

The repo supports two modes:

### `subset`

- balanced 2,500-video MSR-VTT subset
- train / val / test = `2000 / 250 / 250`
- processed outputs under:
  - `data/processed`

### `full`

- full MSR-VTT split configuration
- train / val / test = `6513 / 497 / 2990`
- processed outputs under:
  - `data/processed_full`

The logic that resolves these split-specific roots lives in [data/msrvtt.py](./data/msrvtt.py).

## Expected Data Layout

Metadata files expected in the repo:

- [dataset/MSR-VTT/downsampled_2500.json](./dataset/MSR-VTT/downsampled_2500.json)
- [dataset/MSR-VTT/train_val_videodatainfo.json](./dataset/MSR-VTT/train_val_videodatainfo.json)
- [dataset/MSR-VTT/test_videodatainfo.json](./dataset/MSR-VTT/test_videodatainfo.json)

Video roots typically used by the code:

- `dataset/MSR-VTT/downsampled_2500_videos`
- `dataset/MSR-VTT/full_dataset`
- `dataset/MSR-VTT/TrainValVideo`
- `dataset/MSR-VTT/TestVideo`

In this repo, several of those are often symlinked to larger stores outside the Git tree. That is expected.

## Environment Setup

Install Python dependencies:

```bash
pip install -r requirements-pipeline.txt
pip install -r requirements-keyframe.txt
```

Audio extraction also requires:

```bash
brew install ffmpeg
```

Verify:

```bash
which ffmpeg
which ffprobe
```

## Recommended Notebook Entry Point

If you want a single document that drives the repo by calling the scripts:

- [pipeline_runner.ipynb](./pipeline_runner.ipynb)

The notebook is designed to be run from the repo root and has:

- a configuration cell
- command preview cells
- individual stage cells
- a single "run everything" cell
- final metrics inspection cells

## Full Pipeline: CLI Version

Run commands from the repo root.

### 1. Build processed captions and vocabulary

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

### 2. Build TASCC fused features

Subset:

```bash
python3 prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/downsampled_2500_videos \
  --video-list-json dataset/MSR-VTT/downsampled_2500.json \
  --output-dir features/tascc_fused \
  --timestamps-dir features/tascc_timestamps \
  --manifest-path features/tascc_manifest_subset.json
```

Full train/val:

```bash
python3 prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/full_dataset \
  --video-list-json dataset/MSR-VTT/train_val_videodatainfo.json \
  --output-dir datas/feats/tascc_fused \
  --timestamps-dir datas/feats/tascc_timestamps \
  --manifest-path datas/feats/tascc_manifest_trainval_full.json
```

Full test:

```bash
python3 prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/TestVideo \
  --video-list-json dataset/MSR-VTT/test_videodatainfo.json \
  --output-dir datas/feats/tascc_fused \
  --timestamps-dir datas/feats/tascc_timestamps \
  --manifest-path datas/feats/tascc_manifest_test_full.json
```

### 3. Split fused TASCC features into CLIP and DINOv2 feature stores

Subset:

```bash
python3 data/split_visual_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/split_visual_embeddings.py --dataset-mode full --fused-dir datas/feats/tascc_fused
```

### 4. Extract audio WAV files

Subset:

```bash
python3 data/extract_audio_wav.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_audio_wav.py --dataset-mode full
```

### 5. Extract VGGish embeddings

Subset:

```bash
python3 data/extract_vggish_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_vggish_embeddings.py --dataset-mode full
```

### 6. Optional multimodal cache build

This is optional for the recommended final BART path but useful for older cached experiment flows.

Subset:

```bash
python3 data/extract_multimodal_embeddings.py --dataset-mode subset
```

Full:

```bash
python3 data/extract_multimodal_embeddings.py --dataset-mode full
```

### 7. Train the recommended final model

Subset:

```bash
python3 train_final_bart.py \
  --architecture stable \
  --dataset-mode subset \
  --modalities clip_dino_audio \
  --visual-source auto \
  --decoder-train-mode freeze \
  --xe-epochs 40 \
  --skip-scst \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --eval-batch-size 1 \
  --encoder-lr 1e-4 \
  --bart-lr 2e-5 \
  --val-num-beams 2 \
  --test-num-beams 2 \
  --device mps \
  --run-dir outputs/final_bart_subset_stable_v1
```

Full:

```bash
python3 train_final_bart.py \
  --architecture stable \
  --dataset-mode full \
  --modalities clip_dino_audio \
  --visual-source auto \
  --decoder-train-mode freeze \
  --xe-epochs 40 \
  --skip-scst \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --eval-batch-size 1 \
  --encoder-lr 1e-4 \
  --bart-lr 2e-5 \
  --val-num-beams 2 \
  --test-num-beams 2 \
  --device mps \
  --run-dir outputs/final_bart_full_stable_v1
```

### 8. Tune the stable final model

Single continuation run:

```bash
python3 train_final_bart.py \
  --preset stable_continue_v3 \
  --dataset-mode full \
  --modalities clip_dino_audio \
  --visual-source auto \
  --device mps \
  --run-dir outputs/final_bart_full_tuned_continue_v3
```

Multi-run sweep:

```bash
python3 run_final_stable_tuning.py --device mps
```

### 9. Run ablations

```bash
python3 run_ablation_suite.py \
  --resource-root . \
  --output-root outputs/ablation_suite_full \
  --device mps \
  --epochs-full 20 \
  --epochs-subset 20 \
  --batch-size-full 8 \
  --batch-size-subset 16 \
  --skip-existing
```

### 10. Analyze category-level performance

Subset-trained model:

```bash
python3 analyze_test_by_category.py \
  --predictions outputs/final_bart_subset_stable_v1/test_predictions_xe.json \
  --train-dataset-mode subset \
  --output-dir outputs/final_bart_subset_stable_v1/category_breakdown_xe
```

Full-trained model:

```bash
python3 analyze_test_by_category.py \
  --predictions outputs/final_bart_full_stable_v1/test_predictions_xe.json \
  --train-dataset-mode full \
  --output-dir outputs/final_bart_full_stable_v1/category_breakdown_xe
```

## Output Directories

Generated outputs are intentionally kept out of version control through `.gitignore`.

Typical locations:

- `features/`
  - subset TASCC fused features and timestamps
- `datas/feats/`
  - full TASCC fused features and timestamps
- `data/processed/`
  - processed subset artifacts
- `data/processed_full/`
  - processed full-dataset artifacts
- `outputs/`
  - training runs, ablation results, and category analysis artifacts

## Recommended Default Path

If you want the most reliable current path:

1. use TASCC features
2. use all three modalities: `clip_dino_audio`
3. use the stable BART model
4. freeze BART in the stable path
5. skip SCST unless you have a specific reason to revisit it

## Notes

- `train_final_bart.py` is the recommended training entrypoint.
- `pipeline_runner.ipynb` is the recommended notebook entrypoint.
- `train_end_to_end.py` and the older LSTM stack are still preserved, but they are not the preferred path for the final model.
- `eval.py` remains in the repo for the older pipeline family.
