# Video Captioning Pipeline

This repo contains the MSR-VTT pipeline used for:

- TASCC key-frame extraction
- CLIP + DINOv2 visual features
- VGGish audio features
- BART-based caption generation
- category-level evaluation
- ablation runs

If you want a single entrypoint that calls the scripts for you, use:

- [full_pipeline.ipynb](./full_pipeline.ipynb)

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install audio tooling:

```bash
brew install ffmpeg
```

The repo expects the MSR-VTT metadata files under:

- `dataset/MSR-VTT/train_val_videodatainfo.json`
- `dataset/MSR-VTT/test_videodatainfo.json`
- `dataset/MSR-VTT/downsampled_2500.json`

It also expects the actual videos to be available through the dataset paths used by your local setup.

## Most Important Files

- [prepro_tascc_feats.py](./prepro_tascc_feats.py): TASCC feature extraction
- [data/extract_captions.py](./data/extract_captions.py): processed caption JSONs
- [data/build_vocab.py](./data/build_vocab.py): vocabulary build
- [data/split_visual_embeddings.py](./data/split_visual_embeddings.py): split fused TASCC features into CLIP and DINOv2 streams
- [data/extract_audio_wav.py](./data/extract_audio_wav.py): extract WAV audio from videos
- [data/extract_vggish_embeddings.py](./data/extract_vggish_embeddings.py): build VGGish audio embeddings
- [train_final_bart.py](./train_final_bart.py): final recommended training entrypoint
- [train_bart.py](./train_bart.py): legacy BART trainer with `--freeze-bart` and `--no-audio`
- [train_lstm_decoder.py](./train_lstm_decoder.py): legacy LSTM decoder trainer
- [analyze_test_by_category.py](./analyze_test_by_category.py): category-level test analysis
- [misc/visualize_tascc_keyframes.py](./misc/visualize_tascc_keyframes.py): TASCC single-video visualization
- [ablation/run_ablation_suite.py](./ablation/run_ablation_suite.py): ablation study runner

## Full Pipeline

The recommended final model is the stable BART pipeline on the full MSR-VTT dataset.

### 1. Build processed captions and vocabulary

```bash
python data/extract_captions.py --dataset-mode full
python data/build_vocab.py --dataset-mode full
```

If you want the balanced 2500-video subset instead, change `full` to `subset`.

### 2. Extract TASCC fused visual features

Train/val videos:

```bash
python prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/TrainValVideo \
  --video-list-json dataset/MSR-VTT/train_val_videodatainfo.json
```

Test videos:

```bash
python prepro_tascc_feats.py \
  --video-dir dataset/MSR-VTT/TestVideo \
  --video-list-json dataset/MSR-VTT/test_videodatainfo.json
```

### 3. Split fused features into CLIP and DINOv2 stores

```bash
python data/split_visual_embeddings.py --dataset-mode full --fused-dir datas/feats/tascc_fused
```

### 4. Extract audio and VGGish features

```bash
python data/extract_audio_wav.py --dataset-mode full
python data/extract_vggish_embeddings.py --dataset-mode full
```

### 5. Train the final model

```bash
python train_final_bart.py \
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

### 6. Analyze test performance by category

```bash
python analyze_test_by_category.py \
  --predictions outputs/final_bart_full_stable_v1/test_predictions_xe.json \
  --train-dataset-mode full \
  --output-dir outputs/final_bart_full_stable_v1/category_breakdown_xe
```

## Ablations

To run the ablation suite:

```bash
python ablation/run_ablation_suite.py \
  --device mps \
  --epochs-full 20 \
  --epochs-subset 20 \
  --batch-size-full 8 \
  --batch-size-subset 16 \
  --output-root outputs/ablation_suite_full
```

## Outputs

The most important outputs are:

- training metrics and checkpoints under `outputs/...`
- final test metrics in `test_metrics_xe.json`
- final predictions in `test_predictions_xe.json`
- category plots and tables under `category_breakdown_xe/`
