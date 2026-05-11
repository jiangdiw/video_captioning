# Pipeline Transfer Files

This is the minimal file set for the new TASCC end-to-end pipeline. It excludes legacy thesis files, generated outputs, checkpoints, notebooks, and dataset binaries.

## Top-level files

- `README.md`
- `requirements.txt`
- `prepro_tascc_feats.py`
- `novel_keyframe_extractor.py`
- `train_final_bart.py`
- `run_final_stable_tuning.py`
- `analyze_test_by_category.py`
- `train_end_to_end.py`
- `eval.py`
- `visualize_tascc_keyframes.py`

## `ablation/`

Include these if you want the ablation workflow in the transferred repo:

- `ablation/__init__.py`
- `ablation/run_ablation_suite.py`
- `ablation/run_bart_experiment.py`
- `ablation/build_visual_ablation_features.py`

## `data/`

- `data/__init__.py` if present
- `data/msrvtt.py`
- `data/split_videos_simple.py`
- `data/extract_captions.py`
- `data/build_vocab.py`
- `data/dataset.py`
- `data/vocabulary.py`
- `data/split_visual_embeddings.py`
- `data/extract_audio_wav.py`
- `data/extract_vggish_embeddings.py`
- `data/extract_multimodal_embeddings.py`

## `models/`

- `models/__init__.py` if needed by your target repo
- `models/multimodal_encoder.py`
- `models/captioning_model.py`
- `models/attention_lstm_decoder.py`

## `misc/`

- `misc/cocoeval.py`

## `coco-caption/`

Include the `pycocoevalcap` subtree required by `misc/cocoeval.py`:

- `coco-caption/pycocoevalcap/bleu/`
- `coco-caption/pycocoevalcap/cider/`
- `coco-caption/pycocoevalcap/tokenizer/`
- `coco-caption/pycocoevalcap/meteor/` if you want METEOR support
- `coco-caption/pycocoevalcap/rouge/` if you want ROUGE-L support

## `prompt_banks/`

- `prompt_banks/generic_caption_prompts.txt`

## Dataset metadata to include

These are small enough to version and are required by the pipeline:

- `dataset/MSR-VTT/downsampled_2500.json`
- `dataset/MSR-VTT/train_val_videodatainfo.json`
- `dataset/MSR-VTT/test_videodatainfo.json`

## Do not transfer

- `outputs/`
- `features/`
- `datas/`
- `data/processed/`
- `data/processed_full/`
- raw videos
- notebook backups
- legacy training files unless you explicitly want the old thesis pipeline
