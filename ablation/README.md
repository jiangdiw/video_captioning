# Ablation Workflow

This folder contains the scripts needed specifically for the ablation study pipeline.

## Files

- `run_ablation_suite.py`
  - Top-level orchestrator for the ablation suite.
  - Launches decoder freeze-vs-trainable comparisons, modality ablations, and TASCC-vs-histogram runs.

- `run_bart_experiment.py`
  - Executes one BART experiment configuration and writes metrics/checkpoints for that run.

- `build_visual_ablation_features.py`
  - Prepares CLIP and DINOv2 visual stores needed for ablation experiments.
  - Supports both TASCC and histogram-topk selectors.

## Recommended entrypoint

Run ablations from the repo root with:

```bash
python3 ablation/run_ablation_suite.py --resource-root . --output-root outputs/ablation_suite_full
```
