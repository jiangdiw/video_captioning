# Reproducibility Package

This package contains the derived artifacts for the revised multimedia-journal submission on transfer learning for long-video captioning under Dattalion domain shift.

It intentionally does not include raw Dattalion videos or local training workspaces. The raw videos have separate platform-level access and redistribution conditions. Files that require raw video access use the placeholder directory `external_dattalion_videos/`.

## Contents

- `manuscript/`: submitted TeX source, bibliography, Springer class/style files, generated PDF, and only the figures referenced by the current manuscript.
- `data/submission_artifacts/splits/`: cleaned executable Dattalion split metadata.
- `data/normalized_references_v2/`: normalized-reference sensitivity summary and style guide.
- `results/predictions/`: held-out Dattalion test predictions for the source, adaptive120 direct-transfer, clean adapted model-score baseline, LV-ECR, LV-ECR candidate-pool oracle, BLIP, and GIT systems.
- `results/metrics/`: manuscript-synchronized transfer table plus full-precision clean LV-ECR metric summaries.
- `results/lvecr/`: authoritative clean LV-ECR summary, bootstrap file, and development-set weight-grid artifact.
- `results/robustness/`: clean LV-ECR paired-bootstrap summary.
- `results/baseline_stress/`: retrieval, caption-prior, external frame-captioner, clean adapted baseline, and LV-ECR controls.
- `results/transfer_ablation/`: clean LV-ECR variant summary.
- `code/`: the package-builder script used to assemble this derived-artifact bundle. Full training/evaluation scripts remain in the local project workspace and are not bundled here because several Drive-backed historical helper files are not reliably readable in the current sync state.
- `docs/`: sanitized experiment index and reproducibility notes.

## Main Audit Path

1. Inspect `docs/experiment_index.sanitized.csv` for the experiment protocol, split sizes, selection rules, and primary packaged artifacts.
2. Inspect `results/metrics/main_transfer_table_clean_lvecr.md` for the manuscript-synchronized transfer table.
3. Inspect `results/lvecr/lvecr_summary.json` and `results/lvecr/lvecr_bootstrap_sentence_cider.json` for the clean LV-ECR result, oracle headroom, selected weights, and bootstrap interval.
4. Inspect `results/baseline_stress/` and `results/transfer_ablation/` for the controls and LV-ECR variant summaries used in the revised manuscript.
5. Verify file integrity with `shasum -a 256 -c CHECKSUMS.sha256` from this directory.

## Regeneration Notes

- The analysis scripts assume the broader training environment used for the paper, including this video-captioning repository, trained checkpoints, Python metric dependencies, and externally obtained Dattalion videos.
- The package is designed so that reviewers can audit the final derived results without raw-video redistribution.
- The package was assembled with `tools/build_reproducibility_package.mjs` in the manuscript workspace; an archival copy of that builder is included at `code/tools/build_reproducibility_package.mjs`.
- In this GitHub branch, treat `paper_submission/reproducibility_package/` as the audited derived-artifact snapshot rather than a self-contained raw-video rebuild workspace.

## Integrity Files

- `MANIFEST.json`: package-relative path, category, byte count, SHA-256 checksum, and source/provenance note for each packaged file.
- `CHECKSUMS.sha256`: standard checksum file for command-line verification.
