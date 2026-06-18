# 2026 Journal Submission Artifacts

This folder contains the clean derived-artifact bundle for the revised long-video transfer captioning manuscript.

- `reproducibility_package/`: manuscript source/PDF, current figures, clean LV-ECR predictions, metric summaries, bootstrap summaries, split metadata, normalized-reference sensitivity summary, manifest, and checksums.
- `reproducibility_package/results/lvecr/lvecr_summary.json`: authoritative clean LV-ECR run summary.
- `reproducibility_package/results/metrics/main_transfer_table_clean_lvecr.md`: manuscript-synchronized transfer table.

The current reported held-out Dattalion results are:

- Clean adapted model-score baseline: `0.13744177764881674` CIDEr
- LV-ECR: `0.15210421290642032` CIDEr
- LV-ECR candidate-pool oracle: `0.32048214544191284` CIDEr

The package intentionally excludes raw Dattalion videos. Those videos have separate platform-level access and redistribution conditions.
