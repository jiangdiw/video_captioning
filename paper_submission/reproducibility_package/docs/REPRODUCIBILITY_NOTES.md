# Reproducibility Notes

## Scope

This is a derived-artifact package, not a full raw-data mirror. It supports review of the reported predictions, clean LV-ECR metrics, robustness tables, normalized-reference sensitivity analysis, stress tests, figures, and manuscript source.

## Raw Media

Raw Dattalion videos are excluded. The human-evaluation sanitized sheet uses `external_dattalion_videos/<video_id>.mp4` placeholders so an authorized user can map those rows to their own video copy.

## Private or Internal Files Excluded

Internal reviewer-strategy notes and draft planning files are intentionally excluded. TeX build intermediates, cache folders, temporary spreadsheet lock files, and raw video formats are also excluded.

## Clean LV-ECR Result

The current manuscript does not report the earlier single-rater blinded human-evaluation result. That material is intentionally excluded from this upload package. The final reported system is LV-ECR under clean model-score candidate ordering; the paired bootstrap interval for the gain over the clean adapted baseline is included under `results/lvecr/` and summarized under `results/robustness/`.
