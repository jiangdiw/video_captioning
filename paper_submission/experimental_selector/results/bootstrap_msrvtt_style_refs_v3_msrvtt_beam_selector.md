# Dattalion Bootstrap Confidence Intervals

## Per-method intervals

| Method | Metric | Mean | 95% CI |
| --- | --- | ---: | ---: |
| clean_baseline | CIDEr | 0.2221 | [0.1305, 0.3322] |
| clean_baseline | Bleu_4 | 0.0313 | [0.0000, 0.0788] |
| ridge_selector | CIDEr | 0.2323 | [0.1512, 0.3225] |
| ridge_selector | Bleu_4 | 0.0324 | [0.0000, 0.0827] |
| lvecr_rerun | CIDEr | 0.2395 | [0.1589, 0.3286] |
| lvecr_rerun | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| msrvtt_beam_ridge_dev_selected | CIDEr | 0.2464 | [0.1591, 0.3407] |
| msrvtt_beam_ridge_dev_selected | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| msrvtt_beam_ridge_exploratory | CIDEr | 0.2593 | [0.1722, 0.3501] |
| msrvtt_beam_ridge_exploratory | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| oracle | CIDEr | 0.5132 | [0.3950, 0.6405] |
| oracle | Bleu_4 | 0.0871 | [0.0236, 0.1643] |

## Paired differences

| Comparison | Metric | Mean Diff | 95% CI |
| --- | --- | ---: | ---: |
| msrvtt_beam_ridge_dev_selected - clean_baseline | CIDEr | 0.0243 | [-0.0296, 0.0882] |
| msrvtt_beam_ridge_dev_selected - clean_baseline | Bleu_4 | -0.0070 | [-0.0210, 0.0000] |
| msrvtt_beam_ridge_dev_selected - lvecr_rerun | CIDEr | 0.0070 | [-0.0561, 0.0784] |
| msrvtt_beam_ridge_dev_selected - lvecr_rerun | Bleu_4 | 0.0000 | [-0.0000, 0.0000] |
| msrvtt_beam_ridge_dev_selected - ridge_selector | CIDEr | 0.0141 | [-0.0384, 0.0801] |
| msrvtt_beam_ridge_dev_selected - ridge_selector | Bleu_4 | -0.0082 | [-0.0215, 0.0000] |
| msrvtt_beam_ridge_exploratory - clean_baseline | CIDEr | 0.0372 | [-0.0225, 0.1061] |
| msrvtt_beam_ridge_exploratory - clean_baseline | Bleu_4 | -0.0070 | [-0.0210, 0.0000] |
| msrvtt_beam_ridge_exploratory - lvecr_rerun | CIDEr | 0.0198 | [-0.0368, 0.0860] |
| msrvtt_beam_ridge_exploratory - lvecr_rerun | Bleu_4 | 0.0000 | [-0.0000, 0.0000] |
| oracle - msrvtt_beam_ridge_dev_selected | CIDEr | 0.2667 | [0.1672, 0.3826] |
| oracle - msrvtt_beam_ridge_dev_selected | Bleu_4 | 0.0628 | [0.0100, 0.1278] |
