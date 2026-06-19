# Dattalion Bootstrap Confidence Intervals

## Per-method intervals

| Method | Metric | Mean | 95% CI |
| --- | --- | ---: | ---: |
| clean_baseline | CIDEr | 0.2221 | [0.1305, 0.3322] |
| clean_baseline | Bleu_4 | 0.0313 | [0.0000, 0.0788] |
| lvecr_rerun | CIDEr | 0.2395 | [0.1589, 0.3286] |
| lvecr_rerun | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| msrvtt_beam_sample_ridge | CIDEr | 0.2464 | [0.1591, 0.3407] |
| msrvtt_beam_sample_ridge | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| msrvtt_beam_full_ridge | CIDEr | 0.2357 | [0.1468, 0.3292] |
| msrvtt_beam_full_ridge | Bleu_4 | 0.0243 | [0.0000, 0.0634] |
| oracle | CIDEr | 0.5132 | [0.3950, 0.6405] |
| oracle | Bleu_4 | 0.0871 | [0.0236, 0.1643] |

## Paired differences

| Comparison | Metric | Mean Diff | 95% CI |
| --- | --- | ---: | ---: |
| msrvtt_beam_full_ridge - clean_baseline | CIDEr | 0.0135 | [-0.0498, 0.0837] |
| msrvtt_beam_full_ridge - clean_baseline | Bleu_4 | -0.0070 | [-0.0210, 0.0000] |
| msrvtt_beam_full_ridge - lvecr_rerun | CIDEr | -0.0038 | [-0.0731, 0.0719] |
| msrvtt_beam_full_ridge - lvecr_rerun | Bleu_4 | -0.0000 | [-0.0000, 0.0000] |
| msrvtt_beam_sample_ridge - msrvtt_beam_full_ridge | CIDEr | 0.0108 | [-0.0130, 0.0465] |
| msrvtt_beam_sample_ridge - msrvtt_beam_full_ridge | Bleu_4 | 0.0000 | [-0.0000, 0.0000] |
| oracle - msrvtt_beam_full_ridge | CIDEr | 0.2775 | [0.1767, 0.3973] |
| oracle - msrvtt_beam_full_ridge | Bleu_4 | 0.0628 | [0.0100, 0.1278] |
