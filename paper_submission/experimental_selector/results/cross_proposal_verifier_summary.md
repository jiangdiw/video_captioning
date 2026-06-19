# Cross-Proposal Verifier

## Method

The method starts from the MSR-VTT-pretrained BART selector and only allows a non-BART source switch when adaptive, base, and/or large BLIP keyframe proposal streams independently support a high-specificity visual concept.

## Test Results

| Method | CIDEr | BLEU-4 | Source counts |
| --- | ---: | ---: | --- |
| clean BART baseline | 0.2221 | 0.1303 | `{'bart': 30}` |
| BART selector anchor | 0.2593 | 0.1302 | `{'bart': 30}` |
| cross-proposal verifier | 0.3732 | 0.1352 | `{'adaptive_blip': 3, 'bart': 25, 'evidence_summary': 2}` |
| candidate-pool oracle | 0.5267 | 0.2024 | `{'adaptive_blip': 6, 'bart': 16, 'evidence_summary': 8}` |

## Selection

- Best dev threshold: `8.0`
- OOF dev CIDEr: `0.1619`
- Test CIDEr gain over clean baseline: `0.1511`
- Test CIDEr gain over selector anchor: `0.1139`
