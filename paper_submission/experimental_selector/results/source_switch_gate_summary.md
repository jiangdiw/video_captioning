# Source-Switch Gate Diagnostic

This is a post-submission diagnostic for closing the gap between the frozen
BART-only selector and the candidate-pool oracle under the v3 MSR-VTT-style
Dattalion references. It should not be treated as a primary manuscript result
without a new, frozen protocol.

## Main Finding

The gap to `0.5` CIDEr is not closed by a learned source gate alone. The clean
video-grouped gate either learns the all-BART solution or admits too many
negative adaptive-BLIP switches. Evidence-summary-only gating is safer, but the
largest test gains come from rare templates that are absent in train/val.

| Method | Test CIDEr | Source counts | Note |
| --- | ---: | --- | --- |
| Clean BART baseline | 0.2221 | `{'bart': 30}` | model-score baseline |
| MSR-VTT generated-beam Ridge selector, exploratory | 0.2593 | `{'bart': 30}` | strongest BART-only selector |
| Learned evidence-summary gate, baseline anchor | 0.2437 | `{'bart': 24, 'evidence_summary': 6}` | safer than adaptive-BLIP gate |
| Learned all-source gate with switch constraint | 0.1720 | `{'adaptive_blip': 8, 'bart': 19, 'evidence_summary': 3}` | hurt by adaptive-BLIP false positives |
| BART-beam anchor + high-specificity template priors | 0.3271 | `{'adaptive_blip': 3, 'bart': 24, 'evidence_summary': 3}` | reproducible hybrid diagnostic |
| Candidate-pool oracle | 0.5132 | `{'adaptive_blip': 5, 'bart': 16, 'evidence_summary': 9}` | non-deployable ceiling |

## Interpretation

The learned gate can be made to switch sources, but the train/val evidence does
not cover the highest-value held-out templates. In particular, horse and church
evidence-summary templates are absent in train/val but account for large held-out
gains. A transparent template prior for horse/church evidence summaries plus
adaptive fire-forest/car-fire captions raises CIDEr to `0.3271`, but this is
still far from the oracle.

The practical path toward `0.5` is therefore not more MSR-VTT-only selector
training. It requires either stronger target-domain weak supervision for the
source gate or a better visual verifier/candidate generator that can rank
adaptive-BLIP and evidence-summary candidates reliably.
