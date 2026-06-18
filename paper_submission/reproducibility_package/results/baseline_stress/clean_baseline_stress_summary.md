| method | uses_test_video | uses_target_train_captions | uses_external_pretraining | Bleu_2 | METEOR | ROUGE_L | CIDEr | keyword_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Caption prior medoid | false | true | false | 0.1143 | 0.1741 | 0.1775 | 0.0186 | 100.0% |
| Train-neighbor retrieval | true | true | true | 0.1267 | 0.2021 | 0.2109 | 0.0364 | 90.0% |
| BLIP frame-caption baseline | true | false | true | 0.1729 | 0.2085 | 0.2361 | 0.0827 | 53.3% |
| GIT frame-caption baseline | true | false | true | 0.179 | 0.1949 | 0.2378 | 0.0711 | 43.3% |
| Source-only transfer | true | false | true | 0.1472 | 0.1367 | 0.2009 | 0.0489 | 16.7% |
| Strongest direct transfer | true | true | true | 0.3624 | 0.279 | 0.3336 | 0.1477 | 83.3% |
| Clean adapted model-score baseline | true | true | true | 0.3324 | 0.2744 | 0.3232 | 0.1374 | 90.0% |
| LV-ECR | true | true | true | 0.344 | 0.2885 | 0.3198 | 0.1521 | 93.3% |
