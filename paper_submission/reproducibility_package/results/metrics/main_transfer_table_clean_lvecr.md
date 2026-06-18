| method | Bleu_1 | Bleu_2 | METEOR | ROUGE_L | CIDEr | note |
| --- | --- | --- | --- | --- | --- | --- |
| 0-shot source only | 0.3702 | 0.1472 | 0.1367 | 0.2009 | 0.0491 | Manuscript rounded row; full source metrics are packaged separately. |
| 10-shot direct | 0.4389 | 0.1857 | 0.1975 | 0.271 | 0.0921 | Manuscript rounded row. |
| 20-shot direct | 0.4591 | 0.2027 | 0.1926 | 0.2652 | 0.1031 | Manuscript rounded row. |
| 30-shot direct | 0.464 | 0.2141 | 0.1786 | 0.2689 | 0.0804 | Manuscript rounded row. |
| 74-shot adaptive120 direct | 0.6224 | 0.3624 | 0.279 | 0.3336 | 0.1477 | Rounded from the packaged adaptive120 direct metrics. |
| Clean adapted model-score baseline | 0.5962 | 0.3324 | 0.2744 | 0.3232 | 0.1374 | Rounded from the clean LV-ECR summary baseline metrics. |
| LV-ECR (adaptive visual+support) | 0.6009 | 0.344 | 0.2885 | 0.3198 | 0.1521 | Rounded from the clean LV-ECR summary metrics. |
| LV-ECR candidate-pool oracle | 0.7815 | 0.5055 | 0.3776 | 0.4204 | 0.3205 | Rounded from the clean LV-ECR diagnostic oracle metrics. |
