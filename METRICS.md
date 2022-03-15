# Supported Metrics

The table below shows the current support status for available metrics.

| Metric              | Jury Support* | HF/datasets Support** |
|---------------------|---------------|-----------------------|
| Accuracy-Numeric    | &#10003;      | &#10003;              |
| Accuracy-Text       | &#10003;      | &#10060;              |
| Bartscore           | &#10003;      | &#10060;              |
| Bertscore           | &#10003;      | &#10003;              |
| Bleu                | &#10003;      | &#10003;              |
| Bleurt              | &#10003;      | &#10003;              |
| CER                 | &#10003;      | &#10003;              |
| CHRF                | &#10003;      | &#10003;              |
| CodeEval            | &#129000;     | &#10003;              |
| COMET               | &#10003;      | &#10003;              |
| CompetitionMath     | &#129000;     | &#10003;              |
| COVAL               | &#129000;     | &#10003;              |
| CUAD                | &#129000;     | &#10003;              |
| F1-Numeric          | &#10003;      | &#10003;              |
| F1-Text             | &#10003;      | &#10060;              |
| FrugalScore         | &#129000;     | &#10003;              |
| Gleu                | &#129000;     | &#10003;              |
| Glue                | &#129000;     | &#10003;              |
| GoogleBleu          | &#129000;     | &#10003;              |
| IndicGlue           | &#129000;     | &#10003;              |
| MAE                 | &#129000;     | &#10003;              |
| Mahalanobis         | &#129000;     | &#10003;              |
| MatthewsCorrelation | &#129000;     | &#10003;              |
| MAUVE               | &#129000;     | &#10003;              |
| MeanIOU             | &#129000;     | &#10003;              |
| METEOR              | &#10003;      | &#10003;              |
| MSE                 | &#129000;     | &#10003;              |
| PearsonR            | &#129000;     | &#10003;              |
| Perplexity          | &#129000;     | &#10003;              |
| Precision-Numeric   | &#10003;      | &#10003;              |
| Precision-Text      | &#10003;      | &#10060;              |
| Prism               | &#10003;      | &#10060;              |
| Recall-Numeric      | &#10003;      | &#10003;              |
| Recall-Text         | &#10003;      | &#10060;              |
| ROUGE               | &#10003;      | &#10003;              |
| SacreBleu           | &#10003;      | &#10003;              |
| SARI                | &#129000;     | &#10003;              |
| Seqeval             | &#10003;      | &#10003;              |
| SpearmanR           | &#129000;     | &#10003;              |
| Squad               | &#10003;      | &#10003;              |
| Squadv2             | &#129000;     | &#10003;              |
| SuperGlue           | &#129000;     | &#10003;              |
| TER                 | &#10003;      | &#10003;              |
| WER                 | &#10003;      | &#10003;              |
| WikiSplit           | &#129000;     | &#10003;              |
| XNLI                | &#129000;     | &#10003;              |



_*_ For "Jury Support" column &#129000; means that this metric is supported through the HF/datasets, so that it can be 
used just like the `datasets` metric although unfortunately full Jury support for those metrics are not yet available.

_**_ For metrics marked as &#129000; in "Jury Support" and marked as &#10003; in HF/datasets are available in Jury and 
they can (and should) be used as instructed in `datasets` implementation, and there may not be a unified interface 
that applies for all of these partially supported metrics.