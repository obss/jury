# Supported Metrics

The table below shows the current support status for available metrics.

| Metric              | Jury Support* | HF/datasets Support** |
|---------------------|---------------|-----------------------|
| Accuracy-Numeric    | &#10004;      | &#10004;              |
| Accuracy-Text       | &#10004;      | &#10060;              |
| Bartscore           | &#10004;      | &#10060;              |
| Bertscore           | &#10004;      | &#10004;              |
| Bleu                | &#10004;      | &#10004;              |
| Bleurt              | &#10004;      | &#10004;              |
| CER                 | &#10004;      | &#10004;              |
| CHRF                | &#10004;      | &#10004;              |
| CodeEval            | &#129000;     | &#10004;              |
| COMET               | &#10004;      | &#10004;              |
| CompetitionMath     | &#129000;     | &#10004;              |
| COVAL               | &#129000;     | &#10004;              |
| CUAD                | &#129000;     | &#10004;              |
| F1-Numeric          | &#10004;      | &#10004;              |
| F1-Text             | &#10004;      | &#10060;              |
| FrugalScore         | &#129000;     | &#10004;              |
| Gleu                | &#129000;     | &#10004;              |
| Glue                | &#129000;     | &#10004;              |
| GoogleBleu          | &#129000;     | &#10004;              |
| IndicGlue           | &#129000;     | &#10004;              |
| MAE                 | &#129000;     | &#10004;              |
| Mahalanobis         | &#129000;     | &#10004;              |
| MatthewsCorrelation | &#129000;     | &#10004;              |
| MAUVE               | &#129000;     | &#10004;              |
| MeanIOU             | &#129000;     | &#10004;              |
| METEOR              | &#10004;      | &#10004;              |
| MSE                 | &#129000;     | &#10004;              |
| PearsonR            | &#129000;     | &#10004;              |
| Perplexity          | &#129000;     | &#10004;              |
| Precision-Numeric   | &#10004;      | &#10004;              |
| Precision-Text      | &#10004;      | &#10060;              |
| Prism               | &#10004;      | &#10060;              |
| Recall-Numeric      | &#10004;      | &#10004;              |
| Recall-Text         | &#10004;      | &#10060;              |
| ROUGE               | &#10004;      | &#10004;              |
| SacreBleu           | &#10004;      | &#10004;              |
| SARI                | &#129000;     | &#10004;              |
| Seqeval             | &#10004;      | &#10004;              |
| SpearmanR           | &#129000;     | &#10004;              |
| Squad               | &#10004;      | &#10004;              |
| Squadv2             | &#129000;     | &#10004;              |
| SuperGlue           | &#129000;     | &#10004;              |
| TER                 | &#10004;      | &#10004;              |
| WER                 | &#10004;      | &#10004;              |
| WikiSplit           | &#129000;     | &#10004;              |
| XNLI                | &#129000;     | &#10004;              |



_*_ For "Jury Support" column &#129000; means that this metric is supported through the HF/datasets, so that it can be 
used just like the `datasets` metric although unfortunately full Jury support for those metrics are not yet available.

_**_ For metrics marked as &#129000; in "Jury Support" and marked as &#10004; in HF/datasets are available in Jury and 
they can (and should) be used as instructed in `datasets` implementation, and there may not be a unified interface 
that applies for all of these partially supported metrics.