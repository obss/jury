# Supported Metrics

The table below shows the current support status for available metrics.

|       Metric        |    Jury Support     | HF/datasets Support |
|---------------------|---------------------|---------------------|
| Accuracy-Numeric    | :heavy_check_mark:  | :heavy_check_mark:  |
| Accuracy-Text       | :heavy_check_mark:  | :x:                 |
| Bartscore           | :heavy_check_mark:  | :x:                 |
| Bertscore           | :heavy_check_mark:  | :heavy_check_mark:  |
| Bleu                | :heavy_check_mark:  | :heavy_check_mark:  |
| Bleurt              | :heavy_check_mark:  | :heavy_check_mark:  |
| CER                 | :heavy_check_mark:  | :heavy_check_mark:  |
| CHRF                | :heavy_check_mark:  | :heavy_check_mark:  |
| CodeEval            | :white_check_mark:  | :heavy_check_mark:  |
| COMET               | :heavy_check_mark:  | :heavy_check_mark:  |
| CompetitionMath     | :white_check_mark:  | :heavy_check_mark:  |
| COVAL               | :white_check_mark:  | :heavy_check_mark:  |
| CUAD                | :white_check_mark:  | :heavy_check_mark:  |
| F1-Numeric          | :heavy_check_mark:  | :heavy_check_mark:  |
| F1-Text             | :heavy_check_mark:  | :x:                 |
| FrugalScore         | :white_check_mark:  | :heavy_check_mark:  |
| Gleu                | :white_check_mark:  | :heavy_check_mark:  |
| Glue                | :white_check_mark:  | :heavy_check_mark:  |
| GoogleBleu          | :white_check_mark:  | :heavy_check_mark:  |
| IndicGlue           | :white_check_mark:  | :heavy_check_mark:  |
| MAE                 | :white_check_mark:  | :heavy_check_mark:  |
| Mahalanobis         | :white_check_mark:  | :heavy_check_mark:  |
| MatthewsCorrelation | :white_check_mark:  | :heavy_check_mark:  |
| MAUVE               | :white_check_mark:  | :heavy_check_mark:  |
| MeanIOU             | :white_check_mark:  | :heavy_check_mark:  |
| METEOR              | :heavy_check_mark:  | :heavy_check_mark:  |
| MSE                 | :white_check_mark:  | :heavy_check_mark:  |
| PearsonR            | :white_check_mark:  | :heavy_check_mark:  |
| Perplexity          | :white_check_mark:  | :heavy_check_mark:  |
| Precision-Numeric   | :heavy_check_mark:  | :heavy_check_mark:  |
| Precision-Text      | :heavy_check_mark:  | :x:                 |
| Prism               | :heavy_check_mark:  | :x:                 |
| Recall-Numeric      | :heavy_check_mark:  | :heavy_check_mark:  |
| Recall-Text         | :heavy_check_mark:  | :x:                 |
| ROUGE               | :heavy_check_mark:  | :heavy_check_mark:  |
| SacreBleu           | :heavy_check_mark:  | :heavy_check_mark:  |
| SARI                | :white_check_mark:  | :heavy_check_mark:  |
| Seqeval             | :heavy_check_mark:  | :heavy_check_mark:  |
| SpearmanR           | :white_check_mark:  | :heavy_check_mark:  |
| Squad               | :heavy_check_mark:  | :heavy_check_mark:  |
| Squadv2             | :white_check_mark:  | :heavy_check_mark:  |
| SuperGlue           | :white_check_mark:  | :heavy_check_mark:  |
| TER                 | :heavy_check_mark:  | :heavy_check_mark:  |
| WER                 | :heavy_check_mark:  | :heavy_check_mark:  |
| WikiSplit           | :white_check_mark:  | :heavy_check_mark:  |
| XNLI                | :white_check_mark:  | :heavy_check_mark:  |





_*_ For "Jury Support" column &#129000; means that this metric is supported through the HF/datasets, so that it can be 
used just like the `datasets` metric although unfortunately full Jury support for those metrics are not yet available.

_**_ For metrics marked as &#129000; in "Jury Support" and marked as &#10004; in HF/datasets are available in Jury and 
they can (and should) be used as instructed in `datasets` implementation, and there may not be a unified interface 
that applies for all of these partially supported metrics.