---
title: 'Jury: A Comprehensive Evaluation Toolkit'
tags:
  - Python
  - natural-language-generation
  - evaluation
  - metrics
  - natural-language-processing
authors:
  - name: Devrim Cavusoglu
    equal-contrib: true
    corresponding: true
    affiliation: "1, 2"
  - name: Secil Sen
    equal-contrib: true
    affiliation: "1, 3"
  - name: Ulas Sert
    affiliation: 1
  - name: Sinan Altinuc
    affiliation: "1, 2"
affiliations:
 - name: OBSS AI
   index: 1
 - name: Middle East Technical University
   index: 2
 - name: Bogazici University
   index: 3
date: 23 January 2024
bibliography: paper.bib

# Arxiv information
arxiv-doi: 10.48550/arXiv.2310.02040
---

# Summary

Evaluation plays a critical role in deep learning as a fundamental block of any prediction-based system. However, the vast number of Natural Language Processing (NLP) tasks and the development of various metrics have led to challenges in evaluating different systems with different metrics. To address these challenges, we introduce jury, a toolkit that provides a unified evaluation framework with standardized structures for performing evaluation across different tasks and metrics. The objective of jury is to standardize and improve metric evaluation for all systems and aid the community in overcoming the challenges in evaluation. Since its open-source release, jury has reached a wide audience and is publicly available.

# Statement of need

NLP tasks possess inherent complexity, requiring a comprehensive evaluation of model performance beyond a single metric comparison. Established benchmarks such as WMT [@barrault-etal-2020-findings] and GLUE [@wang-etal-2018-glue] rely on multiple metrics to evaluate models on standardized datasets. This practice promotes fair comparisons across different models and pushes advancements in the field. Embracing multiple metric evaluations provides valuable insights into a model's generalization capabilities. By considering diverse metrics, such as accuracy, F1 score, BLEU, and ROUGE, researchers gain a holistic understanding of a model's response to never-seen inputs and its ability to generalize effectively. Furthermore, task-specific NLP metrics enable the assessment of additional dimensions, such as readability, fluency, and coherence. The comprehensive evaluation facilitated by multiple metric analysis allows for trade-off studies and aids in assessing generalization for task-independent models. Given these numerous advantages, NLP specialists lean towards employing multiple metric evaluations. 

Although employing multiple metric evaluation is common, there is a challenge in practical use because widely-used metric libraries lack support for combined and/or concurrent metric computations. Consequently, researchers face the burden of evaluating their models per metric, a process exacerbated by the scale and complexity of recent models and limited hardware capabilities. This bottleneck impedes the efficient assessment of NLP models and highlights the need for enhanced tooling in the metric computation for convenient evaluation. In order for concurrency to be beneficial at a maximum level, the system may require hardware accordingly. Having said that, the availability of the hardware comes into question. 

The extent of achievable concurrency in NLP research has traditionally relied upon the availability of hardware resources accessible to researchers. However, significant advancements have occurred in recent years, resulting in a notable reduction in the cost of high-end hardware, including multi-core CPUs and GPUs. This progress has transformed high-performance computing resources, which were once prohibitively expensive and predominantly confined to specific institutions or research labs, into more accessible and affordable assets. For instance, in BERT [@devlin-etal-2019-bert] and XLNet [@zhilin-etal-2019-xlnet], it is stated that they leveraged the training process by using powerful yet cost-effective hardware resources. Those advancements show that the previously constraining factor for hardware accessibility has been mitigated, allowing researchers to overcome the limitations associated with achieving concurrent processing capabilities in NLP research.

To ease the use of automatic metrics in NLG research, several hands-on libraries have been developed such as _nlg-eval_ [@sharma2017nlgeval] and _datasets/metrics_ [@lhoest-etal-2021-datasets] (now as _evaluate_). Although those libraries cover widely-used NLG metrics, they don't allow using multiple metrics in one go (i.e. combined evaluation), or they provide a crude way of doing so if they do. Those libraries restrict their users to compute each metric sequentially if users want to evaluate their models with multiple metrics which is time-consuming. Aside from this, there are a few problems in the libraries that support combined evaluation such as individual metric construction and passing compute time arguments (e.g. n-gram for BLEU [@papineni-etal-2002-bleu]), etc. Our system provides an effective computation framework and overcomes the aforementioned challenges.

We designed a system that enables the creation of user-defined metrics with a unified structure and the usage of multiple metrics in the evaluation process. Our library also utilizes the _datasets_ package to promote open-source contribution; when users implement metrics, the implementation can be contributed to the _datasets_ package. Any new metric released by the _datasets_ package will be readily available in our library as well.

# Acknowledgements

We would also like to express our appreciation to Cemil Cengiz for fruitful discussions.

# References