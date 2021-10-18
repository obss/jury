# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERTScore metric. The part of this file is adapted from BERTScore implementation
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/bertscore/bertscore.py """

import functools
from contextlib import contextmanager
from typing import Dict, List

import datasets
import numpy as np
import pandas as pd
from packaging import version

from jury.collator import Collator
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import PackagePlaceholder, requirement_message

# `import bert_score` placeholder
bert_score = PackagePlaceholder(version="0.3.10")


@contextmanager
def filter_logging_context():
    def filter_log(record):
        return False if "This IS expected if you are initializing" in record.msg else True

    logger = datasets.utils.logging.get_logger("transformers.modeling_utils")
    logger.addFilter(filter_log)
    try:
        yield
    finally:
        logger.removeFilter(filter_log)


_CITATION = """\
@inproceedings{bert-score,
  title={BERTScore: Evaluating Text Generation with BERT},
  author={Tianyi Zhang* and Varsha Kishore* and Felix Wu* and Kilian Q. Weinberger and Yoav Artzi},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=SkeHuCVFDr}
}
"""

_DESCRIPTION = """\
BERTScore leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
sentences by cosine similarity.
It has been shown to correlate with human judgment on sentence-level and system-level evaluation.
Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for evaluating different language
generation tasks.

See the `README.md` file at [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score) for more
information.
"""

_KWARGS_DESCRIPTION = """
BERTScore Metrics with the hashcode from a source against one or more references.

Args:
    predictions (list of str): Prediction/candidate sentences.
    references (list of str or list of list of str): Reference sentences.
    lang (str): Language of the sentences; required (e.g. 'en').
    model_type (str): Bert specification, default using the suggested
        model for the target language; has to specify at least one of
        `model_type` or `lang`.
    num_layers (int): The layer of representation to use,
        default using the number of layers tuned on WMT16 correlation data.
    verbose (bool): Turn on intermediate status update.
    idf (bool or dict): Use idf weighting; can also be a precomputed idf_dict.
    device (str): On which the contextual embedding model will be allocated on.
        If this argument is None, the model lives on cuda:0 if cuda is available.
    nthreads (int): Number of threads.
    batch_size (int): Bert score processing batch size,
        at least one of `model_type` or `lang`. `lang` needs to be
        specified when `rescale_with_baseline` is True.
    rescale_with_baseline (bool): Rescale bertscore with pre-computed baseline.
    baseline_path (str): Customized baseline file.
    use_fast_tokenizer (bool): `use_fast` parameter passed to HF tokenizer. New in version 0.3.10.

Returns:
    'score': Bertscore f1. This is always the same as 'f1' in cases single-prediction and single-reference, and
        single-prediction and multiple-references, otherwise it is reduced version of 'f1' by `reduce_fn`. 
    'precision': Precision.
    'recall': Recall.
    'f1': F1 score.
    'hashcode': Hashcode of the library.

Examples:

    >>> bertscore = jury.load_metric("bertscore")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = bertscore.compute(predictions=predictions, references=references)
    >>> print(results)
    {'bertscore': {'score': 0.9473764896392822, 'precision': 0.9467198252677917, 'recall': 0.9480386078357697, 
        'f1': 0.9473764896392822, 'hashcode': 'roberta-large_L17_no-idf_version=0.3.10(hug_trans=4.9.1)'}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BertscoreForLanguageGeneration(MetricForLanguageGeneration):
    def _download_and_prepare(self, dl_manager):
        global bert_score

        try:
            import bert_score
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(metric_name="Bertscore", package_name="bert-score"))
        else:
            super(BertscoreForLanguageGeneration, self)._download_and_prepare(dl_manager)

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/Tiiiger/bert_score",
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            codebase_urls=["https://github.com/Tiiiger/bert_score"],
            reference_urls=[
                "https://github.com/Tiiiger/bert_score",
                "https://arxiv.org/abs/1904.09675",
            ],
        )

    def _compute_single_pred_single_ref(
        self,
        predictions,
        references,
        reduce_fn=None,
        lang="en",
        model_type=None,
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
        return_average_scores=False,
    ):
        get_hash = bert_score.utils.get_hash
        scorer = bert_score.BERTScorer

        if version.parse(bert_score.__version__) >= version.parse("0.3.10"):
            get_hash = functools.partial(get_hash, use_fast_tokenizer=use_fast_tokenizer)
            scorer = functools.partial(scorer, use_fast_tokenizer=use_fast_tokenizer)
        elif use_fast_tokenizer:
            raise ImportWarning(
                "To use a fast tokenizer, the module `bert-score>=0.3.10` is required, and the current version of `bert-score` doesn't match this condition.\n"
                'You can install it with `pip install "bert-score>=0.3.10"`.'
            )

        if model_type is None:
            assert lang is not None, "either lang or model_type should be specified"
            model_type = bert_score.utils.lang2model[lang.lower()]

        if num_layers is None:
            num_layers = bert_score.utils.model2layers[model_type]

        hashcode = get_hash(
            model=model_type,
            num_layers=num_layers,
            idf=idf,
            rescale_with_baseline=rescale_with_baseline,
            use_custom_baseline=baseline_path is not None,
        )

        with filter_logging_context():
            if not hasattr(self, "cached_bertscorer") or self.cached_bertscorer.hash != hashcode:
                self.cached_bertscorer = scorer(
                    model_type=model_type,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    nthreads=nthreads,
                    all_layers=all_layers,
                    idf=idf,
                    device=device,
                    lang=lang,
                    rescale_with_baseline=rescale_with_baseline,
                    baseline_path=baseline_path,
                )

        (P, R, F) = self.cached_bertscorer.score(
            cands=predictions,
            refs=references,
            verbose=verbose,
            batch_size=batch_size,
        )

        P = P.tolist()
        R = R.tolist()
        F = F.tolist()
        score = float(np.mean(F))

        if return_average_scores:
            P = float(np.mean(P))
            R = float(np.mean(R))
            F = float(np.mean(F))

        output_dict = {
            "score": score,
            "precision": P,
            "recall": R,
            "f1": F,
            "hashcode": hashcode,
        }
        return output_dict

    def _compute_single_pred_multi_ref(
        self,
        predictions,
        references,
        reduce_fn=None,
        lang="en",
        model_type=None,
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
        return_average_scores=False,
    ):
        # BERTScore inherently supports multiple references
        return self._compute_single_pred_single_ref(
            predictions=predictions,
            references=references,
            reduce_fn=reduce_fn,
            lang=lang,
            model_type=model_type,
            num_layers=num_layers,
            verbose=verbose,
            idf=idf,
            device=device,
            batch_size=batch_size,
            nthreads=nthreads,
            all_layers=all_layers,
            rescale_with_baseline=rescale_with_baseline,
            baseline_path=baseline_path,
            use_fast_tokenizer=use_fast_tokenizer,
            return_average_scores=return_average_scores,
        )

    def _compute_multi_pred_multi_ref(
        self,
        predictions,
        references,
        reduce_fn=None,
        lang="en",
        model_type=None,
        num_layers=None,
        verbose=False,
        idf=False,
        device=None,
        batch_size=64,
        nthreads=4,
        all_layers=False,
        rescale_with_baseline=False,
        baseline_path=None,
        use_fast_tokenizer=False,
    ):
        scores = []
        for preds, refs in zip(predictions, references):
            pred_scores = [
                self._compute_single_pred_multi_ref(
                    predictions=Collator([pred], keep=True),
                    references=Collator([refs], keep=True),
                    reduce_fn=reduce_fn,
                    lang=lang,
                    model_type=model_type,
                    num_layers=num_layers,
                    verbose=verbose,
                    idf=idf,
                    device=device,
                    batch_size=batch_size,
                    nthreads=nthreads,
                    all_layers=all_layers,
                    rescale_with_baseline=rescale_with_baseline,
                    baseline_path=baseline_path,
                    use_fast_tokenizer=use_fast_tokenizer,
                    return_average_scores=True,
                )
                for pred in preds
            ]
            hashcode = pred_scores[0]["hashcode"]
            reduced_score = self._reduce_multi_pred_scores(pred_scores, reduce_fn=reduce_fn)
            scores.append(reduced_score)

        # Average reduced scores
        return self._reduce_multi_pred_scores(scores, reduce_fn=np.mean, hashcode=hashcode)

    def _reduce_multi_pred_scores(self, results: List[Dict], reduce_fn, **kwargs) -> Dict:
        df = pd.DataFrame(results)
        if "hashcode" in df:
            df.drop("hashcode", axis=1, inplace=True)
        scores = df.apply(reduce_fn, axis=0).to_dict()
        scores.update(kwargs)
        return scores

    def add_batch(self, predictions=None, references=None, **kwargs):
        """Add a batch of predictions and references for the metric's stack."""
        # References can be strings or lists of strings
        # Let's change strings to lists of strings with one element
        if references is not None:
            references = [[ref] if isinstance(ref, str) else ref for ref in references]
        super().add_batch(predictions=predictions, references=references, **kwargs)

    def add(self, prediction=None, reference=None, **kwargs):
        """Add one prediction and reference for the metric's stack."""
        # References can be strings or lists of strings
        # Let's change strings to lists of strings with one element
        if isinstance(reference, str):
            reference = [reference]
        super().add(prediction=prediction, reference=reference, **kwargs)
