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
""" Prism metric. The part of this file is adapted from metric implementations
of datasets package. See
https://github.com/huggingface/datasets/blob/master/metrics/ """

import os
from typing import Dict, Callable

import datasets
import validators

from jury.metrics import LanguageGenerationInstance
from jury.metrics._core import MetricForLanguageGeneration
from jury.metrics._core.utils import download
from jury.utils.io import untar_file

_CITATION = """\
@inproceedings{thompson-post-2020-automatic,
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
"""

_DESCRIPTION = """\
Prism is an automatic MT metric which uses a sequence-to-sequence paraphraser to score MT system outputs 
conditioned on their respective human references. Prism uses a multilingual NMT model as a zero-shot paraphraser, 
which negates the need for synthetic paraphrase data and results in a single model which works in many languages.

See the `README.md` file at [https://github.com/thompsonb/prism](https://github.com/thompsonb/prism) for more
information.
"""

_KWARGS_DESCRIPTION = """
Prism metric arguments.

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
    'score': Prism score.

Examples:

    >>> prism = jury.load_metric("prism")
    >>> predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    >>> references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]
    >>> results = prism.compute(predictions=predictions, references=references)
    >>> print(results)
    {'prism': {'score': 1.0}}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PrismForLanguageGeneration(MetricForLanguageGeneration):
    def __init__(self, resulting_name: str = None, compute_kwargs: Dict = None, model_path_or_url: str = None, lang: str = "en", **kwargs):
        self.model_path_or_url = model_path_or_url
        self.lang = lang
        self.model_dir = None
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_model(self):
        if self.model_path_or_url is None:
            self.model_path_or_url = "http://data.statmt.org/prism/m39v1.tar"

        if not os.path.isdir(self.model_path_or_url) and not validators.url(self.model_path_or_url):
            raise ValueError("Provided 'model_path_or_url' neither points to an existing directory "
                             "nor a valid URL.")
        elif os.path.isdir(self.model_path_or_url):
            self.model_dir = self.model_path_or_url
        else:
            model_source = self.model_path_or_url
            model_dest = os.path.join(self.data_dir, f"prism_model_{self.model_path_or_url}")
            print(f"Downloading the model at {self.model_path_or_url} ...")
            download(source=model_source, destination=model_dest)
            print("Model downloaded.")
            untar_file(model_dest, "/home/devrimcavusoglu/Desktop/abc")

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of Prism score from the implementation
        of Prism computation from thompsonb/prism. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/thompsonb/prism/blob/42e45a46d1c7924e98bceeed2ea81b31efcb6f9d/prism.py
        """
        self._download_model()
        prism_source = "https://raw.githubusercontent.com/thompsonb/prism/42e45a46d1c7924e98bceeed2ea81b31efcb6f9d/prism.py"
        prism_dest = os.path.join(self.data_dir, "prism.py")
        download(
            source=prism_source,
            destination=prism_dest,
        )
        self.external_module_path = prism_dest

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

    def _compute_prism_score(self, predictions: LanguageGenerationInstance, references: LanguageGenerationInstance, **kwargs):
        Prism = self._get_external_resource("prism", attr="Prism")
        prism = Prism()

        score = prism.score(
            ref=references, cand=predictions, **kwargs
        )
        return score

    def _compute_single_pred_single_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn=None,
        lang: str = "en",
        segment_scores: bool = False,
    ):
        prism_score = self._compute_prism_score(predictions, references)
        return {"score": 1}

    def _compute_single_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        pass

    def _compute_multi_pred_multi_ref(
        self,
        predictions: LanguageGenerationInstance,
        references: LanguageGenerationInstance,
        reduce_fn: Callable = None,
        **kwargs,
    ):
        pass


if __name__ == "__main__":
    predictions = ["the cat is on the mat", "Look! a wonderful day."]
    references = ["the cat is playing on the mat.", "Today is a wonderful day"]
    prism = PrismForLanguageGeneration()
    res = prism._compute_single_pred_single_ref(predictions=predictions, references=references)
    print(res)
