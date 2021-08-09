from typing import Dict

from jury.metrics import Metric
from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper

__class_names__ = {"bleu": "BLEU"}


class BLEU(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        params = {} if params is None else params
        tokenizer = params.get("tokenizer", None)
        self.tokenizer = BLEUDefaultTokenizer() if tokenizer is None else tokenizer
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _preprocess(self, predictions, references):
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        predictions, references = tokenizer_wrapper.tokenize(predictions, references)
        if predictions.ndim > 2:
            predictions = predictions.reshape_len(-1)

        if references.ndim == 3:
            ref_count = references.shape[0]
            references = references.reshape(1, ref_count, -1)
        else:
            references = references.reshape(1, -1)

        return predictions, references
