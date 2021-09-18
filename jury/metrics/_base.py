from typing import Dict, Optional

import datasets
import numpy
from datasets.utils.logging import get_logger

from jury.collator import Collator

logger = get_logger(__name__)


class Metric(datasets.Metric):
    def __init__(self, resulting_name: Optional[str] = None, params: Optional[Dict] = None):
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.params = params if params is not None else {"reduce_fn": "mean"}
        super().__init__()

    def evaluate(self, predictions, references, reduce_fn, **kwargs):
        """Common interface for Jury evaluation metrics"""
        raise NotImplementedError

    def _compute(self, predictions, references, **kwargs):
        reduce_fn_name = kwargs.get("reduce_fn", self.params["reduce_fn"])
        reduce_fn = getattr(numpy, reduce_fn_name)
        predictions, references = self._preprocess(predictions, references)
        eval_params = {**self.params, **kwargs}
        eval_params.pop("reduce_fn")
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return result

    def _preprocess(self, predictions, references):
        return Collator(predictions), Collator(references)
