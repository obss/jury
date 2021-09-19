import importlib
from typing import Dict, Optional, Tuple, Union, List

import datasets
import numpy
from datasets.utils.logging import get_logger

from jury.collator import Collator
from jury.utils import NestedSingleType

logger = get_logger(__name__)


def load_metric(metric_name: str, resulting_name: str = None, params: Dict = None) -> "Metric":
    # load the module, will raise ImportError if module cannot be loaded
    metric_name = metric_name.lower()
    base_name = metric_name.split("_")[0]
    module_name = f"jury.metrics.{base_name}"
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, m.__class_names__.get(metric_name))
    return c(resulting_name=resulting_name, params=params)


class Metric(datasets.Metric):
    def __init__(self, resulting_name: Optional[str] = None, params: Optional[Dict] = None):
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.params = params if params is not None else {"reduce_fn": "mean"}
        super().__init__()

    def evaluate(self, predictions: Collator, references: Collator, reduce_fn: callable, **kwargs) -> Dict[str, float]:
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            references = references.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)

    def _compute(self, predictions: List[List[str]], references: List[List[str]], **kwargs) -> Dict[str, float]:
        assert len(predictions) == len(references), "Predictions and references length does not match."
        reduce_fn_name = kwargs.get("reduce_fn", self.params["reduce_fn"])
        reduce_fn = getattr(numpy, reduce_fn_name)
        predictions, references = self._preprocess(predictions, references)
        eval_params = {**self.params, **kwargs}
        eval_params.pop("reduce_fn")
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return result

    def _preprocess(self, predictions: List[List[str]], references: List[List[str]]) -> Tuple[Collator, Collator]:
        return Collator(predictions), Collator(references)

    def _compute_single_pred_single_ref(self, predictions: Collator, references: Collator, **kwargs):
        raise NotImplementedError

    def _compute_single_pred_multi_ref(self, predictions: Collator, references: Collator, reduce_fn: callable, **kwargs):
        raise NotImplementedError

    def _compute_multi_pred_multi_ref(self, predictions: Collator, references: Collator, reduce_fn: callable, **kwargs):
        raise NotImplementedError


class MetricCollator(list):
    def __init__(self, metrics: Union[List[str], List[Metric]]):
        metrics = self._constructor(metrics)
        super(MetricCollator, self).__init__(metrics)

    def _constructor(self, metrics: Union[List[str], List[Metric]]) -> list:
        _type = NestedSingleType.get_type(metrics)
        if _type == "list<str>":
            _metrics = []
            for metric in metrics:
                _metrics.append(load_metric(metric))
            metrics = _metrics
        return metrics

    def add_metric(self, metric_name: str, resulting_name: str = None, params: Dict = None) -> None:
        metric = load_metric(metric_name, resulting_name=resulting_name, params=params)
        self.append(metric)

    def remove_metric(self, resulting_name: str) -> None:
        for i, metric in enumerate(self):
            if metric.resulting_name == resulting_name:
                self.pop(i)
                break
        raise ValueError(f"Metric with resulting name {resulting_name} does not exists.")
