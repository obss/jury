import importlib
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy
import pandas as pd
from datasets.utils.logging import get_logger

from jury.collator import Collator
from jury.metrics._utils import import_module, is_reduce_fn
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


class Metric(datasets.Metric, ABC):
    default_features = datasets.Features(
        {
            "predictions": datasets.Sequence(datasets.Value("string", id="sequence")),
            "references": datasets.Sequence(datasets.Value("string", id="sequence")),
        }
    )

    def __init__(self, resulting_name: Optional[str] = None, params: Optional[Dict] = None):
        super().__init__()
        self.resulting_name = resulting_name if resulting_name is not None else self.name
        self.params = params if params is not None else {}
        if "reduce_fn" not in self.params:
            self.params.update({"reduce_fn": "max"})
        self.download_and_prepare()

    def _compute(self, predictions: List[List[str]], references: List[List[str]], **kwargs) -> Dict[str, float]:
        assert len(predictions) == len(references), "Predictions and references length does not match."
        reduce_fn = kwargs.get("reduce_fn")
        reduce_fn = self.params["reduce_fn"] if reduce_fn is None else reduce_fn
        if isinstance(reduce_fn, str):
            reduce_fn = getattr(numpy, reduce_fn)
        elif reduce_fn is not None and not callable(reduce_fn):
            raise TypeError(f"'reduce_fn' Expected str or callable, got {type(reduce_fn)}")
        if reduce_fn is not None and not is_reduce_fn(reduce_fn):
            raise ValueError("'reduce_fn' must be an aggregation function.")
        eval_params = {**self.params, **kwargs}
        eval_params.pop("reduce_fn")
        predictions, references = Collator(predictions), Collator(references)
        result = self.evaluate(predictions=predictions, references=references, reduce_fn=reduce_fn, **eval_params)
        return {self.resulting_name: result}

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, **kwargs
    ):
        raise NotImplementedError

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs
    ):
        raise NotImplementedError

    def _compute_multi_pred_multi_ref(self, predictions: Collator, references: Collator, reduce_fn: Callable, **kwargs):
        raise NotImplementedError

    def _download_and_prepare(self, dl_manager):
        """Downloads and prepares resources for the metric.

        This is the internal implementation to overwrite called when user calls
        `download_and_prepare`. It should download all required resources for the metric.

        Args:
            dl_manager (:class:`DownloadManager`): `DownloadManager` used to download and cache data.
        """
        self.external_module_path = None
        return None

    def _get_external_resource(self, module_name: Optional[str], attr: Optional[str] = None):
        if self.external_module_path is None:
            raise AttributeError("'external_module_path' is not defined.")
        if module_name is None:
            module_name = "external_module"
        external_module = import_module(module_name, self.external_module_path)
        if attr is None:
            return external_module
        return getattr(external_module, attr)

    @staticmethod
    def _reduce_scores(scores: Union[List[Dict[str, float]], List[float]], reduce_fn: Callable):
        if isinstance(scores[0], dict):
            score = pd.DataFrame(scores).apply(reduce_fn, axis=0).to_dict()
        else:
            score = float(reduce_fn(scores))
        return score

    def _preprocess(self, predictions: List[List[str]], references: List[List[str]]) -> Tuple[Collator, Collator]:
        return Collator(predictions, keep=True), Collator(references, keep=True)

    def evaluate(
        self, predictions: Collator, references: Collator, reduce_fn: Optional[Callable] = None, **kwargs
    ) -> Dict[str, float]:
        if predictions.can_collapse() and references.can_collapse():
            predictions = predictions.collapse()
            references = references.collapse()
            eval_fn = self._compute_single_pred_single_ref
        elif predictions.can_collapse() and not references.can_collapse():
            predictions = predictions.collapse()
            eval_fn = self._compute_single_pred_multi_ref
        else:
            eval_fn = self._compute_multi_pred_multi_ref

        predictions, references = self._preprocess(predictions, references)
        return eval_fn(predictions=predictions, references=references, reduce_fn=reduce_fn, **kwargs)


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
