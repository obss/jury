from typing import Dict, List, Union

import numpy as np

from jury.metrics import Metric, load_metric
from jury.utils import NestedSingleType


class Collator(list):
    def __init__(self, sequence, keep=False):
        sequence = self._construct(sequence, keep=keep)
        super().__init__(sequence)

    @property
    def shape(self):
        return np.array(self, dtype=object).shape

    @property
    def ndim(self):
        return len(self.shape)

    def collapse(self):
        return Collator(np.ravel(self).tolist(), keep=True)

    def nested(self):
        return Collator(self.from_list_of_str(self))

    def reshape(self, *args):
        _seq = np.array(self, dtype=object)
        return Collator(_seq.reshape(args).tolist(), keep=True)

    def reshape_len(self, *args):
        _len = len(self)
        return self.reshape(_len, *args)

    def can_collapse(self):
        if self.ndim != 2:
            return False

        return self.shape[1] == 1

    def to_list(self, collapse=True):
        if collapse:
            return list(self.collapse())
        return list(self)

    def _construct(self, sequence: Union[str, List[str], List[List[str]]], keep: bool) -> List[List[str]]:
        if keep:
            return sequence

        _type = NestedSingleType.get_type(sequence)
        if _type == "str":
            sequence = self.from_str(sequence)
        elif _type == "list<str>" or _type == "list<dict>":
            sequence = self.from_list_of_str(sequence)

        return sequence

    @staticmethod
    def from_list_of_str(seq: List[str]):
        return [[item] for item in seq]

    @classmethod
    def from_str(cls, seq: str):
        return [seq]


class MetricCollator(list):
    def __init__(self, metrics: Union[List[str], List[Metric]]):
        metrics = self._constructor(metrics)
        super(MetricCollator, self).__init__(metrics)

    def _constructor(self, metrics):
        _type = NestedSingleType.get_type(metrics)
        if _type == "list<str>":
            _metrics = []
            for metric in metrics:
                _metrics.append(load_metric(metric))
            metrics = _metrics
        return metrics

    def add_metric(self, metric_name: str, resulting_name: str = None, params: Dict = None):
        metric = load_metric(metric_name, resulting_name=resulting_name, params=params)
        self.append(metric)

    def remove_metric(self, resulting_name: str):
        for i, metric in enumerate(self):
            if metric.resulting_name == resulting_name:
                self.pop(i)
                break
        raise ValueError(f"Metric with resulting name {resulting_name} does not exists.")
