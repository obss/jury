from abc import ABC
from typing import Any, Dict, Optional

from datasets import MetricInfo

from jury.metrics._core.base import MetricForTask, EvaluationInstance, MetricOutput


class TaskMapper(ABC):
    """
    Base metric factory class which will be used as mapper for any metric class. This class is used by Autometric.
    """

    _METRIC_NAME = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError("This class is designed to be instantiated by using 'by_task()' method.")

    @classmethod
    def by_task(
        cls, task: str, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        subclass = cls._get_subclass(task=task)
        resulting_name = resulting_name or cls._get_metric_name()
        return subclass.construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @staticmethod
    def _get_subclass(task: str):
        """
        All metric modules must implement this method as it is used to call metrics by default. Should raise
        proper exception (``TaskNotAvailable``) if the task is not supported by the metric.

        Args:
            task: (``str``) Task name for the desired metric.

        Raises: TaskNotAvailable if given task does not match for desired metric.

        Returns: Metric for proper task.
        """
        raise NotImplementedError

    @classmethod
    def _get_metric_name(cls) -> str:
        """
        All metric modules must implement this method as it is used to form MetricOutput properly.

        Returns: Metric name.
        """
        return cls._METRIC_NAME


class MetricAlias(TaskMapper):
    _SUBCLASS = None

    @classmethod
    def by_task(
            cls, task: str, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs
    ):
        subclass = cls._SUBCLASS
        resulting_name = resulting_name or cls._get_metric_name()
        return subclass.construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)
