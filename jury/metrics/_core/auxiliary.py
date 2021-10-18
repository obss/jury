from typing import Any, Dict, Optional, Union

from jury.metrics._core.base import MetricForTask
from jury.metrics._core.utils import TaskNotAvailable


class TaskMapper:
    """
    Base metric factory class which will be used as mapper for any metric class. This class is used by Autometric.
    """

    _TASKS: Dict[str, MetricForTask] = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError("This class is designed to be instantiated by using 'by_task()' method.")

    @classmethod
    def construct(
        cls, task: str, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        subclass = cls._get_subclass(task=task)
        metric_name = cls._get_metric_name()
        resulting_name = resulting_name or metric_name
        if subclass is None:
            raise TaskNotAvailable(metric_name=metric_name, task=task)
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_subclass(cls, task: str) -> Union[MetricForTask, None]:
        """
        All metric modules must implement this method as it is used to call metrics by default. Should raise
        proper exception (``TaskNotAvailable``) if the task is not supported by the metric.

        Args:
            task: (``str``) Task name for the desired metric.

        Returns: Metric for proper task if available, None otherwise.
        """
        return cls._TASKS.get(task, None)

    @classmethod
    def _get_metric_name(cls):
        return cls.__name__.lower()


class MetricAlias(TaskMapper):
    _SUBCLASS = None

    @classmethod
    def construct(
        cls,
        task: str = None,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        subclass = cls._get_subclass()
        resulting_name = resulting_name or cls._get_metric_name()
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_subclass(cls, *args, **kwargs) -> MetricForTask:
        return cls._SUBCLASS
