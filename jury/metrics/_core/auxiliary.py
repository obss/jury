from typing import Any, Dict, Optional, Union

from jury.metrics._core.base import MetricForTask
from jury.metrics._core.utils import TaskNotAvailable


class TaskMapper:
    """
    Base metric factory class which will be used as mapper for any metric class. This class is used by
    :py:class:`jury.AutoMetric` for loading specified metric. It maps the class to a specified metric class
    if multiple tasks are available for the metric.

    All metrics using TaskMapper must implement _TASKS attribute.

    Note:
        Use :py:class:`jury.metrics.TaskMapper` instead in case of metrics implementing a single task.
    """

    _TASKS: Dict[str, MetricForTask] = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError("This class is designed to be instantiated by using 'by_task()' method.")

    @classmethod
    def construct(
        cls, task: str, resulting_name: Optional[str] = None, compute_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> MetricForTask:
        """
        Common interface for all metrics for specified MetricForTask to be constructed.

        Args:
            task: (``str``) Task name for the desired metric to obtain the subclass.
            resulting_name (Optional ``str``): Resulting name of the computed score returned. If None,
                `~._get_metric_name()` is used.
            compute_kwargs (Optional ``Dict[str, Any]``): Arguments to be passed to `compute()` method of metric at
                computation.

        Raises: :py:class:`TaskNotAvailable`

        Returns: Metric for proper task if available.
        """
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
    """
    Extension of TaskMapper which allows a single :py:class:`jury.metrics.MetricForTask` class to be aliased. If a
    metric has a single task, use this class instead of :py:class:`jury.metrics._core.TaskMapper`.

    All metrics using TaskMapper must implement _SUBCLASS attribute.
    """

    _SUBCLASS: MetricForTask = None

    @classmethod
    def construct(
        cls,
        task: str = None,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> MetricForTask:
        """
        Common interface for all metrics for specified MetricForTask to be constructed. Do not raise
        :py:class:`TaskNotAvailable` unlike :py:class:`TaskMapper` as it directly uses _SUBCLASS defined.

        Args:
            task: (Ignored ``str``) Ignored. Preserved to provide a common interface.
            resulting_name (Optional ``str``): Resulting name of the computed score returned. If None,
                `~._get_metric_name()` is used.
            compute_kwargs (Optional ``Dict[str, Any]``): Arguments to be passed to `compute()` method of metric at
                computation.

        Returns: Metric for proper task if available.
        """
        subclass = cls._get_subclass()
        resulting_name = resulting_name or cls._get_metric_name()
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_subclass(cls, *args, **kwargs) -> MetricForTask:
        return cls._SUBCLASS
