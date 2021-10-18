from typing import Any, Dict, Optional

from jury.metrics._core import MetricAlias
from jury.metrics.bleu.bleu_for_language_generation import BleuForLanguageGeneration

__main_class__ = "Bleu"


class Bleu(MetricAlias):
    _SUBCLASS = BleuForLanguageGeneration

    @classmethod
    def construct(
        cls,
        task: str = None,
        resulting_name: Optional[str] = None,
        compute_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        subclass = cls._get_subclass()
        resulting_name = resulting_name or cls._get_metric_name(compute_kwargs=compute_kwargs)
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_metric_name(cls, compute_kwargs: Dict[str, Any] = None) -> str:
        """
        All metric modules must implement this method as it is used to form MetricOutput properly.

        Returns: Metric name.
        """
        metric_name = cls.__name__.lower()
        if compute_kwargs is None:
            return metric_name

        max_order = compute_kwargs.get("max_order")
        if max_order is not None:
            return f"{metric_name}_{max_order}"
