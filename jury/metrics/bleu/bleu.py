from typing import Any, Dict, Optional

from jury.metrics._core import MetricAlias
from jury.metrics.bleu.bleu_for_language_generation import BleuForLanguageGeneration

__main_class__ = "Bleu"

from jury.utils.common import camel_to_snake


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
        resulting_name = resulting_name or cls._get_path(compute_kwargs=compute_kwargs)
        return subclass._construct(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    @classmethod
    def _get_path(cls, compute_kwargs: Dict[str, Any] = None) -> str:
        """
        All metric modules must implement this method as it is used to form MetricOutput properly.

        Returns: Metric name.
        """
        path = camel_to_snake(cls.__name__)
        if compute_kwargs is None:
            return path

        max_order = compute_kwargs.get("max_order")
        if max_order is not None:
            return f"{path}_{max_order}"
