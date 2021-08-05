from typing import Dict

from jury.collator import Collator
from jury.metrics import Metric

__class_name__ = "SQUAD"

from jury.utils import NestedSingleType


class SQUAD(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = metric_name if resulting_name is None else resulting_name
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["f1"] / 100}
        return super()._postprocess(result, return_dict)

    def _preprocess(self, predictions, references):
        if NestedSingleType.get_type(predictions, order=-1) == "str":
            predictions = [{"prediction_text": pred, "id": str(i)} for i, pred in enumerate(predictions.collapse())]
        if NestedSingleType.get_type(references, order=-1) == "str":
            references = [
                {"answers": {"answer_start": [-1], "text": Collator(ref).collapse()}, "id": str(i)}
                for i, ref in enumerate(references)
            ]
        return predictions, references


if __name__ == "__main__":
    from jury import Jury

    predictions = [["1917", "November 1917"], "Albert Einstein"]
    references = [["1917", "in November 1917"], ["Einstein", "Albert Einstein"]]

    jury = Jury(metrics=[SQUAD()])
    scores = jury.evaluate(predictions, references)
    print(scores)
