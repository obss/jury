from typing import Dict

from jury.metrics._base import Metric

__class_names__ = {"bertscore": "Bertscore"}


class Bertscore(Metric):
    def __init__(self, metric_name: str = None, resulting_name: str = None, params: Dict = None):
        metric_name = self.__class__.__name__ if metric_name is None else metric_name
        resulting_name = metric_name if resulting_name is None else resulting_name
        params = {"lang": "en"} if params is None else params
        super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

    def _preprocess(self, predictions, references, fn_multiple):
        predictions = predictions.to_list()
        references = references.to_list()
        return predictions, references

    def _postprocess(self, result, return_dict):
        result = {self.metric_name: result["f1"][0]}
        return super()._postprocess(result, return_dict)


if __name__ == "__main__":
    # predictions = [
    #     ["It is a guide to action which ensures that the military always obeys the commands of the party"],
    #     ["bar foo foobar"],
    # ]
    # references = [
    #     ["It is a guide to action that ensures that the military will forever heed Party commands"],
    #     ["foo bar foobar"],
    # ]

    # Multi pred multi ref
    predictions = [
        [
            "It is a guide to action which ensures that the military always obeys the commands of the party",
            "It is a guide to action that will ensure that the military always obeys the commands of the party"
        ],
        [
            "bar foo foobar",
            "bar foo"
        ]
    ]
    references = [
        [
            "It is a guide to action that ensures that the military will forever heed Party commands",
            "It is a guide to action which ensures that the military will forever heed Party commands"
        ],
        [
            "foo bar foobar",
            "foo bar"
        ]
    ]
    sb = BERTScore()
    score = sb.compute(predictions=predictions, references=references)
    print(score)
