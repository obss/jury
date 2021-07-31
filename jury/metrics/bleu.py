from typing import Dict

import datasets

from jury.collator import Collator
from jury.metrics import Metric

__class_name__ = "Bleu"

from jury.tokenizer import BLEUDefaultTokenizer, TokenizerWrapper


class Bleu(Metric):
	def __init__(self, resulting_name: str = None, params: Dict = None):
		metric_name = self.__class__.__name__
		params = {} if params is None else params
		tokenizer = params.get("tokenizer", None)
		self.tokenizer = BLEUDefaultTokenizer() if tokenizer is None else tokenizer
		super().__init__(metric_name=metric_name, resulting_name=resulting_name, params=params)

	def _preprocess(self, predictions, references):
		tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
		predictions, references = tokenizer_wrapper.tokenize(predictions, references)
		return predictions.reshape_len(-1), references


if __name__ == "__main__":
	b = Bleu()
	predictions = ["It is a guide to action which ensures that the military always obeys the commands of the party"]
	references = ["It is a guide to action that ensures that the military will forever heed Party commands"]
	res = b.compute(predictions=Collator(predictions), references=Collator(references))
	print(res)

	# preds = [['the', 'cat', 'is', 'on', 'the', 'mat']]
	# refs = [[['the', 'cat', 'is', 'playing', 'on', 'the', 'mat'], ['the', 'cat', 'plays', 'on', 'the', 'mat']]]
	# bleu = datasets.load_metric("bleu")
	# res2 = bleu.compute(predictions=preds, references=refs)
	# print(res2)