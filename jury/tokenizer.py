from typing import Union, List, Tuple

from jury.utils import remove_punctuations


class BLEUDefaultTokenizer:
	@staticmethod
	def tokenize(
			predictions: Union[List[str], List[List[str]]], references: List[str]
	) -> Tuple[List[List[str]], List[List[List[str]]]]:
		if isinstance(predictions[0], str):
			predictions = [remove_punctuations(pred).split() for pred in predictions]
		else:
			_predictions = []
			for preds in predictions:
				_predictions.append([remove_punctuations(pred).split() for pred in preds])
			predictions = _predictions
		references = [[remove_punctuations(ref).split()] for ref in references]
		return predictions, references


class TokenizerWrapper:
	_selected_metrics = [
		"bleu"
	]

	def __init__(self, tokenizer):
		if tokenizer.__class__.__name__ == "BLEUDefaultTokenizer":
			self.is_default_tokenizer = True
		else:
			self.is_default_tokenizer = False
		self.tokenizer = tokenizer

	def _tokenize(self, predictions, references):
		if isinstance(predictions[0], str):
			predictions = [self.tokenizer.tokenize(pred) for pred in predictions]
		else:
			_predictions = []
			for preds in predictions:
				_predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
			predictions = _predictions
		references = [[self.tokenizer.tokenize(ref)] for ref in references]
		return predictions, references

	def tokenize(self, predictions, references) -> Tuple[List[List[str]], List[List[List[str]]]]:
		if self.is_default_tokenizer:
			return self.tokenizer.tokenize(predictions, references)
		else:
			return self._tokenize(predictions, references)
