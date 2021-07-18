from typing import List, Tuple

from jury.core import InputList
from jury.utils import remove_punctuations


class BLEUDefaultTokenizer:
    @staticmethod
    def _preprocess_text(text: str) -> str:
        return remove_punctuations(text).lower()

    def tokenize(self, text: str) -> List[str]:
        return self._preprocess_text(text).split()


class TokenizerWrapper:
    _selected_metrics = ["bleu"]

    def __init__(self, tokenizer):
        if tokenizer.__class__.__name__ == "BLEUDefaultTokenizer":
            self.is_default_tokenizer = True
        else:
            self.is_default_tokenizer = False
        self.tokenizer = tokenizer

    def tokenize(
        self, predictions: InputList, references: InputList
    ) -> Tuple[List[List[List[str]]], List[List[List[str]]]]:
        _predictions = []
        _references = []
        for preds in predictions:
            _predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
        for refs in references:
            _references.append([self.tokenizer.tokenize(ref) for ref in refs])
        return InputList(_predictions, keep=True), InputList(_references, keep=True)
