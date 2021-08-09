from typing import List, Tuple

from jury.collator import Collator
from jury.utils import remove_punctuations


class BLEUDefaultTokenizer:
    @staticmethod
    def _preprocess_text(text: str) -> str:
        return remove_punctuations(text).lower()

    def tokenize(self, text: str) -> List[str]:
        return self._preprocess_text(text).split()


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        _predictions = []
        _references = []
        for preds in predictions:
            _predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
        for refs in references:
            _references.append([self.tokenizer.tokenize(ref) for ref in refs])
        return Collator(_predictions, keep=True), Collator(_references, keep=True)
