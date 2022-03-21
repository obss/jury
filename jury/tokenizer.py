from typing import List, Tuple

from jury.collator import Collator
from jury.utils.nlp import normalize_text


class DefaultTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return normalize_text(text).split()


class TokenizerWrapper:
    """
    Wraps the tokenizer object to adapt tokenize method such that it returns
    a tuple of jury.Collator object instead of list.

    Args:
        tokenizer: Tokenizer object that implements `tokenize` method.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        _predictions = []
        _references = []
        i = 0
        while i < len(predictions):
            preds = predictions[i]
            refs = references[i]
            _predictions.append([self.tokenizer.tokenize(pred) for pred in preds])
            _references.append([self.tokenizer.tokenize(ref) for ref in refs])
            i += 1
        return Collator(_predictions, keep=True), Collator(_references, keep=True)
