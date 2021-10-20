from typing import List, Union

import numpy as np

from jury.utils import NestedSingleType


class Collator(list):
    def __init__(self, sequence, keep=False):
        sequence = self._construct(sequence, keep=keep)
        super().__init__(sequence)

    @property
    def shape(self):
        return np.array(self, dtype=object).shape

    @property
    def ndim(self):
        return len(self.shape)

    def collapse(self):
        return Collator(np.ravel(self).tolist(), keep=True)

    def nested(self):
        return Collator(self.from_list(self))

    def reshape(self, *args):
        _seq = np.array(self, dtype=object)
        if _seq.shape[:2] == (1, 1):
            return Collator(_seq.ravel().reshape(1, -1).tolist(), keep=True)
        elif _seq.ndim == 3 and _seq.shape[1] == 1:
            args = tuple(list(args) + [-1])
        return Collator(_seq.reshape(args).tolist(), keep=True)

    def reshape_len(self, *args):
        _len = len(self)
        return self.reshape(_len, *args)

    def can_collapse(self):
        if self.ndim >= 2:
            return self.shape[1] == 1
        if isinstance(self[0], list):
            n_item = len(self[0])
            return all([len(items) == n_item for items in self])
        return True

    def to_list(self, collapse=True):
        if collapse:
            return list(self.collapse())
        return list(self)

    def _construct(self, sequence: Union[str, List[str], List[List[str]]], keep: bool) -> List[List[str]]:
        if keep:
            return sequence

        _type_primary = NestedSingleType.get_type(sequence, order=0)
        _type_secondary = NestedSingleType.get_type(sequence, order=1)
        if _type_primary in ["str", "dict"]:
            sequence = self.from_str(sequence)
        elif _type_primary == "list" and _type_secondary != "list":
            sequence = self.from_list(sequence)

        return sequence

    @staticmethod
    def from_list(seq: List[str]):
        return [[item] for item in seq]

    @classmethod
    def from_str(cls, seq: str):
        return [seq]
