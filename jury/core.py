from typing import Union, List

import numpy as np

from jury.utils import NestedSingleType


class InputList(list):
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
        return InputList(np.ravel(self).tolist(), keep=True)

    def nested(self):
        return InputList(self.from_list_of_str(self))

    def reshape(self, *args):
        _seq = np.array(self, dtype=object)
        return InputList(_seq.reshape(args).tolist(), keep=True)

    def can_collapse(self):
        if self.ndim != 2:
            return False

        return self.shape[1] == 1

    def _construct(self, sequence: Union[str, List[str], List[List[str]]], keep: bool) -> List[List[str]]:
        if keep:
            return sequence

        _type = NestedSingleType.get_type(sequence)
        if _type == "str":
            sequence = self.from_str(sequence)
        elif _type == "list<str>" or _type == "list<dict>":
            sequence = self.from_list_of_str(sequence)

        return sequence

    @staticmethod
    def from_list_of_str(seq: List[str]):
        return [[item] for item in seq]

    @classmethod
    def from_str(cls, seq: str):
        return [seq]
