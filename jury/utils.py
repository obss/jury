import os
import re
import string
from typing import Callable, Dict, List, Optional

import numpy as np


class NestedSingleType:
    @staticmethod
    def is_iterable(obj):
        if isinstance(obj, str) or isinstance(obj, dict):
            return False
        try:
            iter(obj)
        except TypeError:
            return False
        return True

    @staticmethod
    def join(types: List[str]):
        nested_types = f"{types.pop(-1)}"

        for _type in types:
            nested_types = f"{_type}<{nested_types}>"
        return nested_types.lower()

    @classmethod
    def get_type(cls, obj, order: Optional[int] = None):
        _obj = obj

        types = []
        while cls.is_iterable(_obj):
            types.append(type(_obj).__name__)
            _obj = _obj[0]
        types.append(type(_obj).__name__)
        if order is not None:
            return types[order]

        return cls.join(types)


def remove_punctuations(text: str) -> str:
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    return " ".join(text.split())


def bulk_remove_keys(obj: Dict, keys: List[str]) -> Dict:
    return {k: v for k, v in obj.items() if k not in keys}


def is_reduce_fn(fun: Callable) -> bool:
    result = np.array(fun([1, 2]))
    return result.size == 1


def set_env(name: str, value: str):
    if not isinstance(value, str):
        raise ValueError(f"Expected type str for 'value', got {type(value)}.")
    os.environ[name] = value


def replace(a: List, obj: object, index=-1):
    del a[index]
    a.insert(index, obj)
    return a
