import re
import string
from copy import deepcopy
from typing import List, Dict


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
    def get_type(cls, obj):
        _obj = obj

        types = []
        while cls.is_iterable(_obj):
            types.append(type(_obj).__name__)
            _obj = deepcopy(_obj[0])
        types.append(type(_obj).__name__)
        return cls.join(types)


def remove_punctuations(text: str) -> str:
    regex = re.compile("[%s]" % re.escape(string.punctuation))
    text = regex.sub(" ", text)
    return " ".join(text.split())


def bulk_remove_keys(obj: Dict, keys: List[str]) -> Dict:
    return {k: v for k, v in obj.items() if k not in keys}
