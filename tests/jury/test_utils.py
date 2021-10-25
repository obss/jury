import numpy as np
import pytest

from jury.metrics._core.utils import is_reduce_fn
from jury.utils import bulk_remove_keys, remove_punctuations


@pytest.fixture
def raw_text():
    return '!*This\'s, a (test); text. Testing: 123 alpha123.  Multiple    spaces     "uneven sizes", and-also-this.'


@pytest.fixture
def preprocessed_text():
    return "This s a test text Testing 123 alpha123 Multiple spaces uneven sizes and also this"


def test_remove_punctuations(raw_text, preprocessed_text):
    assert remove_punctuations(raw_text) == preprocessed_text


def test_bulk_remove_keys():
    test_dict = {"a": 0, "b": 1, "c": 2}
    remove_keys = ["a", "b"]
    resulting_dict = {"c": 2}
    assert bulk_remove_keys(test_dict, remove_keys) == resulting_dict


def test_is_reduce_fn():
    assert not is_reduce_fn(np.exp)
    assert is_reduce_fn(np.mean)
