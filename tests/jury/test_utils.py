import numpy as np

from jury.utils import bulk_remove_keys, is_reduce_fn, remove_punctuations


def test_remove_punctuations():
    test_text = "!*This, a (test); text. Testing: 123 alpha123. " "Multiple    spaces     uneven sizes,and-also-this."
    resulting_text = "This a test text Testing 123 alpha123 Multiple spaces " "uneven sizes and also this"
    assert remove_punctuations(test_text) == resulting_text


def test_bulk_remove_keys():
    test_dict = {"a": 0, "b": 1, "c": 2}
    remove_keys = ["a", "b"]
    resulting_dict = {"c": 2}
    assert bulk_remove_keys(test_dict, remove_keys) == resulting_dict


def test_is_reduce_fn():
    assert not is_reduce_fn(np.exp)
    assert is_reduce_fn(np.mean)
