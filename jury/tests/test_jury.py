from jury import Jury

TEST_METRICS = ["bleu_1", "meteor", "rouge"]


def test_evaluate_basic():
    jury = Jury(metrics=TEST_METRICS)
    predictions = ["Peace in the dormitory, peace in the world."]
    references = ["Peace at home, peace in the world."]

    scores = jury.evaluate(predictions, references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_corpus():
    predictions = [["the cat is on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]

    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(predictions, references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_multiple_items():
    predictions = [["the cat is on the mat", "There is cat playing on the mat"], ["Look! a wonderful day."]]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."],
        ["Today is a wonderful day", "The weather outside is wonderful."],
    ]
    jury = Jury(metrics=TEST_METRICS)
    scores = jury.evaluate(predictions=predictions, references=references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_preload():
    jury = Jury(metrics=TEST_METRICS, preload_metrics=True)
    predictions = ["Peace in the dormitory, peace in the world."]
    references = ["Peace at home, peace in the world."]

    scores = jury.evaluate(predictions, references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])


def test_evaluate_concurrent():
    jury = Jury(metrics=TEST_METRICS, preload_metrics=True, run_concurrent=True)
    predictions = ["Peace in the dormitory, peace in the world."]
    references = ["Peace at home, peace in the world."]

    scores = jury.evaluate(predictions, references)

    assert all([scores[metric] is not None for metric in TEST_METRICS])
