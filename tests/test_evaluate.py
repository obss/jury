import unittest


class TestBasicEvaluation(unittest.TestCase):
    def test_basic_evaluate(self):
        from jury import Jury

        jury = Jury()
        predictions = ["Peace in the dormitory, peace in the world."]
        references = ["Peace at home, peace in the world."]

        scores = jury.evaluate(predictions, references)
        for metric in Jury._DEFAULT_METRICS:
            self.assertIsNotNone(scores.get(metric))


if __name__ == "__main__":
    unittest.main()
