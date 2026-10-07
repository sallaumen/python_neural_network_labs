"""Fast tests for the training workflow; no dataset download is required."""

import unittest

from lib.experiment import run_training


class FakeModel:
    def __init__(self):
        self.fit_args = None
        self.evaluate_args = None

    def fit(self, *args, **kwargs):
        self.fit_args = (args, kwargs)

    def evaluate(self, *args, **kwargs):
        self.evaluate_args = (args, kwargs)
        return 0.25, 0.75


class ExperimentTest(unittest.TestCase):
    def test_uses_separate_training_and_test_splits(self):
        model = FakeModel()
        training = ([1, 2, 3], [0, 1, 0])
        test = ([4], [1])
        result = run_training(lambda: (training, test), lambda a, b: (a, b), lambda: model, 3)

        self.assertEqual(model.fit_args, (([1, 2, 3], [0, 1, 0]), {"epochs": 3, "verbose": 2}))
        self.assertEqual(model.evaluate_args, (([4], [1]), {"verbose": 0}))
        self.assertEqual(result["training_samples"], 3)
        self.assertEqual(result["test_samples"], 1)
        self.assertEqual(result["test_accuracy"], 0.75)
        self.assertGreaterEqual(result["training_seconds"], 0)

    def test_rejects_nonpositive_epochs_before_loading_data(self):
        def unexpected_download():
            self.fail("dataset should not be loaded")

        with self.assertRaisesRegex(ValueError, "positive"):
            run_training(unexpected_download, None, None, 0)


if __name__ == "__main__":
    unittest.main()
