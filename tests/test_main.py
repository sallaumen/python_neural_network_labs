"""CLI tests that do not import TensorFlow or download datasets."""

import contextlib
import io
import unittest
from unittest.mock import patch

from lib.main import main


class MainTest(unittest.TestCase):
    def test_dispatches_selected_dataset_and_epochs(self):
        metrics = {
            "training_seconds": 1.25,
            "test_loss": 0.5,
            "test_accuracy": 0.75,
        }
        output = io.StringIO()
        with patch("lib.implementations.MNIST.mnist.run", return_value=metrics) as run:
            with contextlib.redirect_stdout(output):
                result = main(["mnist", "--epochs", "2"])

        run.assert_called_once_with(epochs=2)
        self.assertIs(result, metrics)
        self.assertIn("Training time: 1.250 s (2 epochs)", output.getvalue())
        self.assertIn("Test accuracy: 75.00%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
