import importlib.util
import unittest

from cifar10_research_lab.config import build_config
from cifar10_research_lab.train import create_loss, prepare_labels_for_model


@unittest.skipIf(importlib.util.find_spec("tensorflow") is None, "TensorFlow is not installed")
class TrainingOptionTests(unittest.TestCase):
    def test_create_loss_uses_sparse_loss_by_default(self):
        config = build_config([], environ={})

        loss = create_loss(config)

        self.assertEqual(loss.name, "sparse_categorical_crossentropy")

    def test_create_loss_uses_categorical_label_smoothing_when_requested(self):
        config = build_config(["--label-smoothing", "0.1"], environ={})

        loss = create_loss(config)

        self.assertEqual(loss.name, "categorical_crossentropy")
        self.assertEqual(loss.label_smoothing, 0.1)

    def test_prepare_labels_for_model_keeps_sparse_labels_by_default(self):
        config = build_config([], environ={})
        labels = [[1], [3]]

        prepared = prepare_labels_for_model(labels, config)

        self.assertIs(prepared, labels)

    def test_prepare_labels_for_model_one_hot_encodes_for_label_smoothing(self):
        config = build_config(["--label-smoothing", "0.1"], environ={})

        prepared = prepare_labels_for_model([[1], [3]], config)

        self.assertEqual(prepared.shape, (2, 10))
        self.assertEqual(prepared[0][1], 1.0)
        self.assertEqual(prepared[1][3], 1.0)


if __name__ == "__main__":
    unittest.main()
