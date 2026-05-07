import importlib.util
import unittest

from cifar10_research_lab.config import build_config
from cifar10_research_lab.models import create_model, model_contains_layer


@unittest.skipIf(importlib.util.find_spec("tensorflow") is None, "TensorFlow is not installed")
class ModelConstructionTests(unittest.TestCase):
    def test_research_cnn_can_be_constructed_with_keras_3(self):
        config = build_config(["--architecture", "research_cnn"], environ={})

        model = create_model(config)

        self.assertEqual(model.name, "research_cnn")
        self.assertEqual(model.output_shape[-1], 10)

    def test_research_cnn_adds_gaussian_noise_when_requested(self):
        config = build_config(["--noise-augmentation"], environ={})

        model = create_model(config)

        self.assertTrue(model_contains_layer(model, "GaussianNoise"))

    def test_research_cnn_adds_random_erasing_when_cutout_requested(self):
        config = build_config(["--cutout-augmentation"], environ={})

        model = create_model(config)

        self.assertTrue(model_contains_layer(model, "RandomErasing"))

    def test_wide_cnn_can_be_constructed_with_keras_3(self):
        config = build_config(["--architecture", "wide_cnn"], environ={})

        model = create_model(config)

        self.assertEqual(model.name, "wide_cnn")
        self.assertEqual(model.output_shape[-1], 10)

    def test_resnet_small_can_be_constructed_with_keras_3(self):
        config = build_config(["--architecture", "resnet_small"], environ={})

        model = create_model(config)

        self.assertEqual(model.name, "resnet_small")
        self.assertEqual(model.output_shape[-1], 10)


if __name__ == "__main__":
    unittest.main()
