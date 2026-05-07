import os
import tempfile
import unittest
from pathlib import Path

from cifar10_research_lab.config import build_config, parse_bool


class ConfigTests(unittest.TestCase):
    def test_parse_bool_accepts_common_truthy_and_falsey_values(self):
        for value in ("1", "true", "TRUE", "yes", "on", "y"):
            self.assertTrue(parse_bool(value))

        for value in ("0", "false", "FALSE", "no", "off", "n", "", None):
            self.assertFalse(parse_bool(value))

    def test_build_config_combines_environment_and_cli_arguments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = build_config(
                [
                    "--architecture",
                    "resnet_small",
                    "--epochs",
                    "12",
                    "--output-dir",
                    tmpdir,
                    "--no-plots",
                ],
                environ={"SMOKE_TEST": "1", "NO_PLOTS": "0"},
            )

        self.assertEqual(config.architecture, "resnet_small")
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.requested_epochs, 12)
        self.assertEqual(config.output_dir, Path(tmpdir))
        self.assertTrue(config.no_plots)
        self.assertTrue(config.smoke_test)

    def test_build_config_defaults_are_coursework_friendly(self):
        config = build_config([], environ={})

        self.assertEqual(config.architecture, "research_cnn")
        self.assertEqual(config.epochs, 30)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.validation_fraction, 0.1)
        self.assertEqual(config.seed, 42)
        self.assertFalse(config.no_plots)
        self.assertFalse(config.smoke_test)
        self.assertFalse(config.noise_augmentation)
        self.assertEqual(config.label_smoothing, 0.0)

    def test_build_config_accepts_robustness_training_options(self):
        config = build_config(
            [
                "--noise-augmentation",
                "--cutout-augmentation",
                "--label-smoothing",
                "0.1",
            ],
            environ={},
        )

        self.assertTrue(config.noise_augmentation)
        self.assertTrue(config.cutout_augmentation)
        self.assertEqual(config.label_smoothing, 0.1)

    def test_build_config_accepts_wide_cnn_architecture(self):
        config = build_config(["--architecture", "wide_cnn"], environ={})

        self.assertEqual(config.architecture, "wide_cnn")

    def test_build_config_rejects_invalid_label_smoothing(self):
        with self.assertRaises(ValueError):
            build_config(["--label-smoothing", "0.8"], environ={})


if __name__ == "__main__":
    unittest.main()
