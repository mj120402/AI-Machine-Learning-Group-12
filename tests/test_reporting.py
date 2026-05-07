import json
import tempfile
import unittest
from pathlib import Path

from cifar10_research_lab.reporting import (
    build_report_metrics,
    save_json,
    summarize_per_class_accuracy,
)


class ReportingTests(unittest.TestCase):
    def test_build_report_metrics_tracks_best_validation_epoch(self):
        history = {
            "accuracy": [0.4, 0.6, 0.7],
            "val_accuracy": [0.35, 0.72, 0.68],
            "loss": [1.9, 1.2, 0.8],
            "val_loss": [2.0, 0.95, 1.1],
        }

        metrics = build_report_metrics(history, test_accuracy=0.66, test_loss=1.05)

        self.assertEqual(metrics["best_validation_epoch"], 2)
        self.assertEqual(metrics["best_validation_accuracy"], 0.72)
        self.assertEqual(metrics["best_validation_loss"], 0.95)
        self.assertEqual(metrics["final_test_accuracy"], 0.66)
        self.assertEqual(metrics["final_train_accuracy"], 0.7)

    def test_summarize_per_class_accuracy_returns_accuracy_and_counts(self):
        class_names = ["cat", "dog", "ship"]
        true_labels = [0, 0, 1, 1, 1, 2]
        predicted_labels = [0, 1, 1, 1, 0, 2]

        summary = summarize_per_class_accuracy(true_labels, predicted_labels, class_names)

        self.assertEqual(summary["cat"], {"correct": 1, "total": 2, "accuracy": 0.5})
        self.assertEqual(summary["dog"], {"correct": 2, "total": 3, "accuracy": 2 / 3})
        self.assertEqual(summary["ship"], {"correct": 1, "total": 1, "accuracy": 1.0})

    def test_save_json_creates_parent_directories_and_serializes_pretty_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "nested" / "metrics.json"
            save_json({"b": 2, "a": 1}, target)

            saved = target.read_text(encoding="utf-8")
            loaded = json.loads(saved)

        self.assertIn('\n  "a": 1', saved)
        self.assertEqual(loaded, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
