from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


def _history_dict(history) -> dict[str, list[float]]:
    if hasattr(history, "history"):
        return history.history
    return dict(history)


def _last(values: Sequence[float]) -> float:
    return float(values[-1])


def build_report_metrics(history, test_accuracy: float, test_loss: float) -> dict[str, float | int]:
    """Turn a Keras History object into the exact metrics needed for a report."""

    hist = _history_dict(history)
    validation_accuracy = hist["val_accuracy"]
    validation_loss = hist["val_loss"]
    best_index = max(range(len(validation_accuracy)), key=lambda index: validation_accuracy[index])

    return {
        "final_train_accuracy": _last(hist["accuracy"]),
        "final_validation_accuracy": _last(validation_accuracy),
        "final_train_loss": _last(hist["loss"]),
        "final_validation_loss": _last(validation_loss),
        "best_validation_accuracy": float(validation_accuracy[best_index]),
        "best_validation_loss": float(validation_loss[best_index]),
        "best_validation_epoch": best_index + 1,
        "final_test_accuracy": float(test_accuracy),
        "final_test_loss": float(test_loss),
    }


def confusion_matrix(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    number_of_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(number_of_classes)] for _ in range(number_of_classes)]
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        matrix[int(true_label)][int(predicted_label)] += 1
    return matrix


def summarize_per_class_accuracy(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        total = 0
        correct = 0
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if int(true_label) == class_index:
                total += 1
                if int(predicted_label) == class_index:
                    correct += 1

        summary[class_name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    return summary


def save_json(data: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def format_markdown_report(
    *,
    config,
    metrics: Mapping[str, object],
    per_class_accuracy: Mapping[str, Mapping[str, object]],
    robustness_results: Sequence[Mapping[str, object]],
    artifact_paths: Mapping[str, str],
) -> str:
    lines = [
        "# CIFAR-10 Research Run",
        "",
        "## Research Question",
        "",
        (
            "How well does the selected CNN architecture generalize on CIFAR-10, "
            "and how does its performance change under class-level and robustness "
            "analysis?"
        ),
        "",
        "## Configuration",
        "",
        f"- Architecture: `{config.architecture}`",
        f"- Epochs used: `{config.epochs}` (requested `{config.requested_epochs}`)",
        f"- Batch size: `{config.batch_size}`",
        f"- Learning rate: `{config.learning_rate}`",
        f"- Validation fraction: `{config.validation_fraction}`",
        f"- Seed: `{config.seed}`",
        f"- Strong augmentation: `{config.strong_augmentation}`",
        f"- Noise augmentation: `{config.noise_augmentation}`",
        f"- Cutout augmentation: `{config.cutout_augmentation}`",
        f"- Label smoothing: `{config.label_smoothing}`",
        "",
        "## Main Metrics",
        "",
    ]

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key.replace('_', ' ').title()}: `{value:.4f}`")
        else:
            lines.append(f"- {key.replace('_', ' ').title()}: `{value}`")

    lines.extend(["", "## Per-Class Accuracy", ""])
    for class_name, values in per_class_accuracy.items():
        accuracy = float(values["accuracy"])
        correct = int(values["correct"])
        total = int(values["total"])
        lines.append(f"- {class_name}: `{accuracy:.4f}` ({correct}/{total})")

    if robustness_results:
        lines.extend(["", "## Robustness Checks", ""])
        for result in robustness_results:
            lines.append(
                "- "
                f"{result['corruption']}: accuracy `{float(result['accuracy']):.4f}`, "
                f"loss `{float(result['loss']):.4f}`"
            )

    lines.extend(["", "## Saved Artifacts", ""])
    for name, path in artifact_paths.items():
        lines.append(f"- {name}: `{path}`")

    lines.extend(
        [
            "",
            "## How To Explain This In Coursework",
            "",
            (
                "Use the training curves to discuss convergence and overfitting, the "
                "confusion matrix to discuss class-level failure modes, robustness "
                "results to test whether the model relies on fragile image cues, and "
                "Grad-CAM examples to connect model decisions back to image regions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def save_markdown_report(report: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
