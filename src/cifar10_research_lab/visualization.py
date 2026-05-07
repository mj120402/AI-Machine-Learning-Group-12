from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_history_plot(history, path: Path) -> None:
    hist = history.history if hasattr(history, "history") else history
    plt = _prepare_matplotlib()

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(hist["accuracy"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="Train")
    plt.plot(epochs, hist["val_accuracy"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="Train")
    plt.plot(epochs, hist["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix_plot(matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> None:
    import numpy as np

    plt = _prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    matrix_array = np.array(matrix)
    plt.figure(figsize=(9, 8))
    plt.imshow(matrix_array, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.colorbar(fraction=0.046, pad=0.04)

    threshold = matrix_array.max() / 2 if matrix_array.size else 0
    for row in range(matrix_array.shape[0]):
        for col in range(matrix_array.shape[1]):
            value = int(matrix_array[row, col])
            color = "white" if value > threshold else "black"
            plt.text(col, row, value, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_prediction_grid(
    images,
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    probabilities,
    class_names: Sequence[str],
    path: Path,
    limit: int = 16,
) -> None:
    import numpy as np

    plt = _prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    count = min(limit, len(images))
    side = int(np.ceil(np.sqrt(count)))
    plt.figure(figsize=(side * 2.4, side * 2.6))

    for index in range(count):
        confidence = float(np.max(probabilities[index]))
        true_label = int(true_labels[index])
        predicted_label = int(predicted_labels[index])
        title_color = "green" if true_label == predicted_label else "crimson"

        plt.subplot(side, side, index + 1)
        plt.imshow(images[index])
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"P: {class_names[predicted_label]} ({confidence:.2f})\nT: {class_names[true_label]}",
            color=title_color,
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_robustness_plot(results: Sequence[Mapping[str, object]], path: Path) -> None:
    plt = _prepare_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    names = [str(item["corruption"]).replace("_", "\n") for item in results]
    accuracies = [float(item["accuracy"]) for item in results]

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(results)), accuracies, color="#2f6f9f")
    plt.xticks(range(len(results)), names, rotation=0, fontsize=8)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy Under Image Corruptions")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
