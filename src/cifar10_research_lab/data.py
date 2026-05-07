from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(frozen=True)
class Cifar10Data:
    x_train: object
    y_train: object
    x_val: object
    y_val: object
    x_test: object
    y_test: object
    class_names: list[str]


def label_to_int(label: object) -> int:
    """Handle labels from lists, NumPy arrays, and TensorFlow datasets."""

    if isinstance(label, (list, tuple)):
        return int(label[0])

    shape = getattr(label, "shape", None)
    if shape:
        return int(label[0])

    return int(label)


def flatten_labels(labels: Iterable[object]) -> list[int]:
    return [label_to_int(label) for label in labels]


def limit_dataset(items, labels, max_items: int | None):
    """Keep an item/label pair aligned when using small smoke-test subsets."""

    if max_items is None:
        return items, labels

    return items[:max_items], labels[:max_items]


def choose_balanced_indices(
    labels: Sequence[object],
    samples_per_class: int,
    seed: int,
) -> list[int]:
    """Select an equal number of examples from each available class."""

    if samples_per_class <= 0:
        return []

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label_to_int(label)].append(index)

    rng = random.Random(seed)
    chosen: list[int] = []
    for class_id in sorted(grouped):
        indices = grouped[class_id][:]
        rng.shuffle(indices)
        chosen.extend(indices[: min(samples_per_class, len(indices))])

    rng.shuffle(chosen)
    return chosen


def _take_indices(values, indices: Sequence[int]):
    """Index Python lists and NumPy arrays with the same helper."""

    try:
        return values[list(indices)]
    except TypeError:
        return [values[index] for index in indices]


def stratified_train_validation_split(x_train, y_train, validation_fraction: float, seed: int):
    """Create a validation set with roughly equal representation per class."""

    labels = flatten_labels(y_train)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)

    rng = random.Random(seed)
    validation_indices: list[int] = []
    for class_id in sorted(grouped):
        indices = grouped[class_id][:]
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:count])

    validation_set = set(validation_indices)
    training_indices = [index for index in range(len(labels)) if index not in validation_set]

    rng.shuffle(training_indices)
    rng.shuffle(validation_indices)

    return (
        _take_indices(x_train, training_indices),
        _take_indices(y_train, training_indices),
        _take_indices(x_train, validation_indices),
        _take_indices(y_train, validation_indices),
    )


def load_cifar10_data(config) -> Cifar10Data:
    """Load, normalize, split, and optionally shrink CIFAR-10."""

    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # CIFAR-10 pixels are 0-255. Neural networks train better on a 0-1 range.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train, y_train, x_val, y_val = stratified_train_validation_split(
        x_train,
        y_train,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    if config.smoke_test:
        x_train, y_train = limit_dataset(x_train, y_train, config.smoke_train_size)
        x_val, y_val = limit_dataset(x_val, y_val, config.smoke_val_size)
        x_test, y_test = limit_dataset(x_test, y_test, config.smoke_test_size)

    return Cifar10Data(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=CLASS_NAMES,
    )
