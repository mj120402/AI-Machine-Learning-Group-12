from __future__ import annotations

from pathlib import Path

from .config import build_config
from .data import flatten_labels, load_cifar10_data
from .models import create_model
from .reporting import (
    build_report_metrics,
    confusion_matrix,
    format_markdown_report,
    save_json,
    save_markdown_report,
    summarize_per_class_accuracy,
)
from .robustness import evaluate_robustness
from .visualization import (
    save_confusion_matrix_plot,
    save_history_plot,
    save_per_class_accuracy_plot,
    save_prediction_grid,
    save_robustness_plot,
)


def _compile_model(model, config):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=create_loss(config),
        metrics=["accuracy"],
    )
    return model


def create_loss(config):
    """Choose sparse labels by default, or one-hot labels for label smoothing."""

    import tensorflow as tf

    if config.label_smoothing:
        return tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=config.label_smoothing,
            name="categorical_crossentropy",
        )

    return tf.keras.losses.SparseCategoricalCrossentropy(
        name="sparse_categorical_crossentropy",
    )


def prepare_labels_for_model(labels, config):
    if not config.label_smoothing:
        return labels

    import tensorflow as tf

    return tf.keras.utils.to_categorical(labels, num_classes=10)


def _callbacks(config):
    import tensorflow as tf

    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config.architecture}_best.weights.h5"

    return checkpoint_path, [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(config.output_dir / "training_log.csv")),
    ]


def _artifact_paths(config, checkpoint_path: Path) -> dict[str, Path]:
    return {
        "checkpoint": checkpoint_path,
        "metrics_json": config.output_dir / "metrics.json",
        "per_class_json": config.output_dir / "per_class_accuracy.json",
        "confusion_matrix_json": config.output_dir / "confusion_matrix.json",
        "robustness_json": config.output_dir / "robustness.json",
        "report_markdown": config.output_dir / "research_report.md",
        "history_plot": config.output_dir / "plots" / "training_curves.png",
        "confusion_matrix_plot": config.output_dir / "plots" / "confusion_matrix.png",
        "per_class_accuracy_plot": config.output_dir / "plots" / "per_class_accuracy.png",
        "prediction_grid": config.output_dir / "plots" / "prediction_examples.png",
        "robustness_plot": config.output_dir / "plots" / "robustness.png",
        "gradcam_grid": config.output_dir / "plots" / "gradcam_examples.png",
    }


def run_experiment(config):
    import numpy as np
    import tensorflow as tf

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.seed)

    data = load_cifar10_data(config)
    model = _compile_model(create_model(config), config)
    model.summary()

    checkpoint_path, callbacks = _callbacks(config)
    paths = _artifact_paths(config, checkpoint_path)

    history = model.fit(
        data.x_train,
        prepare_labels_for_model(data.y_train, config),
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(data.x_val, prepare_labels_for_model(data.y_val, config)),
        callbacks=callbacks,
    )

    # The report should use the best validation model, not just the last epoch.
    model.load_weights(checkpoint_path)
    test_loss, test_accuracy = model.evaluate(
        data.x_test,
        prepare_labels_for_model(data.y_test, config),
        verbose=2,
    )

    probabilities = model.predict(data.x_test, batch_size=config.batch_size, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1).tolist()
    true_labels = flatten_labels(data.y_test)

    metrics = build_report_metrics(history, test_accuracy=test_accuracy, test_loss=test_loss)
    per_class = summarize_per_class_accuracy(true_labels, predicted_labels, data.class_names)
    matrix = confusion_matrix(true_labels, predicted_labels, len(data.class_names))
    robustness_results = []

    save_json(config.to_json_dict(), config.output_dir / "config.json")
    save_json(metrics, paths["metrics_json"])
    save_json(per_class, paths["per_class_json"])
    save_json({"matrix": matrix, "class_names": data.class_names}, paths["confusion_matrix_json"])

    if not config.skip_robustness:
        robustness_results = evaluate_robustness(
            model,
            data.x_test,
            prepare_labels_for_model(data.y_test, config),
            sample_count=min(config.robustness_samples, len(data.x_test)),
            seed=config.seed,
        )
        save_json({"results": robustness_results}, paths["robustness_json"])

    if not config.no_plots:
        save_history_plot(history, paths["history_plot"])
        save_confusion_matrix_plot(matrix, data.class_names, paths["confusion_matrix_plot"])
        save_per_class_accuracy_plot(per_class, paths["per_class_accuracy_plot"])
        save_prediction_grid(
            data.x_test,
            true_labels,
            predicted_labels,
            probabilities,
            data.class_names,
            paths["prediction_grid"],
        )
        if robustness_results:
            save_robustness_plot(robustness_results, paths["robustness_plot"])

        if not config.skip_gradcam:
            from .interpretability import save_gradcam_grid

            save_gradcam_grid(
                model,
                data.x_test,
                true_labels,
                predicted_labels,
                data.class_names,
                paths["gradcam_grid"],
            )

    artifact_strings = {
        name: str(path)
        for name, path in paths.items()
        if path.exists() or name in {"checkpoint", "report_markdown"}
    }
    report = format_markdown_report(
        config=config,
        metrics=metrics,
        per_class_accuracy=per_class,
        robustness_results=robustness_results,
        artifact_paths=artifact_strings,
    )
    save_markdown_report(report, paths["report_markdown"])

    print("\n=== REPORT METRICS ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved research artifacts to: {config.output_dir}")
    return {
        "config": config,
        "metrics": metrics,
        "per_class_accuracy": per_class,
        "robustness": robustness_results,
        "artifacts": artifact_strings,
    }


def main(argv: list[str] | None = None) -> int:
    config = build_config(argv)
    run_experiment(config)
    return 0
