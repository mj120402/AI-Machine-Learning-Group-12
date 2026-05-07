from __future__ import annotations

from pathlib import Path
from typing import Sequence


def find_last_conv_layer_name(model) -> str:
    """Find the last convolutional layer for Grad-CAM without hard-coding names."""

    for layer in reversed(model.layers):
        if layer.__class__.__name__.lower().startswith("conv2d"):
            return layer.name
    raise ValueError("No Conv2D layer found for Grad-CAM")


def make_gradcam_heatmap(model, image, layer_name: str | None = None, pred_index: int | None = None):
    import numpy as np
    import tensorflow as tf

    target_layer = layer_name or find_last_conv_layer_name(model)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer).output, model.output],
    )

    image_batch = np.expand_dims(image, axis=0)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    gradients = tape.gradient(class_channel, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_grid(
    model,
    images,
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
    path: Path,
    limit: int = 8,
) -> None:
    import numpy as np
    import tensorflow as tf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    count = min(limit, len(images))

    plt.figure(figsize=(count * 2.2, 4.6))
    for index in range(count):
        image = images[index]
        heatmap = make_gradcam_heatmap(model, image, pred_index=int(predicted_labels[index]))
        heatmap = tf.image.resize(heatmap[..., np.newaxis], (image.shape[0], image.shape[1])).numpy()
        heatmap = np.squeeze(heatmap)

        plt.subplot(2, count, index + 1)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"P: {class_names[int(predicted_labels[index])]}\nT: {class_names[int(true_labels[index])]}",
            fontsize=8,
        )

        plt.subplot(2, count, count + index + 1)
        plt.imshow(image)
        plt.imshow(heatmap, cmap="jet", alpha=0.42)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
