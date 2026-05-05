import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib

SMOKE_TEST = os.getenv("SMOKE_TEST", "0") == "1"
NO_PLOTS = os.getenv("NO_PLOTS", "0") == "1"

if NO_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Optionally, create validation set from training data
val_fraction = 0.1
num_val = int(len(x_train) * val_fraction)

x_val = x_train[:num_val]
y_val = y_train[:num_val]
x_train = x_train[num_val:]
y_train = y_train[num_val:]

if SMOKE_TEST:
    # Keep smoke test fast while still exercising train/val/test flow.
    x_train = x_train[:5000]
    y_train = y_train[:5000]
    x_val = x_val[:1000]
    y_val = y_val[:1000]
    x_test = x_test[:1000]
    y_test = y_test[:1000]

# 2. Data augmentation (helps performance and generalisation)
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)


# 3. Build a moderately deep CNN model
def create_model():
    inputs = layers.Input(shape=(32, 32, 3))

    x = data_augmentation(inputs)

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)

    # Dense head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn_deep")
    return model


model = create_model()
model.summary()

# 4. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 5. Train model
batch_size = 64
epochs = 1 if SMOKE_TEST else 30
checkpoint_path = "best_model.weights.h5"

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[checkpoint_cb],
)

# 6. Evaluate on test data
# Load the best validation checkpoint before test evaluation.
model.load_weights(checkpoint_path)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)
print("Loaded best checkpoint:", checkpoint_path)


def print_report_metrics(hist, final_test_acc, final_test_loss):
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]

    best_val_epoch_idx = int(np.argmax(val_acc))
    best_val_epoch = best_val_epoch_idx + 1

    print("\n=== REPORT METRICS ===")
    print(f"Final train accuracy: {train_acc[-1]:.4f}")
    print(f"Final validation accuracy: {val_acc[-1]:.4f}")
    print(f"Final train loss: {train_loss[-1]:.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")
    print(f"Best validation accuracy: {val_acc[best_val_epoch_idx]:.4f}")
    print(f"Best validation loss (same epoch): {val_loss[best_val_epoch_idx]:.4f}")
    print(f"Best validation epoch: {best_val_epoch}")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    print(f"Final test loss: {final_test_loss:.4f}")


print_report_metrics(history, test_acc, test_loss)


# 7. Plot training & validation accuracy/loss (for report)
def plot_history(hist):
    acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Training and Validation Loss")

    plt.tight_layout()
    plt.show()


if not NO_PLOTS:
    plot_history(history)

# 8. Example predictions (for qualitative analysis)
class_names = [
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

probabilities = model.predict(x_test[:16])
pred_labels = np.argmax(probabilities, axis=1)

if not NO_PLOTS:
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, 1 + i)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i])
        plt.xlabel(f"Pred: {class_names[pred_labels[i]]}\nTrue: {class_names[int(y_test[i])]}")
    plt.tight_layout()
    plt.show()
