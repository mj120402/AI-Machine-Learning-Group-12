from __future__ import annotations


def _regularizer(weight_decay: float = 1e-4):
    import tensorflow as tf

    return tf.keras.regularizers.l2(weight_decay)


def create_data_augmentation(
    strong: bool = False,
    include_noise: bool = False,
    include_cutout: bool = False,
):
    """Build augmentation inside the model so it is saved with the architecture."""

    import tensorflow as tf
    from tensorflow.keras import layers

    transforms = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08 if not strong else 0.12),
        layers.RandomZoom(0.10 if not strong else 0.18),
    ]

    if strong:
        transforms.extend(
            [
                layers.RandomTranslation(0.08, 0.08),
                layers.RandomContrast(0.15),
            ]
        )

    if include_cutout:
        # Random erasing is the Keras equivalent of cutout-style masking. CIFAR-10
        # inputs are already scaled to 0-1, so erased regions use black pixels.
        transforms.append(
            layers.RandomErasing(
                factor=0.45,
                scale=(0.03, 0.18),
                fill_value=0.0,
                value_range=(0.0, 1.0),
            )
        )

    if include_noise:
        # Keep the standard deviation mild: enough to train for robustness,
        # not so much that small CIFAR-10 objects become unreadable.
        transforms.append(layers.GaussianNoise(0.05))

    return tf.keras.Sequential(transforms, name="data_augmentation")


def _conv_bn_relu(x, filters: int, kernel_size: int = 3, strides: int = 1):
    from tensorflow.keras import layers

    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_regularizer=_regularizer(),
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def _residual_block(x, filters: int, strides: int = 1):
    from tensorflow.keras import layers

    shortcut = x
    x = _conv_bn_relu(x, filters, strides=strides)
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=_regularizer(),
    )(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_regularizer=_regularizer(),
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def create_research_cnn(
    input_shape=(32, 32, 3),
    strong_augmentation: bool = False,
    noise_augmentation: bool = False,
    cutout_augmentation: bool = False,
):
    """A compact CNN baseline with modern training-friendly choices."""

    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)
    x = create_data_augmentation(
        strong=strong_augmentation,
        include_noise=noise_augmentation,
        include_cutout=cutout_augmentation,
    )(inputs)

    for filters, dropout in ((32, 0.20), (64, 0.25), (128, 0.30)):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout)(x)

    # Global average pooling keeps the classifier small and easier to explain.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(192, activation="relu", kernel_regularizer=_regularizer())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="research_cnn")


def create_wide_cnn(
    input_shape=(32, 32, 3),
    strong_augmentation: bool = False,
    noise_augmentation: bool = False,
    cutout_augmentation: bool = False,
):
    """A wider VGG-style CNN for testing whether extra capacity improves accuracy."""

    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)
    x = create_data_augmentation(
        strong=strong_augmentation,
        include_noise=noise_augmentation,
        include_cutout=cutout_augmentation,
    )(inputs)

    for filters, dropout in ((64, 0.20), (128, 0.30), (256, 0.40)):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=_regularizer())(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="wide_cnn")


def create_resnet_small(
    input_shape=(32, 32, 3),
    strong_augmentation: bool = False,
    noise_augmentation: bool = False,
    cutout_augmentation: bool = False,
):
    """A small residual model for comparing against the custom CNN."""

    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)
    x = create_data_augmentation(
        strong=strong_augmentation,
        include_noise=noise_augmentation,
        include_cutout=cutout_augmentation,
    )(inputs)
    x = _conv_bn_relu(x, 32)

    x = _residual_block(x, 32)
    x = _residual_block(x, 32)
    x = _residual_block(x, 64, strides=2)
    x = _residual_block(x, 64)
    x = _residual_block(x, 128, strides=2)
    x = _residual_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="resnet_small")


def create_model(config):
    if config.architecture == "research_cnn":
        return create_research_cnn(
            strong_augmentation=config.strong_augmentation,
            noise_augmentation=config.noise_augmentation,
            cutout_augmentation=config.cutout_augmentation,
        )
    if config.architecture == "wide_cnn":
        return create_wide_cnn(
            strong_augmentation=config.strong_augmentation,
            noise_augmentation=config.noise_augmentation,
            cutout_augmentation=config.cutout_augmentation,
        )
    if config.architecture == "resnet_small":
        return create_resnet_small(
            strong_augmentation=config.strong_augmentation,
            noise_augmentation=config.noise_augmentation,
            cutout_augmentation=config.cutout_augmentation,
        )
    raise ValueError(f"Unknown architecture: {config.architecture}")


def model_contains_layer(model, class_name: str) -> bool:
    """Search nested Keras models for a layer class name."""

    for layer in model.layers:
        if layer.__class__.__name__ == class_name:
            return True
        nested_layers = getattr(layer, "layers", [])
        if any(nested.__class__.__name__ == class_name for nested in nested_layers):
            return True
    return False
