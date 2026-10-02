"""CNN classifier: ImageNet-pretrained EfficientNetB0 backbone with a binary head."""


def build_model(tf, image_size=224, dropout=0.3):
    inputs = tf.keras.Input((image_size, image_size, 3), name="face")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    backbone.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(backbone.output)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    # float32 output keeps the sigmoid numerically stable under mixed precision
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="fake_prob")(x)
    return tf.keras.Model(inputs, outputs, name="deepfake_effnetb0"), backbone


def load_trained(tf, weights_path, image_size=224):
    """Rebuild the architecture and load trained weights.

    Weights-only checkpoints are used because TF 2.10 cannot serialise the
    EfficientNet normalisation constants into a full-model .h5 config.
    """
    model, _ = build_model(tf, image_size)
    model.load_weights(str(weights_path))
    return model


def unfreeze_top(backbone, n_layers):
    """Unfreeze the last `n_layers` of the backbone, keeping BatchNorm layers frozen.

    Frozen BatchNorm keeps ImageNet running statistics intact, which matters when
    fine-tuning with small batches on a 6 GB GPU.
    """
    import tensorflow as tf

    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
