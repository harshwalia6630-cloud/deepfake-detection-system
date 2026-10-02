"""Train the deepfake face classifier in two phases.

Phase 1 trains only the classification head on top of a frozen EfficientNetB0.
Phase 2 unfreezes the top of the backbone and fine-tunes at a lower learning rate.
The checkpoint with the best validation AUC is kept.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector import env  # noqa: E402

tf = env.setup()
from deepfake_detector.data import make_dataset, read_manifest  # noqa: E402
from deepfake_detector.model import build_model, unfreeze_top  # noqa: E402


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--faces", default=str(ROOT / "data/faces"))
    ap.add_argument("--out", default=str(ROOT / "models/deepfake_effnetb0.weights.h5"))
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=12)
    ap.add_argument("--unfreeze-layers", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    print("GPUs:", tf.config.list_physical_devices("GPU") or "none (training on CPU)")
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tr_paths, tr_labels, _ = read_manifest(args.faces, "train")
    va_paths, va_labels, _ = read_manifest(args.faces, "val")
    print(f"train: {len(tr_paths)} faces ({sum(tr_labels)} fake) | val: {len(va_paths)} faces ({sum(va_labels)} fake)")

    train_ds = make_dataset(tf, tr_paths, tr_labels, args.image_size, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(tf, va_paths, va_labels, args.image_size, args.batch_size)

    n_fake = sum(tr_labels)
    n_real = len(tr_labels) - n_fake
    class_weight = {0: len(tr_labels) / (2 * n_real), 1: len(tr_labels) / (2 * n_fake)}

    model, backbone = build_model(tf, args.image_size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out), monitor="val_auc", mode="max", save_best_only=True,
                                           save_weights_only=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(ROOT / "reports/training_log.csv"), append=True),
    ]

    print("\n== Phase 1: training head on frozen backbone")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs,
                   class_weight=class_weight, callbacks=callbacks)

    print(f"\n== Phase 2: fine-tuning top {args.unfreeze_layers} backbone layers")
    unfreeze_top(backbone, args.unfreeze_layers)
    compile_model(model, 1e-4)
    h2 = model.fit(
        train_ds, validation_data=val_ds,
        initial_epoch=args.head_epochs, epochs=args.head_epochs + args.finetune_epochs,
        class_weight=class_weight,
        callbacks=callbacks + [
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3, patience=2, verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        ],
    )

    history = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
    (ROOT / "reports/history.json").write_text(json.dumps(history, indent=2))
    best = max(history["val_auc"])
    print(f"\nBest val AUC: {best:.4f}  ->  {out}")


if __name__ == "__main__":
    main()
