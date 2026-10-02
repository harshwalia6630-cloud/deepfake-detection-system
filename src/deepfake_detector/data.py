"""tf.data input pipelines for face-crop datasets."""

import csv
from pathlib import Path

LABELS = {"real": 0, "fake": 1}


def read_manifest(faces_dir, split):
    """Return (paths, labels, video_ids) for one split of data/faces/manifest.csv."""
    faces_dir = Path(faces_dir)
    paths, labels, videos = [], [], []
    with open(faces_dir / "manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            paths.append(str(faces_dir / row["path"]))
            labels.append(LABELS[row["label"]])
            # real and fake clips share numeric stems ("033" vs "033_097"); prefix keeps them distinct
            videos.append(f"{row['label']}/{row['video']}")
    return paths, labels, videos


def make_dataset(tf, paths, labels, image_size=224, batch_size=32, training=False, seed=42):
    """Decode JPEG face crops into float32 [0, 255] tensors (EfficientNet rescales internally)."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (image_size, image_size))
        return img, tf.cast(label, tf.float32)

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15 * 255)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        # mimic re-compression seen in shared videos so the model doesn't key on pristine pixels
        img = tf.image.random_jpeg_quality(tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8), 60, 100)
        return tf.cast(img, tf.float32), label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
