from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf


def _list_images_and_labels(root_dir: str) -> tuple[list[str], list[int], list[str]]:
    root = Path(root_dir)
    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    paths: list[str] = []
    labels: list[int] = []

    for class_name in class_names:
        class_dir = root / class_name
        for fp in class_dir.rglob("*"):
            if fp.is_file() and fp.suffix.lower() in exts:
                paths.append(str(fp))
                labels.append(class_to_idx[class_name])

    if not paths:
        raise RuntimeError(f"No images found in: {root_dir}")

    return paths, labels, class_names


def _make_dataset(
    image_paths: list[str],
    label_indices: list[int],
    num_classes: int,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    augment: bool,
):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_indices))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    aug_layers = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="data_augmentation",
    )

    def _load(path, label_idx):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        if augment:
            image = aug_layers(image, training=True)
        label = tf.one_hot(label_idx, depth=num_classes)
        return image, label

    autotune = tf.data.AUTOTUNE
    ds = ds.map(_load, num_parallel_calls=autotune)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds


def build_generators(
    train_dir: str,
    val_dir: str,
    image_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    augment: bool,
):
    train_paths, train_labels, class_names = _list_images_and_labels(train_dir)
    val_paths, val_labels, val_class_names = _list_images_and_labels(val_dir)
    if class_names != val_class_names:
        raise RuntimeError("Train/validation class folders do not match.")

    num_classes = len(class_names)
    train_ds = _make_dataset(
        train_paths,
        train_labels,
        num_classes,
        image_size,
        batch_size,
        shuffle=True,
        seed=seed,
        augment=augment,
    )
    val_ds = _make_dataset(
        val_paths,
        val_labels,
        num_classes,
        image_size,
        batch_size,
        shuffle=False,
        seed=seed,
        augment=False,
    )
    return train_ds, val_ds, class_names


def build_test_generator(test_dir: str, image_size: Tuple[int, int], batch_size: int):
    test_paths, test_labels, class_names = _list_images_and_labels(test_dir)
    return _make_dataset(
        test_paths,
        test_labels,
        len(class_names),
        image_size,
        batch_size,
        shuffle=False,
        seed=42,
        augment=False,
    )


def infer_class_names(train_dir: str) -> list[str]:
    classes = [p.name for p in Path(train_dir).iterdir() if p.is_dir()]
    return sorted(classes)
