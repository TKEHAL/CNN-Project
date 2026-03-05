from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_and_prepare_image(path: str | Path, image_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(image_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
