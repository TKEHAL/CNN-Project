import numpy as np
from PIL import Image

from src.features.preprocess import load_and_prepare_image


def test_load_and_prepare_image_shape(tmp_path):
    img_path = tmp_path / "x.png"
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    out = load_and_prepare_image(img_path, (128, 128))
    assert out.shape == (1, 128, 128, 3)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
