"""Tests for image preprocessing and prediction in model.py."""
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import UnidentifiedImageError

from conftest import TEST_IMAGES, make_image_bytes
from model import predict_result, preprocess_img


def test_preprocess_returns_normalized_batch(sample_image_path):
    """A real image becomes one float32 224x224x3 array scaled to [0, 1]."""
    arr = preprocess_img(sample_image_path)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_preprocess_upscales_small_image():
    """Images smaller than 224x224 are resized up, not rejected."""
    arr = preprocess_img(make_image_bytes("RGB", size=(10, 10)))
    assert arr.shape == (1, 224, 224, 3)


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_preprocess_converts_to_three_channels(mode):
    """Transparent and grayscale images are converted to RGB."""
    arr = preprocess_img(make_image_bytes(mode))
    assert arr.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    """Random bytes raise PIL's UnidentifiedImageError, which app.py handles."""
    with pytest.raises(UnidentifiedImageError):
        preprocess_img(io.BytesIO(b"not an image"))


def test_preprocess_missing_path_raises():
    """A path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        preprocess_img("does/not/exist.jpeg")


@patch("model.get_model")
def test_predict_result_returns_argmax_as_int(mock_get_model):
    """predict_result returns the index of the highest score as a plain int."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.05, 0.10, 0.70, 0.15]])
    mock_get_model.return_value = fake_model

    result = predict_result(np.zeros((1, 224, 224, 3), dtype="float32"))

    assert result == 2
    assert isinstance(result, int)


@pytest.mark.slow
@pytest.mark.parametrize("digit", [0, 3, 7, 9])
def test_real_model_predicts_sample_digits(digit):
    """Integration check: the trained model recognises one sample per digit."""
    image_path = sorted((TEST_IMAGES / str(digit)).glob("*.jpeg"))[0]
    assert predict_result(preprocess_img(image_path)) == digit
