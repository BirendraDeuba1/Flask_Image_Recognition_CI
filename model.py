"""Model loading, image preprocessing and prediction for the digit recognizer."""
import os
from functools import lru_cache

import numpy as np
from keras.models import load_model
from PIL import Image

# Resolve the model next to this file so the app works from any working directory.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.h5")
IMAGE_SIZE = (224, 224)


@lru_cache(maxsize=1)
def get_model():
    """Load the Keras model on first use and reuse it for later predictions."""
    return load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Open an image and convert it to a normalized (1, 224, 224, 3) array.

    ``img_path`` can be a file path or a file-like object such as an upload stream.
    Raises ``PIL.UnidentifiedImageError`` if the data is not an image.
    """
    with Image.open(img_path) as op_img:
        img_resize = op_img.resize(IMAGE_SIZE)
    img2arr = np.asarray(img_resize, dtype="float32") / 255.0
    return img2arr.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)


def predict_result(predict):
    """Return the predicted digit (0-9) for a preprocessed image array."""
    pred = get_model().predict(predict, verbose=0)
    return int(np.argmax(pred[0], axis=-1))
