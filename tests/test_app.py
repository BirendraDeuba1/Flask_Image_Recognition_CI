"""Tests for the Flask routes in app.py.

The model is mocked in these tests so they run in milliseconds; the real model
is exercised separately in test_model.py.
"""
import io
from unittest.mock import patch

from conftest import make_image_bytes


def post_file(client, data, filename):
    """POST a file to /prediction the same way the HTML form does."""
    return client.post(
        "/prediction",
        data={"file": (data, filename)},
        content_type="multipart/form-data",
    )


# ---------- Normal cases ----------

def test_home_page_loads(client):
    """GET / returns the upload page with a file input."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit" in response.data
    assert b'name="file"' in response.data


@patch("app.predict_result", return_value=3)
def test_valid_image_returns_prediction(mock_predict, client, sample_image_path):
    """A real JPEG upload is preprocessed and the predicted digit is shown."""
    with open(sample_image_path, "rb") as image_file:
        response = post_file(client, image_file, "sign3.jpeg")

    assert response.status_code == 200
    assert b">3</h2>" in response.data
    mock_predict.assert_called_once()
    # The route must hand the model a single normalized 224x224 RGB image.
    assert mock_predict.call_args.args[0].shape == (1, 224, 224, 3)


# ---------- Missing file ----------

def test_missing_file_field_returns_400(client):
    """A form submitted without a 'file' field shows a clear error."""
    response = client.post("/prediction", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No file was uploaded." in response.data


def test_empty_filename_returns_400(client):
    """Clicking submit without choosing a file sends an empty filename."""
    response = post_file(client, io.BytesIO(b""), "")
    assert response.status_code == 400
    assert b"No file was uploaded." in response.data


# ---------- Invalid or unexpected input ----------

@patch("app.predict_result")
def test_text_file_is_rejected(mock_predict, client):
    """A non-image file never reaches the model."""
    response = post_file(client, io.BytesIO(b"this is not an image"), "notes.txt")
    assert response.status_code == 400
    assert b"File cannot be processed." in response.data
    mock_predict.assert_not_called()


@patch("app.predict_result")
def test_truncated_jpeg_is_rejected(mock_predict, client, sample_image_path):
    """A corrupted (cut-off) JPEG is handled instead of crashing the app."""
    truncated = sample_image_path.read_bytes()[:500]
    response = post_file(client, io.BytesIO(truncated), "broken.jpeg")
    assert response.status_code == 400
    assert b"File cannot be processed." in response.data
    mock_predict.assert_not_called()


def test_get_on_prediction_not_allowed(client):
    """The prediction route only accepts POST."""
    assert client.get("/prediction").status_code == 405


def test_oversized_upload_rejected(client):
    """Uploads above MAX_CONTENT_LENGTH (16 MB) are refused with 413."""
    big = io.BytesIO(b"0" * (17 * 1024 * 1024))
    response = post_file(client, big, "huge.jpeg")
    assert response.status_code == 413


# ---------- Edge cases: unusual but valid images ----------

@patch("app.predict_result", return_value=5)
def test_png_with_transparency_is_accepted(mock_predict, client):
    """PNG screenshots often have an alpha channel (RGBA) and must still work."""
    response = post_file(client, make_image_bytes("RGBA"), "sign.png")
    assert response.status_code == 200
    assert b">5</h2>" in response.data
    mock_predict.assert_called_once()


@patch("app.predict_result", return_value=0)
def test_grayscale_image_is_accepted(mock_predict, client):
    """A black-and-white image is converted to 3 channels; digit 0 is still displayed."""
    response = post_file(client, make_image_bytes("L"), "gray.png")
    assert response.status_code == 200
    assert b">0</h2>" in response.data
    mock_predict.assert_called_once()
