"""Shared pytest fixtures for the Flask hand-sign digit app."""
import io
from pathlib import Path

import pytest
from PIL import Image

from app import app as flask_app

TEST_IMAGES = Path(__file__).resolve().parent.parent / "test_images"


@pytest.fixture
def client():
    """Flask test client with testing mode enabled."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_image_path():
    """Path to a real hand-sign image of the digit 3 from the project's test set."""
    return TEST_IMAGES / "3" / "Sign 3 (30).jpeg"


def make_image_bytes(mode="RGB", size=(64, 64), fmt="PNG"):
    """Build an in-memory image file so tests do not depend on files on disk."""
    color = {"RGB": (120, 80, 40), "RGBA": (120, 80, 40, 200), "L": 128}[mode]
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt)
    buffer.seek(0)
    return buffer
