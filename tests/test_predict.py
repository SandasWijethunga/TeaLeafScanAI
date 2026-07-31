"""
Unit tests for predict.py in both the web app and the REST API app.

The real leafscan_model.keras (~16MB) is intentionally NOT committed to
version control (see docs/DVC_GUIDE.md) and is not available in CI, so
tf.keras.models.load_model is monkeypatched with a lightweight stand-in
before predict.py is imported. This lets us test the preprocessing and
post-processing logic (which is what's most likely to break) without
needing the trained weights at all.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


class DummyModel:
    """Stands in for the real Keras model. Always predicts class index 1
    ('Healthy' / 'GL') with high confidence, so assertions are deterministic."""

    def predict(self, x, verbose=0):
        return np.array([[0.02, 0.90, 0.05, 0.03]])


def _load_predict_module(app_dir_name, monkeypatch):
    """Imports predict.py from the given app folder with load_model mocked out."""
    import tensorflow as tf
    monkeypatch.setattr(tf.keras.models, "load_model", lambda path: DummyModel())

    app_dir = str(REPO_ROOT / app_dir_name)
    sys.path.insert(0, app_dir)
    sys.modules.pop("predict", None)
    try:
        module = importlib.import_module("predict")
    finally:
        sys.path.remove(app_dir)
    return module


@pytest.fixture
def web_predict(monkeypatch):
    return _load_predict_module("leafscan_flask_app", monkeypatch)


@pytest.fixture
def api_predict(monkeypatch):
    return _load_predict_module("leafscan_api_app", monkeypatch)


def _sample_image():
    return Image.new("RGB", (400, 300), color=(40, 120, 60))


def test_preprocess_image_shape_and_range(web_predict):
    arr = web_predict.preprocess_image(_sample_image())
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_web_predict_disease_returns_expected_fields(web_predict):
    result = web_predict.predict_disease(_sample_image())
    assert result["predicted_class"] == "Healthy"
    assert 0.0 <= result["confidence"] <= 100.0
    assert "treatment" in result and isinstance(result["treatment"], str)
    assert set(result["all_probabilities"].keys()) == {
        "Brown Blight", "Healthy", "Red Rust", "Red Spider Mite",
    }
    # probabilities should sum to ~100%
    assert abs(sum(result["all_probabilities"].values()) - 100.0) < 0.5


def test_api_predict_disease_returns_tuple(api_predict):
    disease, confidence = api_predict.predict_disease(_sample_image())
    assert disease == "Healthy"
    assert 0.0 <= confidence <= 100.0


def test_class_name_display_map_covers_all_classes(web_predict):
    assert set(web_predict.CLASS_NAMES) == set(web_predict.DISPLAY_NAME_MAP.keys())
