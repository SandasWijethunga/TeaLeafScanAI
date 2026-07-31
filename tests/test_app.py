"""
Route-level tests for both Flask apps. predict_disease() is mocked at the
module level, so these tests never touch TensorFlow or the real model —
they check request validation, status codes, and response shape only.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_app_module(app_dir_name, monkeypatch, fake_predict):
    """Imports app.py with predict.predict_disease patched out first."""
    import types
    fake_predict_module = types.ModuleType("predict")
    fake_predict_module.predict_disease = fake_predict
    monkeypatch.setitem(sys.modules, "predict", fake_predict_module)

    app_dir = str(REPO_ROOT / app_dir_name)
    sys.path.insert(0, app_dir)
    sys.modules.pop("app", None)
    try:
        module = importlib.import_module("app")
    finally:
        sys.path.remove(app_dir)
    return module


# ---------------------------------------------------------------- API app --

@pytest.fixture
def api_client(monkeypatch):
    fake = lambda image: ("Healthy", 91.5)
    module = _load_app_module("leafscan_api_app", monkeypatch, fake)
    module.app.config["TESTING"] = True
    with module.app.test_client() as client:
        yield client


def test_api_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_api_predict_missing_image(api_client):
    resp = api_client.post("/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_predict_rejects_bad_extension(api_client):
    data = {"image": (Path(__file__).open("rb"), "notes.txt")}
    resp = api_client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400


def test_api_predict_success(api_client, tmp_path):
    from PIL import Image
    import io
    img = Image.new("RGB", (50, 50), color=(10, 200, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    data = {"image": (buf, "leaf.jpg")}
    resp = api_client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["disease"] == "Healthy"
    assert body["confidence"] == 91.5


# ------------------------------------------------------------ Web app -----

@pytest.fixture
def web_client(monkeypatch):
    fake = lambda image: {
        "predicted_class": "Healthy",
        "confidence": 91.5,
        "treatment": "Continue regular monitoring.",
        "all_probabilities": {
            "Brown Blight": 2.0, "Healthy": 91.5, "Red Rust": 3.5, "Red Spider Mite": 3.0,
        },
    }
    module = _load_app_module("leafscan_flask_app", monkeypatch, fake)
    module.app.config["TESTING"] = True
    with module.app.test_client() as client:
        yield client


def test_web_index_loads(web_client):
    resp = web_client.get("/")
    assert resp.status_code == 200


def test_web_predict_no_file_redirects(web_client):
    resp = web_client.post("/predict", data={})
    assert resp.status_code in (302, 400)
