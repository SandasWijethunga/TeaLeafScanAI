"""
app.py — LeafScan AI REST API (Flask).

Endpoint:
    POST /predict
        Input:  multipart/form-data with an image file under the key "image"
        Output: JSON, e.g. {"disease": "Gray Blight", "confidence": 98.4}
"""

import logging

from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError

from predict import predict_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("leafscan_api")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    """Checks whether the uploaded file has an accepted image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts an uploaded tea leaf image and returns the predicted disease
    and confidence as JSON.

    Expected request: multipart/form-data with the image under key "image".
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No image file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use JPG or PNG."}), 400

    try:
        image = Image.open(file.stream)
        disease, confidence = predict_disease(image)
    except UnidentifiedImageError:
        return jsonify({"error": "Uploaded file could not be read as a valid image."}), 400
    except Exception as exc:  # noqa: BLE001 — never leak internals to API consumers
        logger.exception("Prediction failed: %s", exc)
        return jsonify({"error": "An internal error occurred while processing the image."}), 500

    return jsonify({
        "disease": disease,
        "confidence": confidence,
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """
    Basic health check endpoint — not part of your spec, but included since
    it's near-zero cost and is what your CI/CD pipeline and deployment
    monitoring will want to hit to confirm the API is alive.
    """
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
