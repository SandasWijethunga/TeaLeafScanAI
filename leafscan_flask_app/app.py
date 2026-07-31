"""
app.py — LeafScan AI Flask web application.

Routes:
    GET  /         Homepage with the image upload form.
    POST /predict  Accepts an uploaded image, runs inference, shows the result.
"""

import logging
import uuid
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError

from predict import predict_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("leafscan_app")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = "replace-this-with-a-real-secret-key-in-production"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        flash("No file selected. Please choose a tea leaf image to upload.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a tea leaf image to upload.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Please upload a JPG or PNG image.")
        return redirect(url_for("index"))

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    saved_path = Path(app.config["UPLOAD_FOLDER"]) / unique_name
    file.save(saved_path)

    try:
        image = Image.open(saved_path)
        result = predict_disease(image)
    except UnidentifiedImageError:
        flash("The uploaded file could not be read as a valid image. Please try again.")
        return redirect(url_for("index"))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        flash("Something went wrong while analyzing the image. Please try again.")
        return redirect(url_for("index"))

    image_url = url_for("static", filename=f"uploads/{unique_name}")

    return render_template(
        "result.html",
        image_url=image_url,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        treatment=result["treatment"],
        all_probabilities=result["all_probabilities"],
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
