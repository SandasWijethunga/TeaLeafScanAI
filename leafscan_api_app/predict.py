"""
predict.py — Inference logic for the LeafScan AI REST API.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("leafscan_api_predict")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "leafscan_model.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["BB", "GL", "RR", "RSM"]

DISPLAY_NAME_MAP = {
    "BB": "Brown Blight",
    "GL": "Healthy",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite",
}

logger.info("Loading model from: %s", MODEL_PATH)
_model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_disease(image: Image.Image) -> Tuple[str, float]:
    """Returns (disease_name, confidence_percentage_rounded_to_1_decimal)."""
    preprocessed = preprocess_image(image)
    probabilities = _model.predict(preprocessed, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    raw_class_name = CLASS_NAMES[predicted_index]
    disease_name = DISPLAY_NAME_MAP[raw_class_name]
    confidence = round(float(probabilities[predicted_index]) * 100.0, 1)

    return disease_name, confidence
