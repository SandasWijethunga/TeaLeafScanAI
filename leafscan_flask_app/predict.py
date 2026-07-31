"""
predict.py — Inference logic for LeafScan AI.

Loads the trained model once at import time and exposes predict_disease(),
used by both the web app and (in a near-identical copy) the REST API.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
from PIL import Image
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("leafscan_predict")

MODEL_PATH = Path(__file__).resolve().parent / "models" / "leafscan_model.keras"
IMAGE_SIZE = (224, 224)

# Folder-code order MUST match the sorted class order used during training
# (alphabetical: BB, GL, RR, RSM).
CLASS_NAMES = ["BB", "GL", "RR", "RSM"]

# Confirmed from the TeaLeafNet dataset's own "About this directory" description
DISPLAY_NAME_MAP = {
    "BB": "Brown Blight",
    "GL": "Healthy",
    "RR": "Red Rust",
    "RSM": "Red Spider Mite",
}

# General treatment/management guidance per class. Advisory only, not a
# substitute for professional agricultural advice.
TREATMENT_SUGGESTIONS: Dict[str, str] = {
    "BB": (
        "Prune and destroy affected shoots and leaves to limit spread. "
        "Ensure good field drainage and avoid excess leaf wetness. "
        "Fungicidal treatment combined with balanced fertilization is "
        "commonly recommended — consult a local agricultural extension "
        "officer for approved products."
    ),
    "GL": (
        "No signs of disease detected. Continue regular monitoring, "
        "maintain balanced fertilization, and ensure proper irrigation and "
        "spacing to keep the crop healthy."
    ),
    "RR": (
        "Improve field drainage and reduce excess moisture on foliage. "
        "Remove and destroy severely affected leaves. Copper-based "
        "fungicidal sprays are commonly used for management — confirm "
        "dosage and approved products with a local expert."
    ),
    "RSM": (
        "This is a pest (mite) infestation rather than a fungal disease. "
        "Consider miticide/acaricide treatment, and encourage natural "
        "predatory mites where possible. Avoid excess dust and drought "
        "stress on plants, which favor mite population growth. Consult a "
        "local agricultural extension officer for approved treatments."
    ),
}

logger.info("Loading model from: %s", MODEL_PATH)
_model = tf.keras.models.load_model(MODEL_PATH)
logger.info("Model loaded successfully.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Converts to RGB, resizes to 224x224, normalizes to [0,1], adds batch dim."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_disease(image: Image.Image) -> Dict[str, Any]:
    """
    Runs inference on a single PIL image.

    Returns a dict with predicted_class (display name), confidence (%),
    treatment suggestion, and the full per-class probability breakdown.
    """
    preprocessed = preprocess_image(image)
    probabilities = _model.predict(preprocessed, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    raw_class_name = CLASS_NAMES[predicted_index]
    predicted_class = DISPLAY_NAME_MAP[raw_class_name]
    confidence = float(probabilities[predicted_index]) * 100.0

    all_probabilities = {
        DISPLAY_NAME_MAP[CLASS_NAMES[i]]: round(float(probabilities[i]) * 100.0, 2)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "treatment": TREATMENT_SUGGESTIONS[raw_class_name],
        "all_probabilities": all_probabilities,
    }
