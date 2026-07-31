"""
log_predictions.py — Lightweight prediction logging for production monitoring.

This is the "eyes" for model drift detection: every prediction the deployed
model makes gets appended as one JSON line to logs/predictions.jsonl. MLflow
(used during training) tracks *experiments*; this tracks *live inference*,
which MLflow doesn't see at all once the model is deployed.

Usage (from predict.py, after computing a prediction):

    from monitoring.log_predictions import log_prediction
    log_prediction(predicted_class="Brown Blight", confidence=91.2,
                    all_probabilities={...})

Kept deliberately dependency-free (just stdlib) so it can be dropped into
either the web app or the API app without adding requirements.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"
_lock = threading.Lock()


def log_prediction(predicted_class: str, confidence: float, all_probabilities: dict) -> None:
    """Appends one prediction record. Never raises — logging failures should
    never break an actual user-facing prediction request."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        # Logging is best-effort; a full disk shouldn't take the API down.
        pass
