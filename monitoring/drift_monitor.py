"""
drift_monitor.py — Studies model drift for LeafScan AI.

WHAT "DRIFT" MEANS HERE
------------------------
Two related but different things can drift after deployment:

1. Prediction drift (a proxy for data drift): the *distribution of classes
   the model predicts* on live traffic starts to look different from the
   distribution it saw during training/testing. E.g. if the model suddenly
   predicts "Healthy" 80% of the time when training data was a balanced 25%
   per class, that's a signal something about incoming images has changed
   (different camera, different region's disease prevalence, users
   photographing the wrong thing, etc.) — even without ever knowing the
   true labels of the new images.

2. Confidence drift: the distribution of the model's own confidence scores
   shifts lower over time, suggesting it's seeing inputs less like what it
   was trained on.

Neither requires ground-truth labels for the new data, which matters
because in production you don't have labels for images as they arrive —
this is the practical, label-free version of drift monitoring.

METHOD
------
- Population Stability Index (PSI) on the predicted class distribution,
  comparing a reference distribution (the training/test class distribution)
  against a recent window of live predictions logged by
  monitoring/log_predictions.py.
- A Kolmogorov-Smirnov test on the confidence-score distributions
  (reference vs. recent) as a secondary signal.

PSI interpretation (standard industry thresholds):
    PSI < 0.1                → no significant drift
    0.1 <= PSI < 0.25        → moderate drift, worth watching
    PSI >= 0.25              → significant drift, investigate / consider retraining

USAGE
-----
    python monitoring/drift_monitor.py

Reads monitoring's reference distribution (from training) and
logs/predictions.jsonl (from monitoring/log_predictions.py), and prints a
drift report. If logs/predictions.jsonl doesn't exist yet (e.g. running
this before the app has served real traffic), a small simulated log is
generated instead so the script — and the technique — can still be
demonstrated end-to-end.
"""
import json
import random
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"

# The TeaLeafNet training set is perfectly balanced: 1,250 images per class
# out of 5,000 total (see notebooks/LeafScanAI_full_pipeline.ipynb, Section 7).
REFERENCE_CLASS_DISTRIBUTION = {
    "Brown Blight": 0.25,
    "Healthy": 0.25,
    "Red Rust": 0.25,
    "Red Spider Mite": 0.25,
}
CLASSES = list(REFERENCE_CLASS_DISTRIBUTION.keys())

# The test-set classification report (Section 13) gives a sense of the
# model's typical confidence when it's right; used here as the reference
# confidence distribution for the KS-test.
REFERENCE_CONFIDENCE_MEAN = 92.0
REFERENCE_CONFIDENCE_STD = 8.0


def compute_psi(reference: dict, current: dict, epsilon: float = 1e-4) -> float:
    """Population Stability Index between two categorical distributions."""
    psi = 0.0
    for cls in reference:
        ref_pct = max(reference.get(cls, 0.0), epsilon)
        cur_pct = max(current.get(cls, 0.0), epsilon)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi)


def load_recent_predictions(n: int = 200):
    """Loads the most recent n predictions from the JSONL log."""
    if not LOG_PATH.exists():
        return None
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[-n:]


def simulate_predictions(n: int = 200, drifted: bool = True, seed: int = 42):
    """
    Generates a synthetic prediction log so this script can be demonstrated
    without real production traffic. `drifted=True` simulates a plausible
    real-world scenario: a field deployment sees far more Healthy leaves
    than the balanced training set did, and average confidence is lower
    because field photos differ from the training distribution.
    """
    rng = random.Random(seed)
    records = []
    if drifted:
        weights = [0.10, 0.55, 0.10, 0.25]  # Healthy over-represented
        conf_mean, conf_std = 78.0, 12.0
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
        conf_mean, conf_std = REFERENCE_CONFIDENCE_MEAN, REFERENCE_CONFIDENCE_STD

    for _ in range(n):
        cls = rng.choices(CLASSES, weights=weights, k=1)[0]
        confidence = max(1.0, min(99.9, rng.gauss(conf_mean, conf_std)))
        records.append({"predicted_class": cls, "confidence": confidence})
    return records


def analyze_drift(records) -> dict:
    n = len(records)
    counts = {cls: 0 for cls in CLASSES}
    for r in records:
        counts[r["predicted_class"]] = counts.get(r["predicted_class"], 0) + 1
    current_distribution = {cls: counts[cls] / n for cls in CLASSES}

    psi = compute_psi(REFERENCE_CLASS_DISTRIBUTION, current_distribution)

    confidences = np.array([r["confidence"] for r in records])
    rng = np.random.default_rng(seed=42)
    reference_sample = rng.normal(
        REFERENCE_CONFIDENCE_MEAN, REFERENCE_CONFIDENCE_STD, size=5000
    )
    ks_stat, ks_pvalue = ks_2samp(reference_sample, confidences)

    return {
        "n_predictions": n,
        "current_class_distribution": current_distribution,
        "psi": psi,
        "psi_verdict": (
            "no significant drift" if psi < 0.1
            else "moderate drift — worth watching" if psi < 0.25
            else "significant drift — investigate / consider retraining"
        ),
        "mean_confidence": float(confidences.mean()),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "confidence_drift_flagged": bool(ks_pvalue < 0.05),
    }


def print_report(report: dict, source: str) -> None:
    print(f"\n=== LeafScan AI — Drift Report ({source}) ===")
    print(f"Predictions analyzed: {report['n_predictions']}")
    print("\nCurrent predicted-class distribution vs. training reference (25% each):")
    for cls, pct in report["current_class_distribution"].items():
        print(f"  {cls:<18} {pct*100:5.1f}%   (reference: 25.0%)")
    print(f"\nPopulation Stability Index (class distribution): {report['psi']:.4f}")
    print(f"  -> {report['psi_verdict']}")
    print(f"\nMean prediction confidence: {report['mean_confidence']:.1f}%  "
          f"(reference: ~{REFERENCE_CONFIDENCE_MEAN:.1f}%)")
    print(f"KS-test on confidence distribution: statistic={report['ks_statistic']:.3f}, "
          f"p-value={report['ks_pvalue']:.4f}")
    print(f"  -> {'confidence distribution has shifted (p < 0.05)' if report['confidence_drift_flagged'] else 'no significant shift in confidence'}")
    print()


if __name__ == "__main__":
    records = load_recent_predictions()
    if records:
        print_report(analyze_drift(records), source=str(LOG_PATH))
    else:
        print(f"No prediction log found at {LOG_PATH} yet — demonstrating with "
              f"a simulated drifted traffic sample instead.\n")
        print_report(analyze_drift(simulate_predictions(drifted=True)), source="simulated drifted traffic")
        print_report(analyze_drift(simulate_predictions(drifted=False)), source="simulated non-drifted traffic (control)")
