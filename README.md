# LeafScan AI — Tea Leaf Disease Detection (MLOps Course Work 2)

Deep learning tea leaf disease classifier (4 classes: Brown Blight, Healthy,
Red Rust, Red Spider Mite), built with EfficientNetB0 transfer learning and
deployed as a Flask web app + REST API, with MLflow experiment tracking,
Docker containerization, GitHub Actions CI/CD, and label-free production
drift monitoring.

**Full report:** [`notebooks/CourseWork2_MLOps_Report.ipynb`](notebooks/CourseWork2_MLOps_Report.ipynb)
**Training pipeline:** `notebooks/LeafScanAI_full_pipeline.ipynb` *(add your existing copy here)*

## Quickstart

```bash
# 1. Place the trained model (not committed to Git — see docs/DVC_GUIDE.md)
cp /path/to/leafscan_model.keras leafscan_flask_app/models/leafscan_model.keras
cp /path/to/leafscan_model.keras leafscan_api_app/models/leafscan_model.keras

# 2. Run everything
docker compose up --build
```
- Web app: http://localhost:5001
- REST API: http://localhost:8001/predict (`POST`, multipart field `image`)
- API health check: http://localhost:8001/health

## Repository structure

```
leafscan-ai/
├── leafscan_flask_app/         # Web application (Flask + Bootstrap UI)
│   ├── app.py / predict.py
│   ├── templates/
│   └── Dockerfile
├── leafscan_api_app/           # REST API (Flask, JSON in/out)
│   ├── app.py / predict.py
│   └── Dockerfile
├── notebooks/
│   ├── LeafScanAI_full_pipeline.ipynb   # data → training → evaluation
│   └── CourseWork2_MLOps_Report.ipynb   # the graded report (all rubric sections)
├── tests/                      # pytest suite, run automatically by CI
│   ├── test_predict.py
│   └── test_app.py
├── monitoring/
│   ├── log_predictions.py      # logs every live prediction to logs/predictions.jsonl
│   └── drift_monitor.py        # PSI + KS-test drift detection over that log
├── docs/
│   └── DVC_GUIDE.md            # data/model version control explanation
├── .github/workflows/ci.yml    # test → docker build pipeline
└── docker-compose.yml          # runs both apps together
```

## Running tests locally

```bash
pip install -r leafscan_flask_app/requirements.txt pytest
pytest tests/ -v
```

## Checking for model drift

```bash
pip install scipy numpy
python monitoring/drift_monitor.py
```
Reads `logs/predictions.jsonl` if it exists (populated by
`monitoring/log_predictions.py` from live traffic); otherwise runs a
simulated demo so the technique is visible without needing real traffic yet.

## Known limitations

See Section 1.5 and 3.3.2 of `notebooks/CourseWork2_MLOps_Report.ipynb` for
the full, honest list — most importantly: the training/test data has
artificially removed backgrounds (not yet validated on natural field photos),
and DVC is documented but not yet initialized in this repo.
