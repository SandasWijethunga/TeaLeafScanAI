# Data & Model Version Control with DVC

## Why Git alone isn't enough here

Git tracks source code well but is a poor fit for the two largest assets in this
project:

- The **TeaLeafNet dataset** — 5,000 images, several hundred MB.
- The trained **model artifact**, `leafscan_model.keras` (~16MB, and larger again
  once fine-tuned versions or future 8-class retrains exist).

Committing these directly to Git bloats the repository, makes every `clone` slow,
and Git has no concept of "this model version was produced by this exact data
version" — which matters a lot for reproducibility and debugging drift later.

**DVC (Data Version Control)** solves this by keeping Git responsible for code and
small pointer files, while the actual large files live in separate storage (e.g. an
S3 bucket, Google Drive, or a local network share) that DVC manages.

## How it would work for LeafScan AI

### 1. Initialize DVC in the repo
```bash
pip install dvc
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### 2. Track the dataset and the model with DVC instead of Git
```bash
dvc add data/raw/tealeafnet
dvc add models/leafscan_model.keras
git add data/raw/tealeafnet.dvc models/leafscan_model.keras.dvc .gitignore
git commit -m "Track TeaLeafNet dataset and trained model with DVC"
```
This replaces the large files in Git with small `.dvc` pointer files (a hash +
size), while DVC moves the actual content to a remote you configure:
```bash
dvc remote add -d storage s3://leafscan-ai-artifacts/dvc-store
dvc push
```

### 3. What this buys the project
- **Reproducibility**: anyone who clones the repo and runs `dvc pull` gets the
  *exact* dataset and model version that matches the code at that Git commit —
  not "whatever's currently in the shared drive."
- **Versioning without bloat**: `git log models/leafscan_model.keras.dvc` shows
  every model version ever produced, without the repo itself growing by 16MB per
  retrain.
- **Rollback**: if a retrained model underperforms in production, `git checkout
  <previous-commit> -- models/leafscan_model.keras.dvc && dvc checkout` restores
  the exact previous model file.
- **Linking data → model**: since both the dataset and the model artifact are
  DVC-tracked at the same commit, it's always possible to answer "which data
  produced this model?" — directly relevant to diagnosing the drift this
  project's `monitoring/drift_monitor.py` is designed to detect.

### 4. DVC pipelines (a further step, not yet implemented here)
DVC can also express the full training pipeline as reproducible stages, so
`dvc repro` re-runs only what's actually changed:

```yaml
# dvc.yaml (illustrative — mirrors notebooks/LeafScanAI_full_pipeline.ipynb)
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps: [data/raw/tealeafnet]
    outs: [data/processed]

  train:
    cmd: python src/train.py
    deps: [data/processed, src/train.py]
    params: [train.learning_rate, train.epochs]
    outs: [models/leafscan_model.keras]

  evaluate:
    cmd: python src/evaluate.py
    deps: [data/processed, models/leafscan_model.keras]
    metrics: [metrics/test_metrics.json]
```
This is a natural next step beyond the current single-notebook pipeline: it
would let CI (`.github/workflows/ci.yml`) run `dvc repro` to verify the full
pipeline still reproduces the reported metrics whenever preprocessing or
training code changes — closing the loop between version control, CI/CD, and
model development.

## Current status in this project

DVC is documented here as the recommended approach but has **not** been set up
in this repository yet — the dataset is currently fetched fresh via `kagglehub`
in the notebook each run, and the model artifact is distributed manually. This
is flagged explicitly as a limitation/next step rather than something silently
left out — see the report's MLOps section for the honest version of this note.
