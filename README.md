# logd-predictor

End-to-end pipeline for predicting aqueous lipophilicity (logD at pH 7.4) from SMILES strings. Trains on experimental ChEMBL data, exposes MC Dropout and conformal prediction uncertainty, and evaluates against the TDC Lipophilicity_AstraZeneca benchmark.

---

## Setup

```bash
git clone <repo>
cd logd-predictor
uv sync                   # creates .venv and installs all dependencies
source .venv/bin/activate
```

For GPU training, ensure a CUDA-capable PyTorch wheel is installed (the lockfile targets CUDA 12.x on Linux).

---

## Data assembly

The pipeline downloads ChEMBL, extracts logD measurements, deduplicates, and produces scaffold-stratified train/val/test CSVs:

```bash
uv run chembl-scraper           # downloads ChEMBL SQLite (~3 GB) and extracts logD data
uv run chembl-logd-pipeline     # scaffold split → data/processed/splits/{train,val,test}.csv
```

Pre-split CSVs are expected at `data/processed/splits/` with columns `canonical_smiles` and `cx_logd`. The training script featurises molecules on first run and caches the result under `data/processed/datasets/`.

---

## Training

```bash
# Single run — default: AttentiveFP + MolGraphConv
python train.py

# Swap model or featurizer
python train.py model=gcn featurizer=circular
python train.py model=random_forest featurizer=circular

# Override hyperparameters
python train.py model.epochs=50 model.learning_rate=5e-4 model.batch_size=8192

# Optuna sweep (20 trials, TPE sampler)
python train.py --multirun
```

Available models: `attentive_fp`, `gcn`, `random_forest`
Available featurizers: `mol_graph_conv` (default), `circular`, `rdkit`

Training runs are logged to MLflow (`mlflow.db`) and checkpoints are saved under the Hydra output directory (e.g. `multirun/YYYY-MM-DD/HH-MM-SS/<trial>/model/best.ckpt`).

The live Textual TUI shows per-trial progress, metric charts, GPU utilisation, and throughput stats.

---

## Inference and uncertainty

```bash
# From a Hydra trial directory, CSV input:
python predict_uncertainty.py \
    --run-dir multirun/2024-01-01/12-00-00/0 \
    --input molecules.csv --smiles-col smiles \
    --output predictions.csv

# Ad-hoc SMILES, 90 % conformal intervals:
python predict_uncertainty.py \
    --run-dir multirun/... \
    --smiles "CCO" "c1ccccc1" \
    --coverage 0.9 --mc-samples 100
```

Output columns:

| Column | Description |
|---|---|
| `logd_pred` | Mean prediction (MC Dropout ensemble mean) |
| `mc_std` | Epistemic uncertainty: std across N stochastic forward passes |
| `conformal_lower/upper_Xpct` | Guaranteed-coverage prediction interval (split conformal) |
| `conformal_width` | Interval width (constant per run; determined by calibration set) |

---

## Benchmark evaluation

```bash
# Auto-downloads TDC Lipophilicity_AstraZeneca (requires pip install pytdc)
python evaluate_openadmet.py \
    --run-dir multirun/... \
    --mc-samples 100 \
    --coverage 0.9 \
    --output-dir eval_output/
```
