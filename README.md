# logd-predictor — molecular logD prediction

End-to-end pipeline for predicting aqueous lipophilicity (logD) from SMILES strings using graph neural networks and molecular fingerprints.

## Data

Pre-split CSVs are expected under `data/processed/splits/` with `canonical_smiles` and `cx_logd` columns. To build from scratch:

```bash
uv run chembl-scraper        # downloads ChEMBL SQLite, extracts logD data
uv run chembl-logd-pipeline  # prepares, splits, and saves CSVs
```

## Training

```bash
uv run python train.py                                        # default: AttentiveFP + MolGraphConv
uv run python train.py model=gcn featurizer=circular          # swap model/featurizer
uv run python train.py model.epochs=100 model.learning_rate=5e-4  # override hyperparams
uv run python train.py --multirun                             # Optuna sweep via Hydra
```

Available models: `attentive_fp`, `gcn`, `random_forest`
Available featurizers: `mol_graph_conv` (default), `circular`, `rdkit`

## Experiment tracking

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open [http://localhost:5000](http://localhost:5000).

## Tests

```bash
uv run pytest
```

63 tests cover featurization, model forward passes, dataset collation, and training utilities. No GPU or data files required.

## Project layout

```
logd_predictor/
  featurize.py        # SMILES → graph / fingerprint / RDKit descriptor arrays
  datasets.py         # PyTorch Datasets and LightningDataModule
  models.py           # MLPRegressor, GCNRegressor, AttentiveFPRegressor
  training.py         # LitRegressor, callbacks, train/evaluate helpers
  chembl_scraper.py   # ChEMBL data download and extraction
  scaffold_split.py   # scaffold-based train/val/test splitting
  configs.py          # Pydantic config models
conf/                 # Hydra config files (model, featurizer, eval)
train.py              # Hydra entry point
tests/                # pytest unit tests
```
