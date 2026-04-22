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

### Results — AttentiveFP + MolGraphConv

| Metric | Value |
|---|---|
| MAE | 0.760 |
| RMSE | 1.039 |
| R² | 0.254 |
| N molecules | 4,200 |

**Conformal coverage:** 70.2 % actual vs 90 % requested. The shortfall reflects distribution shift: the model was trained on ChEMBL logD (diverse, multi-lab measurements) and evaluated on AstraZeneca-specific assay data. The conformal guarantee is marginal — it holds exactly for samples exchangeable with the calibration set.

**Uncertainty calibration (MC Dropout):** Spearman r = 0.16 between `mc_std` and |error|. Modest overall, but the quintile table shows a monotonically increasing relationship — the highest-uncertainty quintile has mean |error| = 1.02 vs 0.58 for the lowest. This means high `mc_std` does flag harder predictions, even if the absolute scale is not well-calibrated.

### Poorly predicted compounds — chemical analysis

The worst predictions share recurring patterns:

1. **Peptide-like macrocycles (|error| up to 12.7):** SMILES encodes a single conformer; logD of cyclic peptides is strongly conformation-dependent and the model has no 3D input. MW > 1000 is also far outside the training distribution.

2. **Extreme logP outliers (|error| 5–7):** Compounds with clogP > 6 or < −4 push the model into extrapolation territory. The model conflates clogP with logD and over-predicts lipophilicity for compounds where the neutral form is strongly lipophilic but ionisation at pH 7.4 substantially reduces the measured logD.

3. **Heavily fluorinated scaffolds:** Fluorine substitution patterns introduce atypical electrostatic effects that the atom-level featuriser captures poorly — F atoms lower measured logD more than a simple lipophilicity proxy predicts.

4. **Phosphonate and amino acid analogues:** High TPSA (> 130 Å²) combined with permanent ionisation is rare in ChEMBL logD data; the model reverts toward the mean.

---

## Tests

```bash
pytest                  # runs all 4 test modules
pytest -v               # verbose output
pytest tests/test_scaffold_split.py  # specific module
```

Coverage: featurization (atom/bond features, SMILES→graph/fingerprint), dataset classes and collation, model forward passes (MLP/GCN/AttentiveFP), training utilities (metrics, model builder), scaffold splitting (Murcko, greedy bin-packing, no-leakage), and uncertainty utilities (conformal calibration, MC Dropout variance, in-memory featurization).

---

## Profiling results

Profiled on RTX 3090 (CUDA 12.x), AttentiveFP + MolGraphConv featurizer. Each measurement is the mean of 5 runs after 3 warm-up passes.

| batch_size | total ms | ms / mol | mol / s | feat % | h2d % | fwd % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.0 | 7.97 | 125 | 5 % | 34 % | 61 % |
| 100 | 24.0 | 0.240 | 4,163 | 65 % | 10 % | 25 % |
| 1,000 | 168 | 0.168 | 5,939 | 92 % | 0 % | 7 % |
| 10,000 | 1,580 | 0.158 | 6,331 | 94 % | 0 % | 6 % |

**Where time is being spent:**

- At batch size 1, forward-pass kernel launch overhead dominates (61 %) and H2D transfer is disproportionately expensive (34 %).
- From batch 100 onward, **Python-side featurisation is the bottleneck (65–94 %)**. `smiles_to_graph` runs sequentially in the main process; the GPU and H2D pipeline are largely idle.
- Throughput plateaus at ~6,300 mol/s between batch 1k and 10k because the CPU featurisation ceiling is reached.

**Optimisations for 100k+ molecules per request:**

1. **Pre-featurise and persist graph tensors.** Featurisation from SMILES is pure CPU work that scales linearly. For a service, molecules should be featurised once at write-time and stored as NPZ (or a columnar binary format). At query time only the model forward pass runs. This would eliminate the 94 % featurisation cost at large batch sizes, increasing throughput by ~15×.

2. **Parallelise featurisation with a worker pool.** The featurisation step (`smiles_to_graph`) is embarrassingly parallel and CPU-bound. Using `concurrent.futures.ProcessPoolExecutor` with `forkserver` (already done in the training featurisation pipeline) would scale with CPU count. On a 16-core machine this would reduce featurisation from ~1,500 ms to ~100 ms for 10k molecules, lifting throughput to ~50k mol/s — dominated then by GPU batch execution.

---

## Design decisions

**Featurisation strategy.** Three options are provided: MolGraphConv (default), Morgan fingerprints, and RDKit descriptors. Graph featurisation preserves local chemical environment context that fingerprints discard, at the cost of variable-size inputs and slower featurisation. For logD, where long-range intramolecular interactions (macrocycle conformation, through-space effects) matter, a graph model is the principled choice, though the results show it still struggles with large and structurally unusual compounds.

**Splitting approach.** Scaffold-based splitting (Murcko scaffolds, greedy bin-packing) is used instead of random splitting. Random splits would leak structurally similar molecules across train/test, inflating reported generalisation metrics. Scaffold splits approximate the real deployment scenario — predicting logD for novel chemical series not seen during training.

**Uncertainty quantification.** Two complementary methods are used. MC Dropout provides a cheap, per-molecule epistemic uncertainty signal with no architectural changes (dropout layers are already present for regularisation). Split conformal prediction provides a statistically rigorous coverage guarantee calibrated on the validation set; it is model-agnostic and correct by construction for in-distribution inputs. The two signals are complementary: MC Dropout reflects model confidence; conformal intervals give actionable bounds.

**Model size and capacity.** The default model (`graph_feat_size=256`, `num_layers=4` from the best Optuna trial) is relatively small. For a production service handling diverse drug-like chemistry, a larger model with more message-passing steps and higher hidden dimension would likely improve generalisation. The current architecture is a reasonable trade-off between training speed, memory, and accuracy for a fixed compute budget.

---

## What I'd do with more time

- **Better uncertainty calibration.** Replace the constant-width conformal interval with a locally-adaptive conformity score (e.g. normalised by a difficulty estimate), giving tighter intervals for easy molecules and wider intervals for hard ones. Evaluate calibration using reliability diagrams and expected calibration error.

- **Distribution shift detection.** Monitor the mean MC Dropout std across a request batch; a sudden increase flags out-of-distribution inputs before they produce silently bad predictions.

- **ONNX export for cold-start serving.** Export the trained `net` to ONNX for deployment without a PyTorch runtime dependency. The variable-size graph input requires dynamic axes, which ONNX supports.

- **Tautomer and ionisation state standardisation.** Use RDKit's `MolStandardize` (or the Dimorphite-DL ioniser) to canonicalise SMILES at pH 7.4 before featurisation. This would reduce the systematic error on the ionisable-compound failures identified above.

- **More diverse training data.** The ChEMBL logD distribution is biased toward CNS-active compounds. Augmenting with proprietary in-house data or public AstraZeneca/Novartis datasets would improve generalisation to the drug-like space the benchmark covers.
