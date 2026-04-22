"""Load curated LogD data from the OpenADMET intake data catalog.

The OpenADMET project (https://openadmet.org) publishes curated ADMET datasets
as Parquet files on S3, accessible via intake catalog YAML files.

Two datasets are available:
  LogD_aggregated  — 23 k unique molecules; per-compound mean / median / std
                     across all ChEMBL assays.  Best choice for training.
  LogD_raw         — 26 k individual measurements; useful when you want per-
                     assay provenance or to study measurement variance.

Reference catalog:
  https://github.com/OpenADMET/data-catalogs

Column mapping used throughout this package:
  OPENADMET_CANONICAL_SMILES → canonical_smiles
  standard_value_mean        → cx_logd   (aggregated dataset)
  standard_value             → cx_logd   (raw dataset)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CATALOG_URL = (
    "https://raw.githubusercontent.com/OpenADMET/data-catalogs/main"
    "/catalogs/activities/ChEMBL_LogD/CATALOG_ChEMBL35_LogD.yaml"
)

# Sensible range for drug-like logD.  Values outside this window are almost
# certainly unit-conversion artefacts in ChEMBL (e.g. µM reported as logD).
_LOGD_MIN = -8.0
_LOGD_MAX = 8.0

_SMILES_COL = "canonical_smiles"
_TARGET_COL = "cx_logd"


def _open_catalog():
    try:
        import intake  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "intake is required to load OpenADMET data.  "
            "Install it with: pip install intake s3fs"
        ) from exc
    return intake.open_catalog(CATALOG_URL)


def load_logd_data(
    use_raw: bool = False,
    logd_min: float = _LOGD_MIN,
    logd_max: float = _LOGD_MAX,
    min_assay_count: int = 1,
) -> pd.DataFrame:
    """Load and clean the OpenADMET LogD dataset.

    Parameters
    ----------
    use_raw:
        If True, load individual measurements (``LogD_raw``) rather than the
        per-compound aggregated dataset.  Aggregated is recommended for ML.
    logd_min / logd_max:
        Filter to remove unit-conversion artefacts in ChEMBL.
    min_assay_count:
        Drop compounds measured in fewer than this many assays (aggregated
        dataset only).  Helps remove single low-confidence measurements.

    Returns
    -------
    DataFrame with columns ``canonical_smiles`` and ``cx_logd``, deduplicated
    on InChIKey with the highest assay-count entry kept.
    """
    cat = _open_catalog()

    if use_raw:
        logger.info("Loading OpenADMET LogD_raw from S3…")
        df = cat["LogD_raw"].read()
        df = df.rename(
            columns={
                "OPENADMET_CANONICAL_SMILES": _SMILES_COL,
                "standard_value": _TARGET_COL,
                "OPENADMET_INCHIKEY": "inchikey",
            }
        )
        # Keep only '=' relations (exact measurements, not '>' / '<')
        if "standard_relation" in df.columns:
            df = df[df["standard_relation"] == "="]
    else:
        logger.info("Loading OpenADMET LogD_aggregated from S3…")
        df = cat["LogD_aggregated"].read()
        df = df.rename(
            columns={
                "OPENADMET_CANONICAL_SMILES": _SMILES_COL,
                "standard_value_mean": _TARGET_COL,
                "OPENADMET_INCHIKEY": "inchikey",
            }
        )
        if min_assay_count > 1 and "assay_id_count" in df.columns:
            before = len(df)
            df = df[df["assay_id_count"] >= min_assay_count]
            logger.info(
                "Dropped %d compounds with fewer than %d assay(s)",
                before - len(df),
                min_assay_count,
            )

    n_before = len(df)
    df = df.dropna(subset=[_SMILES_COL, _TARGET_COL])
    df = df[df[_TARGET_COL].between(logd_min, logd_max)]
    logger.info(
        "Filtered %d → %d molecules (logD range [%.1f, %.1f])",
        n_before,
        len(df),
        logd_min,
        logd_max,
    )

    # Deduplicate on InChIKey, keeping highest assay-count entry.
    if "inchikey" in df.columns and "assay_id_count" in df.columns:
        df = (
            df.sort_values("assay_id_count", ascending=False)
            .drop_duplicates(subset="inchikey")
            .reset_index(drop=True)
        )
    elif "inchikey" in df.columns:
        df = df.drop_duplicates(subset="inchikey").reset_index(drop=True)
    else:
        df = df.drop_duplicates(subset=_SMILES_COL).reset_index(drop=True)

    logger.info("Final dataset: %d unique molecules", len(df))
    return df[[_SMILES_COL, _TARGET_COL]]


def prepare_splits(
    output_dir: str | Path = "data/processed",
    use_raw: bool = False,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    random_seed: int = 42,
    min_assay_count: int = 1,
) -> dict[str, int]:
    """Load OpenADMET LogD, scaffold-split, and write train/val/test CSVs.

    Returns a dict of {split_name: n_molecules}.
    """
    from logd_predictor.scaffold_split import scaffold_split

    out = Path(output_dir)
    molecules_path = out / "openadmet_logd.csv"
    split_dir = out / "splits"

    df = load_logd_data(use_raw=use_raw, min_assay_count=min_assay_count)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(molecules_path, index=False)
    logger.info("Saved %d molecules → %s", len(df), molecules_path)

    return scaffold_split(
        prepared_path=molecules_path,
        split_dir=split_dir,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
        smiles_col="canonical_smiles",
    )
