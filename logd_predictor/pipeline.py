"""Entry points for data preparation pipelines.

Two sources are supported:

  openadmet (default)
    Loads curated ChEMBL LogD data from the OpenADMET intake catalog
    (https://github.com/OpenADMET/data-catalogs).  Requires ``intake``
    and ``s3fs``: pip install intake s3fs

  chembl
    Downloads the full ChEMBL SQLite database and extracts logD records
    directly.  Slower and requires more disk space, but gives access to
    the raw measurement metadata.

Called via:
    uv run chembl-logd-pipeline                  # OpenADMET (default)
    uv run chembl-logd-pipeline --source chembl  # raw ChEMBL download
    uv run chembl-logd-pipeline --min-assays 3   # require ≥3 assays/compound
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare LogD train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        choices=["openadmet", "chembl"],
        default="openadmet",
        help="Data source: 'openadmet' (intake catalog) or 'chembl' (raw download).",
    )
    p.add_argument(
        "--output-dir",
        default="data/processed",
        help="Root directory for output CSVs.",
    )
    p.add_argument(
        "--use-raw",
        action="store_true",
        help="[OpenADMET] Use LogD_raw instead of LogD_aggregated.",
    )
    p.add_argument(
        "--min-assays",
        type=int,
        default=1,
        help="[OpenADMET] Minimum assay count per compound.",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
    )
    p.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _run_openadmet(args: argparse.Namespace) -> None:
    from logd_predictor.openadmet import prepare_splits

    counts = prepare_splits(
        output_dir=args.output_dir,
        use_raw=args.use_raw,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
        min_assay_count=args.min_assays,
    )
    logger.info("Scaffold split complete: %s", counts)


def _run_chembl(args: argparse.Namespace) -> None:
    from logd_predictor.chembl_scraper import PipelineConfig, run_data_pipeline
    from pathlib import Path

    cfg = PipelineConfig(
        output_dir=Path(args.output_dir),
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
    )
    run_data_pipeline(cfg)


def main() -> None:
    args = _parse_args()
    if args.source == "openadmet":
        _run_openadmet(args)
    else:
        _run_chembl(args)


if __name__ == "__main__":
    main()
