from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import chembl_downloader
import pandas as pd
from pydantic import BaseModel, Field
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from logd_predictor.scaffold_split import scaffold_split

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


class PipelineConfig(BaseModel):
    """All tuneable parameters for the ChEMBL data pipeline."""

    chembl_version: str = "35"
    standard_types: list[str] = Field(default_factory=lambda: ["logD", "logD7.4"])
    sample_size: int = None
    batch_size: int = 100_000

    fp_radius: int = 2
    fp_size: int = 1024

    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    random_seed: int = 42

    output_dir: Path = Path("data/processed")
    smiles_col: str = "canonical_smiles"
    scaffold_chunk_size: int = 50_000

    model_config = {"arbitrary_types_allowed": True}

    @property
    def fp_dir(self) -> Path:
        return self.output_dir / "fingerprints"

    @property
    def split_dir(self) -> Path:
        return self.output_dir / "splits"

    @property
    def molecules_path(self) -> Path:
        return self.output_dir / "sampled_molecules.csv"


def download_molecules(chembl_version: str = "35") -> pd.DataFrame:
    """Fetch all ChEMBL compound structures with physicochemical properties."""
    sql = """
    SELECT cs.canonical_smiles, cp.*
    FROM compound_structures cs
    JOIN compound_properties cp ON cs.molregno = cp.molregno
    WHERE cs.canonical_smiles IS NOT NULL
    """
    df = chembl_downloader.query(sql, version=chembl_version)
    logger.info("Downloaded %d molecules", len(df))
    return df


def sample_molecules(df: pd.DataFrame, n: int, random_seed: int = 42) -> pd.DataFrame:
    """Random subsample. Returns the full dataframe when n >= len(df)."""
    if n and n < len(df):
        df = df.sample(n, random_state=random_seed)
    return df.reset_index(drop=True)


def generate_fingerprints(
    smiles_list: list[str],
    fp_dir: Path,
    batch_size: int,
    radius: int = 2,
    fp_size: int = 1024,
) -> int:
    """Generate Morgan fingerprints for all molecules, writing one JSON file per batch."""
    mg = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for batch_idx, start in enumerate(range(0, len(smiles_list), batch_size)):
        batch = smiles_list[start : start + batch_size]
        fps: list[str] = []
        for smi in batch:
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp_bytes = DataStructs.BitVectToBinaryText(mg.GetFingerprint(mol))
            fps.append(base64.b64encode(fp_bytes).decode("ascii"))

        (fp_dir / f"batch_{batch_idx:05d}.json").write_text(json.dumps(fps))
        total_written += len(fps)
        logger.info(
            "Fingerprints: %d / %d molecules (%d fps written)",
            min(start + batch_size, len(smiles_list)),
            len(smiles_list),
            total_written,
        )

    return total_written


def run_data_pipeline(config: PipelineConfig | None = None) -> dict[str, Path]:
    """Download ChEMBL molecules → fingerprints → scaffold split."""
    cfg = config or PipelineConfig()

    molecules = download_molecules(chembl_version=cfg.chembl_version)
    sampled = sample_molecules(
        molecules, n=cfg.sample_size, random_seed=cfg.random_seed
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(cfg.molecules_path, index=False)

    smiles = sampled[cfg.smiles_col].tolist()
    n_fps = generate_fingerprints(
        smiles_list=smiles,
        fp_dir=cfg.fp_dir,
        batch_size=cfg.batch_size,
        radius=cfg.fp_radius,
        fp_size=cfg.fp_size,
    )
    logger.info("Generated %d fingerprints total", n_fps)

    split_counts = scaffold_split(
        prepared_path=cfg.molecules_path,
        split_dir=cfg.split_dir,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
        random_seed=cfg.random_seed,
        smiles_col=cfg.smiles_col,
        chunk_size=cfg.scaffold_chunk_size,
    )
    logger.info("Split counts: %s", split_counts)

    return {"fingerprints": cfg.fp_dir, "splits": cfg.split_dir}


def main() -> None:
    run_data_pipeline()


if __name__ == "__main__":
    main()
