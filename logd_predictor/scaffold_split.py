import logging
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

CHUNK_SIZE = 50_000


def _murcko_scaffold(smiles: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold or None
    except Exception:
        return None


def compute_scaffolds(
    smiles_list: list[str], chunk_size: int = CHUNK_SIZE
) -> list[str | None]:
    """Compute Murcko scaffolds in chunks to keep peak memory flat."""
    scaffolds: list[str | None] = []
    total = len(smiles_list)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        scaffolds.extend(_murcko_scaffold(s) for s in smiles_list[start:end])
        logger.info("Scaffolds: %d / %d", end, total)
    return scaffolds


def _assign_groups_to_splits(
    scaffold_groups: dict[str, list[int]],
    n_total: int,
    train_fraction: float,
    validation_fraction: float,
    random_seed: int,
) -> dict[str, list[int]]:
    """Greedy bin-packing: largest scaffold groups assigned first.

    Equal-size groups are shuffled deterministically to remove ordering bias.
    """
    sorted_groups = sorted(scaffold_groups.values(), key=len, reverse=True)

    rng = random.Random(random_seed)
    shuffled: list[list[int]] = []
    i = 0
    while i < len(sorted_groups):
        j = i
        group_len = len(sorted_groups[i])
        while j < len(sorted_groups) and len(sorted_groups[j]) == group_len:
            j += 1
        same_size = sorted_groups[i:j]
        rng.shuffle(same_size)
        shuffled.extend(same_size)
        i = j

    train_target = int(n_total * train_fraction)
    val_target = int(n_total * validation_fraction)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for group in shuffled:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def scaffold_split(
    prepared_path: Path,
    split_dir: Path,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    random_seed: int = 42,
    smiles_col: str = "canonical_smiles",
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(prepared_path, low_memory=False)
    n = len(df)

    logger.info("Computing scaffolds for %d molecules ...", n)
    raw_scaffolds = compute_scaffolds(df[smiles_col].tolist(), chunk_size=chunk_size)

    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for pos, scaffold in enumerate(raw_scaffolds):
        key = scaffold if scaffold else f"__singleton_{pos}"
        scaffold_groups[key].append(pos)

    logger.info("%d scaffold groups for %d molecules", len(scaffold_groups), n)

    splits = _assign_groups_to_splits(
        scaffold_groups, n, train_fraction, validation_fraction, random_seed
    )

    cols = list(df.columns)
    for split_name, indices in splits.items():
        df.iloc[indices].to_csv(
            split_dir / f"{split_name}.csv", index=False, columns=cols
        )
        logger.info("  %s: %d rows", split_name, len(indices))

    counts = {k: len(v) for k, v in splits.items()}
    logger.info("Scaffold split complete: %s", counts)
    return counts
