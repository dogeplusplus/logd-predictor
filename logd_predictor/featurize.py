import concurrent.futures
import json
import logging
import multiprocessing
import os
from enum import Enum
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from logd_predictor._io import console

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


class FeaturizerType(str, Enum):
    MOL_GRAPH_CONV = "MolGraphConv"
    CIRCULAR = "CircularFingerprint"
    RDKIT = "RDKitDescriptors"


class FeaturizerConfig(BaseModel):
    featurizer_type: FeaturizerType = FeaturizerType.MOL_GRAPH_CONV
    radius: int = 2
    fp_size: int = 2048
    use_edges: bool = True


_ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]
_HYBRIDS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]

# Node: 13 (atom+other) + 7 (degree+other) + 6 (hybrid+other) + 5 (nH+other)
#       + 1 (aromatic) + 1 (ring) + 1 (charge) = 34
# Edge: 4 (bond type) + 1 (conjugated) + 1 (ring) + 6 (stereo) = 12
GRAPH_NODE_DIM = 34
GRAPH_EDGE_DIM = 12

_DESC_FNS = [fn for _, fn in Descriptors.descList]

_ATOM_SYMBOL_IDX = {s: i for i, s in enumerate(_ATOM_SYMBOLS)}
_HYBRID_IDX = {h: i for i, h in enumerate(_HYBRIDS)}
_BOND_TYPE_IDX = {t: i for i, t in enumerate(_BOND_TYPES)}
_STEREO_IDX = {s: i for i, s in enumerate(_STEREO_TYPES)}


def _atom_features(atom: Chem.rdchem.Atom) -> np.ndarray:
    out = np.zeros(GRAPH_NODE_DIM, dtype=np.float32)
    out[_ATOM_SYMBOL_IDX.get(atom.GetSymbol(), 12)] = 1.0
    deg = atom.GetDegree()
    out[13 + (deg if deg < 6 else 6)] = 1.0
    out[20 + _HYBRID_IDX.get(atom.GetHybridization(), 5)] = 1.0
    nh = atom.GetTotalNumHs()
    out[26 + (nh if nh < 4 else 4)] = 1.0
    out[31] = float(atom.GetIsAromatic())
    out[32] = float(atom.IsInRing())
    out[33] = float(max(-2, min(2, atom.GetFormalCharge()))) / 2.0
    return out


def _bond_features(bond: Chem.rdchem.Bond) -> np.ndarray:
    out = np.zeros(GRAPH_EDGE_DIM, dtype=np.float32)
    bt_idx = _BOND_TYPE_IDX.get(bond.GetBondType())
    if bt_idx is not None:
        out[bt_idx] = 1.0
    out[4] = float(bond.GetIsConjugated())
    out[5] = float(bond.IsInRing())
    st_idx = _STEREO_IDX.get(bond.GetStereo())
    if st_idx is not None:
        out[6 + st_idx] = 1.0
    return out


def smiles_to_graph(smiles: str, use_edges: bool = True) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_atoms = mol.GetNumAtoms()
    node_feats = np.empty((n_atoms, GRAPH_NODE_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        node_feats[i] = _atom_features(atom)
    n_bonds = mol.GetNumBonds()
    if n_bonds == 0:
        return {
            "node_feats": node_feats,
            "edge_feats": np.zeros((0, GRAPH_EDGE_DIM), dtype=np.float32),
            "edge_index": np.zeros((2, 0), dtype=np.int64),
        }
    n_edges = n_bonds * 2
    edge_index = np.empty((2, n_edges), dtype=np.int64)
    edge_feats = np.zeros((n_edges, GRAPH_EDGE_DIM), dtype=np.float32)
    for k, bond in enumerate(mol.GetBonds()):
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0, 2 * k] = i
        edge_index[1, 2 * k] = j
        edge_index[0, 2 * k + 1] = j
        edge_index[1, 2 * k + 1] = i
        if use_edges:
            bf = _bond_features(bond)
            edge_feats[2 * k] = bf
            edge_feats[2 * k + 1] = bf
    return {
        "node_feats": node_feats,
        "edge_feats": edge_feats,
        "edge_index": edge_index,
    }


def smiles_to_fingerprint(smiles: str, cfg: FeaturizerConfig) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (
        GetMorganGenerator(radius=cfg.radius, fpSize=cfg.fp_size)
        .GetFingerprintAsNumPy(mol)
        .astype(np.float32)
    )


def smiles_to_rdkit_desc(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = []
    for fn in _DESC_FNS:
        try:
            v = float(fn(mol))
        except Exception:
            v = 0.0
        vals.append(0.0 if not np.isfinite(v) else v)
    return np.array(vals, dtype=np.float32)


CHUNK_SIZE = 2048


def _run_parallel(chunks: list[list[str]], fn, progress: Progress, task_id) -> list:
    ordered: list = [None] * len(chunks)
    n_workers = min(os.cpu_count() or 1, len(chunks))
    # Use forkserver instead of the default fork start method.  fork copies the
    # parent's address space including any locks held by background threads
    # (e.g. the Textual asyncio event loop), causing workers to deadlock.
    # forkserver spawns a clean helper process before threads exist, so workers
    # are uncontaminated.
    ctx = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as ex:
        futures = {ex.submit(fn, chunk): i for i, chunk in enumerate(chunks)}
        for fut in concurrent.futures.as_completed(futures):
            ordered[futures[fut]] = fut.result()
            progress.advance(task_id)
    return [item for chunk_result in ordered for item in chunk_result]


def _featurize_graph_chunk(
    chunk: list[str], use_edges: bool = True
) -> list[dict | None]:
    return [smiles_to_graph(s, use_edges) for s in chunk]


def _featurize_fp_chunk(
    chunk: list[str], cfg: FeaturizerConfig
) -> list[np.ndarray | None]:
    if cfg.featurizer_type == FeaturizerType.CIRCULAR:
        return [smiles_to_fingerprint(s, cfg) for s in chunk]
    return [smiles_to_rdkit_desc(s) for s in chunk]


def save_graph_dataset(
    smiles: list[str],
    targets: np.ndarray,
    cfg: FeaturizerConfig,
    out_dir: Path,
    progress: Progress,
    task_id,
) -> None:
    chunks = [smiles[i : i + CHUNK_SIZE] for i in range(0, len(smiles), CHUNK_SIZE)]
    results = _run_parallel(
        chunks,
        partial(_featurize_graph_chunk, use_edges=cfg.use_edges),
        progress,
        task_id,
    )

    valid = [(i, g) for i, g in enumerate(results) if g is not None]
    n_skipped = len(results) - len(valid)
    if n_skipped:
        logger.warning("Skipped %d molecules that failed featurization", n_skipped)
    if not valid:
        raise ValueError("All molecules failed graph featurization")

    idxs, graphs = zip(*valid)
    n_nodes = np.array([g["node_feats"].shape[0] for g in graphs], dtype=np.int64)
    n_edges = np.array([g["edge_index"].shape[1] for g in graphs], dtype=np.int64)

    np.savez(
        out_dir / "data.npz",
        node_feats=np.concatenate([g["node_feats"] for g in graphs], axis=0),
        edge_feats=np.concatenate([g["edge_feats"] for g in graphs], axis=0)
        if n_edges.sum() > 0
        else np.zeros((0, GRAPH_EDGE_DIM), dtype=np.float32),
        edge_index=np.concatenate([g["edge_index"] for g in graphs], axis=1)
        if n_edges.sum() > 0
        else np.zeros((2, 0), dtype=np.int64),
        graph_offsets=np.concatenate([[0], np.cumsum(n_nodes[:-1])]),
        edge_offsets=np.concatenate([[0], np.cumsum(n_edges[:-1])]),
        n_nodes=n_nodes,
        n_edges=n_edges,
        targets=targets[list(idxs)].astype(np.float32),
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "featurizer_type": "MolGraphConv",
                "n_mols": len(graphs),
                "node_feat_dim": GRAPH_NODE_DIM,
                "edge_feat_dim": GRAPH_EDGE_DIM,
            },
            indent=2,
        )
    )


def save_fp_dataset(
    smiles: list[str],
    targets: np.ndarray,
    cfg: FeaturizerConfig,
    out_dir: Path,
    progress: Progress,
    task_id,
) -> None:
    chunks = [smiles[i : i + CHUNK_SIZE] for i in range(0, len(smiles), CHUNK_SIZE)]
    results = _run_parallel(
        chunks, partial(_featurize_fp_chunk, cfg=cfg), progress, task_id
    )

    valid = [(i, fp) for i, fp in enumerate(results) if fp is not None]
    n_skipped = len(results) - len(valid)
    if n_skipped:
        logger.warning("Skipped %d molecules that failed featurization", n_skipped)
    if not valid:
        raise ValueError("All molecules failed fingerprint featurization")

    idxs, fps = zip(*valid)
    X = np.stack(fps, axis=0)
    np.savez(out_dir / "data.npz", X=X, targets=targets[list(idxs)].astype(np.float32))
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "featurizer_type": cfg.featurizer_type.value,
                "n_mols": len(fps),
                "feat_dim": int(X.shape[1]),
            },
            indent=2,
        )
    )


def load_split(
    path: Path, smiles_col: str, target_col: str, max_samples: int | None = None
) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path, nrows=max_samples).dropna(subset=[smiles_col, target_col])
    if max_samples:
        df = df.head(max_samples)
    return df[smiles_col].tolist(), df[target_col].values.astype(float)


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(spinner_name="dots", style="bright_blue"),
        TextColumn("[bold]{task.description}"),
        BarColumn(
            bar_width=36,
            style="blue3",
            complete_style="bright_blue",
            finished_style="green",
        ),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def featurize_to_disk(
    split_dir: str,
    cfg: FeaturizerConfig,
    dataset_dir: str,
    smiles_col: str = "canonical_smiles",
    target_col: str = "cx_logd",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
) -> dict[str, str]:
    """Featurize all splits and save npz files. Fast-paths if data already exists."""
    out_root = Path(dataset_dir)
    split_names = ("train", "validation", "test")
    existing = {s: str(out_root / s) for s in split_names}

    if all((Path(p) / "metadata.json").exists() for p in existing.values()):
        console.print(
            "[bold green]✓[/] Datasets found on disk - skipping featurization\n"
        )
        return existing

    split_limits = {
        "train": max_train_samples,
        "validation": max_val_samples,
        "test": None,
    }
    splits_data = {
        s: load_split(
            Path(split_dir) / f"{s}.csv", smiles_col, target_col, split_limits[s]
        )
        for s in split_names
    }
    is_graph = cfg.featurizer_type == FeaturizerType.MOL_GRAPH_CONV
    progress = _make_progress()
    dataset_dirs: dict[str, str] = {}

    with progress:
        for split, (smiles, targets) in splits_data.items():
            n_chunks = max(1, (len(smiles) + CHUNK_SIZE - 1) // CHUNK_SIZE)
            task_id = progress.add_task(
                f"[cyan]{split:<12}[/] {len(smiles):>7,} molecules", total=n_chunks
            )
            split_out = out_root / split
            split_out.mkdir(parents=True, exist_ok=True)
            if is_graph:
                save_graph_dataset(smiles, targets, cfg, split_out, progress, task_id)
            else:
                save_fp_dataset(smiles, targets, cfg, split_out, progress, task_id)
            dataset_dirs[split] = str(split_out)

    total = sum(len(v[0]) for v in splits_data.values())
    console.print(
        f"[bold green]✓[/] Featurization complete - {total:,} molecules → [dim]{out_root}[/]\n"
    )
    return dataset_dirs
