"""PyTorch Datasets and LightningDataModule for molecular graph and fingerprint data."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

from logd_predictor.featurize import FeaturizerType
from logd_predictor.models import BatchedGraph


class GraphDataset(Dataset):
    """Molecular graph dataset. All flat arrays pre-loaded as tensors for O(1) slicing."""

    def __init__(self, data_dir: str, max_samples: int | None = None) -> None:
        data = np.load(Path(data_dir) / "data.npz")
        self.node_feats = torch.from_numpy(data["node_feats"])
        self.edge_feats = torch.from_numpy(data["edge_feats"])
        self.edge_index = torch.from_numpy(data["edge_index"])
        self.graph_offsets: list[int] = data["graph_offsets"].tolist()
        self.edge_offsets: list[int] = data["edge_offsets"].tolist()
        self.n_nodes: list[int] = data["n_nodes"].tolist()
        self.n_edges: list[int] = data["n_edges"].tolist()
        self.targets = torch.from_numpy(data["targets"])
        n = len(self.targets)
        self._indices = list(range(min(max_samples, n) if max_samples else n))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        idx = self._indices[idx]
        ns = self.graph_offsets[idx]
        es = self.edge_offsets[idx]
        return (
            self.node_feats[ns : ns + self.n_nodes[idx]],
            self.edge_feats[es : es + self.n_edges[idx]],
            self.edge_index[:, es : es + self.n_edges[idx]],  # local node indices
            self.targets[idx],
        )

    @staticmethod
    def collate(batch):
        node_lists, edge_lists, ei_lists, target_list = zip(*batch)
        n_nodes = torch.tensor([nf.shape[0] for nf in node_lists], dtype=torch.long)
        n_edges = torch.tensor([ei.shape[1] for ei in ei_lists], dtype=torch.long)
        node_offsets = torch.zeros(len(n_nodes), dtype=torch.long)
        node_offsets[1:] = n_nodes[:-1].cumsum(0)
        # batch vector: each node labelled with its molecule index
        batch_ids = torch.repeat_interleave(
            torch.arange(len(node_lists), dtype=torch.long), n_nodes
        )
        # reindex edge_index: broadcast per-molecule node offset across its edges
        all_ei = torch.cat(ei_lists, dim=1)
        edge_offsets = torch.repeat_interleave(node_offsets, n_edges)
        return (
            BatchedGraph(
                node_features=torch.cat(node_lists),
                edge_features=torch.cat(edge_lists),
                edge_index=all_ei + edge_offsets.unsqueeze(0),
                batch=batch_ids,
            ),
            torch.stack(target_list),
        )


class FingerprintDataset(Dataset):
    """Fingerprint/descriptor dataset - the full X matrix lives in a single tensor."""

    def __init__(self, data_dir: str, max_samples: int | None = None) -> None:
        data = np.load(Path(data_dir) / "data.npz")
        self.X = torch.from_numpy(data["X"])
        self.targets = torch.from_numpy(data["targets"])
        n = len(self.targets)
        self._n = min(max_samples, n) if max_samples else n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return self.X[idx], self.targets[idx]


class MoleculeDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dirs: dict[str, str],
        featurizer_type: FeaturizerType,
        batch_size: int = 2048,
        num_workers: int | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dirs = dataset_dirs
        self.featurizer_type = featurizer_type
        self.batch_size = batch_size
        self.num_workers = (
            num_workers if num_workers is not None else min(os.cpu_count() or 1, 12)
        )
        self.is_graph = featurizer_type == FeaturizerType.MOL_GRAPH_CONV
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self._datasets: dict[str, Dataset] = {}

    def _max_for(self, split: str) -> int | None:
        if split == "train":
            return self.max_train_samples
        if split == "validation":
            return self.max_val_samples
        return None

    def setup(self, stage: str | None = None) -> None:
        for split in ("train", "validation", "test"):
            if split not in self._datasets:
                lim = self._max_for(split)
                self._datasets[split] = (
                    GraphDataset(self.dataset_dirs[split], max_samples=lim)
                    if self.is_graph
                    else FingerprintDataset(self.dataset_dirs[split], max_samples=lim)
                )

    def _loader(self, split: str, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=GraphDataset.collate if self.is_graph else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._loader("test")
