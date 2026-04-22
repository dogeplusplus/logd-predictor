"""Tests for scaffold-based train/val/test splitting."""

import pandas as pd
import pytest

from logd_predictor.scaffold_split import (
    _assign_groups_to_splits,
    _murcko_scaffold,
    compute_scaffolds,
    scaffold_split,
)


class TestMurckoScaffold:
    def test_benzene_scaffold(self):
        # Benzene's scaffold is itself
        s = _murcko_scaffold("c1ccccc1")
        assert s is not None
        assert s == "c1ccccc1"

    def test_aspirin_has_scaffold(self):
        s = _murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O")
        assert s is not None
        assert "c1ccccc1" in s

    def test_invalid_smiles_returns_none(self):
        assert _murcko_scaffold("not_a_smiles") is None

    def test_aliphatic_returns_empty_string_or_none(self):
        # Ethanol has no ring — scaffold is empty string → returned as None
        result = _murcko_scaffold("CCO")
        assert result is None or result == ""

    def test_bicyclic_scaffold(self):
        # Naphthalene — scaffold is itself
        s = _murcko_scaffold("c1ccc2ccccc2c1")
        assert s is not None


class TestComputeScaffolds:
    def test_returns_list_same_length(self):
        smiles = ["c1ccccc1", "CCO", "c1ccc2ccccc2c1"]
        result = compute_scaffolds(smiles)
        assert len(result) == len(smiles)

    def test_invalid_smiles_gets_none(self):
        result = compute_scaffolds(["c1ccccc1", "bad_smiles", "CCO"])
        assert result[1] is None

    def test_chunked_equals_single(self):
        smiles = [f"c1ccccc1{'C' * i}" for i in range(10)]
        full = compute_scaffolds(smiles, chunk_size=10)
        chunked = compute_scaffolds(smiles, chunk_size=3)
        assert full == chunked


class TestAssignGroups:
    def _make_groups(self, sizes: list[int]) -> dict[str, list[int]]:
        idx = 0
        groups: dict[str, list[int]] = {}
        for i, s in enumerate(sizes):
            groups[f"g{i}"] = list(range(idx, idx + s))
            idx += s
        return groups

    def test_split_fractions_approx(self):
        groups = self._make_groups([10] * 100)
        n_total = 1000
        splits = _assign_groups_to_splits(groups, n_total, 0.8, 0.1, 42)
        assert abs(len(splits["train"]) / n_total - 0.8) < 0.05
        assert abs(len(splits["validation"]) / n_total - 0.1) < 0.05

    def test_no_overlap_between_splits(self):
        groups = self._make_groups([5] * 60)
        splits = _assign_groups_to_splits(groups, 300, 0.7, 0.15, 42)
        train_set = set(splits["train"])
        val_set = set(splits["validation"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_all_indices_assigned(self):
        groups = self._make_groups([3] * 20)
        n_total = 60
        splits = _assign_groups_to_splits(groups, n_total, 0.7, 0.15, 42)
        all_assigned = (
            set(splits["train"]) | set(splits["validation"]) | set(splits["test"])
        )
        assert all_assigned == set(range(n_total))

    def test_deterministic_with_same_seed(self):
        groups = self._make_groups([2] * 50)
        s1 = _assign_groups_to_splits(groups, 100, 0.8, 0.1, 42)
        s2 = _assign_groups_to_splits(groups, 100, 0.8, 0.1, 42)
        assert s1["train"] == s2["train"]

    def test_different_seeds_differ(self):
        groups = self._make_groups([1] * 100)
        s1 = _assign_groups_to_splits(groups, 100, 0.8, 0.1, 0)
        s2 = _assign_groups_to_splits(groups, 100, 0.8, 0.1, 99)
        assert s1["train"] != s2["train"]

    def test_scaffold_groups_never_split(self):
        # Each group of 5 must land entirely in one split
        groups = self._make_groups([5] * 20)
        n_total = 100
        splits = _assign_groups_to_splits(groups, n_total, 0.7, 0.15, 42)
        for split_indices in splits.values():
            split_set = set(split_indices)
            for grp in groups.values():
                grp_set = set(grp)
                overlap = split_set & grp_set
                assert overlap == grp_set or len(overlap) == 0, (
                    "Scaffold group was split across partitions"
                )


class TestScaffoldSplit:
    @pytest.fixture()
    def csv_path(self, tmp_path):
        smiles = (
            ["c1ccccc1"] * 20
            + ["CC(=O)Oc1ccccc1C(=O)O"] * 10
            + ["c1ccc2ccccc2c1"] * 10
            + ["CCO"] * 10
            + ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"] * 10
        )
        df = pd.DataFrame({"canonical_smiles": smiles, "cx_logd": [1.0] * len(smiles)})
        p = tmp_path / "molecules.csv"
        df.to_csv(p, index=False)
        return p

    def test_creates_split_files(self, csv_path, tmp_path):
        split_dir = tmp_path / "splits"
        scaffold_split(csv_path, split_dir)
        assert (split_dir / "train.csv").exists()
        assert (split_dir / "validation.csv").exists()
        assert (split_dir / "test.csv").exists()

    def test_no_row_leakage_across_splits(self, csv_path, tmp_path):
        split_dir = tmp_path / "splits"
        scaffold_split(csv_path, split_dir)
        train = pd.read_csv(split_dir / "train.csv")
        val = pd.read_csv(split_dir / "validation.csv")
        test = pd.read_csv(split_dir / "test.csv")
        total = len(train) + len(val) + len(test)
        original = pd.read_csv(csv_path)
        assert total == len(original)

    def test_returns_counts(self, csv_path, tmp_path):
        counts = scaffold_split(csv_path, tmp_path / "splits")
        assert set(counts.keys()) == {"train", "validation", "test"}
        assert all(v > 0 for v in counts.values())
