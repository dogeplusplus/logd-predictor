"""Tests for molecular featurization."""

import numpy as np
import pytest

from logd_predictor.featurize import (
    GRAPH_EDGE_DIM,
    GRAPH_NODE_DIM,
    FeaturizerConfig,
    FeaturizerType,
    _atom_features,
    _bond_features,
    smiles_to_fingerprint,
    smiles_to_graph,
    smiles_to_rdkit_desc,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
INVALID = "not_a_smiles"


class TestAtomFeatures:
    def test_output_shape(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ETHANOL)
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat.shape == (GRAPH_NODE_DIM,)

    def test_dtype(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ETHANOL)
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat.dtype == np.float32

    def test_one_hot_atom_symbol(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")  # benzene - all carbons
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat[0] == 1.0  # C is index 0

    def test_unknown_atom_symbol(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("[Au]")  # gold not in symbol list
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat[12] == 1.0  # 'other' bucket at index 12

    def test_aromatic_flag(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat[31] == 1.0

    def test_formal_charge_scaled(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("[NH4+]")
        feat = _atom_features(mol.GetAtomWithIdx(0))
        assert feat[33] == pytest.approx(0.5)  # charge +1 → 1/2


class TestBondFeatures:
    def test_output_shape(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ETHANOL)
        feat = _bond_features(mol.GetBondWithIdx(0))
        assert feat.shape == (GRAPH_EDGE_DIM,)

    def test_single_bond(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(ETHANOL)
        feat = _bond_features(mol.GetBondWithIdx(0))
        assert feat[0] == 1.0  # single bond at index 0
        assert feat[1] == 0.0

    def test_aromatic_bond(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("c1ccccc1")
        feat = _bond_features(mol.GetBondWithIdx(0))
        assert feat[3] == 1.0  # aromatic at index 3


class TestSmilesToGraph:
    def test_valid_molecule(self):
        result = smiles_to_graph(ASPIRIN)
        assert result is not None
        assert "node_feats" in result
        assert "edge_feats" in result
        assert "edge_index" in result

    def test_node_feat_shape(self):
        from rdkit import Chem

        result = smiles_to_graph(ASPIRIN)
        n_atoms = Chem.MolFromSmiles(ASPIRIN).GetNumAtoms()
        assert result["node_feats"].shape == (n_atoms, GRAPH_NODE_DIM)

    def test_edge_index_is_bidirectional(self):
        result = smiles_to_graph(ETHANOL)
        from rdkit import Chem

        n_bonds = Chem.MolFromSmiles(ETHANOL).GetNumBonds()
        assert result["edge_index"].shape == (2, n_bonds * 2)

    def test_edge_index_dtype(self):
        result = smiles_to_graph(ETHANOL)
        assert result["edge_index"].dtype == np.int64

    def test_invalid_smiles_returns_none(self):
        assert smiles_to_graph(INVALID) is None

    def test_single_atom_no_bonds(self):
        result = smiles_to_graph("[Na+]")
        assert result is not None
        assert result["edge_index"].shape == (2, 0)

    def test_no_edge_feats_when_disabled(self):
        result = smiles_to_graph(ASPIRIN, use_edges=False)
        assert result is not None
        assert np.all(result["edge_feats"] == 0.0)


class TestSmilesToFingerprint:
    def test_output_shape(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR, fp_size=1024)
        fp = smiles_to_fingerprint(ASPIRIN, cfg)
        assert fp is not None
        assert fp.shape == (1024,)

    def test_dtype(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR)
        fp = smiles_to_fingerprint(ASPIRIN, cfg)
        assert fp.dtype == np.float32

    def test_invalid_smiles_returns_none(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR)
        assert smiles_to_fingerprint(INVALID, cfg) is None

    def test_different_molecules_differ(self):
        cfg = FeaturizerConfig(featurizer_type=FeaturizerType.CIRCULAR)
        fp1 = smiles_to_fingerprint(ASPIRIN, cfg)
        fp2 = smiles_to_fingerprint(ETHANOL, cfg)
        assert not np.array_equal(fp1, fp2)


class TestSmilesToRDKitDesc:
    def test_returns_array(self):
        desc = smiles_to_rdkit_desc(ASPIRIN)
        assert desc is not None
        assert isinstance(desc, np.ndarray)

    def test_all_finite(self):
        desc = smiles_to_rdkit_desc(ASPIRIN)
        assert np.all(np.isfinite(desc))

    def test_invalid_smiles_returns_none(self):
        assert smiles_to_rdkit_desc(INVALID) is None
